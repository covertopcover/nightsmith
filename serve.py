"""The model server nightsmith runs: the pinned mlx_lm server, with its HTTP
edges made honest for clients.

Nightsmith embeds this file and runs it as

    python3 serve.py --served-model-id <repo> [--drain-seconds N] <mlx_lm flags>

Everything that generates text is the pinned server, untouched. What changes
is what a client sees at the edges, where the stock server misleads it:

- A request with no "messages" hit a bare assert, and nothing caught it: the
  connection closed with no HTTP response at all, which a client cannot tell
  from the server being down. Same for any bad parameter type. Now: a 400
  that names the field.
- /health said "ok" whenever the generation thread was alive — including
  while the weights were still loading. Now: loading, ready or failed.
- /v1/models advertised the model's snapshot path on this Mac, and the repo
  id was rejected. Now: the repo id is advertised and accepted.
- Any model id other than the one being served made the server unload the
  served model BEFORE trying to load the other one. A misspelled id threw away
  6.8 GB of loaded weights, and the next good request paid for reloading them.
  Now: an unknown id is refused before anything is unloaded. draft_model and
  adapters reach the same code, and are refused for the same reason.
- response_format was accepted and ignored: a request for JSON, or for a
  schema, got a 200 and whatever the model wrote, fences and all. Nothing
  here constrains the output, so asking for it is now a 400 that says so.
- A stop (SIGTERM) killed the process mid-answer, and every client waiting
  on it got a closed connection. Now: requests already in progress finish
  (up to DRAIN_SECONDS), new ones get a 503 with Retry-After, and /health
  says "stopping".
- A bad Content-Length or a body that is not JSON answered {"error": "text"},
  the one reply in the whole API that was not in the OpenAI error shape. A
  client reading err["error"]["message"] threw on exactly those paths. Now
  every error, on every path, has the same shape.
- Too much work at once killed the generation thread outright, and the Metal
  allocator takes the whole process with it. Measured: four requests of 4,000
  tokens at once is fatal, and so is one prompt of ~58,000. Now the server
  admits what it has been measured to survive and makes everything else wait
  — or, past the ceiling for a single prompt, refuses it with a 400 before a
  byte of it reaches the model. This is the same reasoning nightsmith already
  applies to a model's weights, applied to context.
- When the generation thread died anyway, the process stayed up as a zombie
  answering 404 "generation thread died" to everything, for ever. Now it says
  so in the log and exits, so the state is one a person can see and `start`
  can fix.

/health is a hint for clients. Nightsmith itself never trusts it: every
checkmark it prints still comes from a real completion.

The pure checks at the top import nothing from mlx, so they can be tested with
any Python. The patching below them depends on the exact pinned version, and
refuses to run against any other.
"""

import json
import os
import sys

PINNED_MLX_LM = "0.32.0"

CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")
TEXT_PATHS = ("/v1/completions",)
DEFAULT = "default_model"
KNOWN_ROLES = frozenset(("system", "user", "assistant", "tool"))

# How long a stop waits for requests in progress. Set from --drain-seconds,
# which nightsmith derives from max_tokens and the measured writing speed: the
# fixed 30 s this used to be killed any answer longer than ~360 tokens.
DRAIN_SECONDS = 30

# ── What this Mac will take at once ─────────────────────────────────────────
#
# Measured on a base M4 / 16 GB, gemma-4-12B-it-4bit, restarting between runs:
#
#   one prompt   43,389 tokens answered in 406 s · ~58,000 killed the server
#   at once      3 clients x 8,000 tokens fine · 4 x 4,000 fatal (twice) ·
#                5 x 2,000 fatal
#
# So the number of parallel streams dominates, not the combined size: 24,000
# tokens across three requests is fine while 10,000 across five is not. The
# failure is not a slow answer — the Metal allocator fails, the generation
# thread dies, and nothing on the machine recovers it.
#
# Deliberately not settings. Every number here was measured on one machine,
# and a setting whose right value is unknown everywhere else is not
# documentation, it is a trap.
MAX_CONCURRENT = 3
CONCURRENT_PROMPT_TOKENS = 24_000
MAX_PROMPT_TOKENS = 40_000

# How long a request waits for a slot before it is turned away. Long, because
# waiting is the point: at 12.6 tokens/s an answer takes tens of seconds, and
# a client that gets a 503 while three are running would only retry into the
# same queue.
ADMIT_WAIT_SECONDS = 120

# A body this size is not a prompt, and reading it into memory to find that
# out is itself the problem. The real limit is MAX_PROMPT_TOKENS below.
MAX_BODY_BYTES = 32 * 1024 * 1024

# What the log says when the generation thread has died. `nightsmith status`
# looks for exactly this to explain a server that is no longer there, so the
# two must stay in step (see statusFromLog in server.go).
FATAL_MARKER = "nightsmith-fatal:"


class ClientError(Exception):
    def __init__(self, status, message, param=None, code=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.retry_after = retry_after


def model_aliases(repo, model_path):
    """Every id that means "the model being served"."""
    ids = [DEFAULT]
    if repo:
        ids.append(repo)
    if model_path:
        ids.append(model_path)
        resolved = os.path.realpath(model_path)
        if resolved != model_path:
            ids.append(resolved)
    return ids


def parse_body(raw):
    """The request body, or a ClientError describing why it isn't one.

    The stock server answers both of these with {"error": "some text"} — the
    only replies in the API that are not in the OpenAI error shape, so a
    client reading err["error"]["message"] throws on exactly the paths it was
    written to handle.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ClientError(400, f"the request body is not valid UTF-8: {e}")
    try:
        body = json.loads(text)
    except ValueError as e:
        raise ClientError(400, f"the request body is not valid JSON: {e}")
    if not isinstance(body, dict):
        raise ClientError(400, "the request body must be a JSON object")
    return body


def check_request(path, body, aliases, advertised):
    """Validate a completion request before any of it reaches the model.

    Returns None when the request may proceed, else a ClientError to send.
    `aliases` are the ids accepted for the served model; `advertised` is the
    one to name in error messages.
    """
    if path in CHAT_PATHS:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return ClientError(
                400,
                "'messages' is required: a non-empty list of "
                '{"role": ..., "content": ...} objects',
                param="messages",
            )
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or not isinstance(msg.get("role"), str):
                return ClientError(
                    400,
                    f"messages[{i}] must be an object with a string 'role'",
                    param="messages",
                )
            # A role the chat template has never heard of was accepted with a
            # 200 and quietly folded into the prompt, so "sytem" looked like it
            # worked. The set is the OpenAI one; whether a given model's
            # template does anything sensible with each is the template's
            # business, not ours to guess at.
            if msg["role"] not in KNOWN_ROLES:
                return ClientError(
                    400,
                    f"messages[{i}] has role {msg['role']!r}; this server takes "
                    + ", ".join(sorted(KNOWN_ROLES)),
                    param="messages",
                )
    elif path in TEXT_PATHS:
        if not isinstance(body.get("prompt"), str):
            return ClientError(400, "'prompt' is required and must be a string",
                               param="prompt")

    rf = body.get("response_format")
    if rf is not None and rf != {"type": "text"}:
        return ClientError(
            400,
            "'response_format' is not supported here: nothing constrains the "
            "output to JSON or to a schema, so it would be ignored. Ask for "
            "JSON in the prompt and parse the reply, or leave "
            "'response_format' out",
            param="response_format",
        )

    # stream_options reaches the pinned server as a bare dict index
    # (`self.stream_options["include_usage"]`), so a dict without that key
    # raises *after* the response headers are out. At that point nothing can
    # be written back, and the client sees a stream that simply stops — the
    # exact failure the streaming client has no way to explain. Refuse it
    # before a byte is sent instead.
    if "stream_options" in body and body["stream_options"] is not None:
        so = body["stream_options"]
        if not isinstance(so, dict) or not isinstance(so.get("include_usage"), bool):
            return ClientError(
                400,
                "'stream_options' must be {\"include_usage\": true} or left "
                "out: without that key the server fails mid-stream, after the "
                "headers are sent, and the error cannot reach you",
                param="stream_options",
            )

    if "model" in body:
        model = body["model"]
        if not isinstance(model, str) or model == "":
            return ClientError(
                400,
                f"'model' must be \"{advertised}\", or left out",
                param="model",
            )
        if model not in aliases:
            return ClientError(
                404,
                f"model {model!r} is not served here. Use \"{advertised}\", "
                "or leave 'model' out",
                param="model",
                code="model_not_found",
            )

    # Both reach the same unload-then-load path as an unknown model. Nothing
    # nightsmith serves uses either.
    if body.get("draft_model", DEFAULT) not in (DEFAULT, None):
        return ClientError(400, "'draft_model' is not supported here",
                           param="draft_model")
    if body.get("adapters") is not None:
        return ClientError(400, "'adapters' is not supported here",
                           param="adapters")
    return None


def estimate_tokens(text):
    """Characters divided by four, the usual rough figure for English prose.

    An estimate, never a count: the tokenizer lives in the generation thread
    and the model may not even be loaded when a request arrives, so counting
    properly here would mean doing the work twice and guessing at the chat
    template. estimateTokens in window.go is the same arithmetic for the same
    reason. The ceilings it is compared against are set far enough below what
    was measured that the difference decides nothing.
    """
    return (len(text) + 3) // 4


def prompt_text(path, body):
    """Everything in the request the model will have to read."""
    parts = []
    if path in CHAT_PATHS:
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # The OpenAI "content parts" form. Only text is servable here.
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
    elif path in TEXT_PATHS:
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            parts.append(prompt)
    return "\n".join(parts)


def measure_prompt(path, body, count=None):
    """(tokens, counted). `count` is the model's own tokenizer when it has one.

    Characters ÷ 4 is close for English prose and wrong in both directions
    elsewhere: it over-counts repetitive text and under-counts code and any
    language that is not mostly ASCII. Under-counting is the dangerous
    direction — it is how a prompt past the cliff gets through a check meant
    to stop it — so the real tokenizer is used whenever the model is loaded,
    and the estimate is the fallback for the seconds before that.
    """
    text = prompt_text(path, body)
    if count is not None:
        try:
            return count(text), True
        except Exception:  # a tokenizer that cannot count must not 500 a request
            pass
    return estimate_tokens(text), False


def check_size(size, counted, path):
    """A prompt past the ceiling is a 400, before any of it reaches the model.

    Past roughly 50,000 tokens one request takes the whole server down with
    it, and the client gets no reply at all — 8.7 minutes of work and then a
    dropped socket. Refusing it in a hundredth of a second is the same trade
    nightsmith already makes for a model whose weights will not fit.
    """
    if size <= MAX_PROMPT_TOKENS:
        return None
    return ClientError(
        400,
        f"that prompt is {'' if counted else 'about '}{size:,} tokens, and this "
        f"server answers up to {MAX_PROMPT_TOKENS:,}. Measured on a 16 GB M4: "
        "43,389 tokens answered in 406 s, and ~58,000 killed the server "
        "outright, with no reply. Send less at a time",
        param="messages" if path in CHAT_PATHS else "prompt",
        code="prompt_too_large",
    )


def admit(size, running, running_tokens):
    """Whether a request of this size may start generating now.

    Three rules, in the order they were measured:

      - nothing else running: always yes. One prompt under MAX_PROMPT_TOKENS
        is the case the ladder measured directly, and refusing it would make
        the server unable to do the one thing it can definitely do.
      - at most MAX_CONCURRENT at once: four was fatal, twice.
      - within CONCURRENT_PROMPT_TOKENS combined: three at 8,000 was fine, so
        the budget is what was seen to work rather than what might.
    """
    if running <= 0:
        return True
    if running >= MAX_CONCURRENT:
        return False
    return running_tokens + size <= CONCURRENT_PROMPT_TOKENS


def health_state(generation_available, model_loaded, stopping=False):
    if stopping:
        return "stopping"  # finishing what it has, taking nothing new
    if not generation_available:
        return "failed"  # the generation thread died; the process is exiting
    if not model_loaded:
        return "loading"  # weights still loading: requests will wait
    return "ready"


def error_body(err):
    return {
        "error": {
            "message": err.message,
            "type": "invalid_request_error" if err.status < 500 else "server_error",
            "param": err.param,
            "code": err.code,
        }
    }


# ── The patching ────────────────────────────────────────────────────────────


def _patch(server, repo):
    import logging
    import signal
    import threading
    import time
    from io import BytesIO

    H = server.APIHandler
    orig_post = H.do_POST
    orig_validate = H.validate_model_parameters
    orig_send_response = H.send_response
    orig_handle_completion = H.handle_completion
    orig_run_http_server = server._run_http_server

    # running: generations under way. queued: requests waiting for a slot.
    # `in_flight` in /health keeps meaning the first of those, because
    # `nightsmith stop` counts on it to decide what it is waiting for.
    running = [0]
    running_tokens = [0]
    queued = [0]
    stopping = [False]
    cv = threading.Condition()
    created = int(time.time())

    def reply(self, status, obj, headers=()):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())
        self.wfile.flush()

    def fail(self, err):
        headers = []
        if err.retry_after is not None:
            headers.append(("Retry-After", str(err.retry_after)))
        reply(self, err.status, error_body(err), headers=headers)

    def send_response(self, *args, **kwargs):
        self._ns_headers_sent = True
        return orig_send_response(self, *args, **kwargs)

    def validate_model_parameters(self):
        # The request was checked in do_POST, before it could take a slot.
        # What is left is to stop the stock server unloading 6.8 GB of
        # weights to go looking for a model it is already serving.
        self.requested_model = DEFAULT
        self.requested_draft_model = DEFAULT
        return orig_validate(self)

    # The tokenizer belongs to the generation thread, which is using it. Rust
    # tokenizers are safe to call from several threads, but counting is
    # milliseconds against a generation's tens of seconds, so there is nothing
    # to win by overlapping them and one less thing to be wrong about.
    counting = threading.Lock()

    def token_counter(self):
        """The model's own tokenizer, or None before the weights are loaded."""
        tok = self.response_generator.model_provider.tokenizer
        if tok is None:
            return None

        def count(text):
            with counting:
                return len(tok.encode(text))

        return count

    def read_body(self):
        length = self.headers.get("Content-Length")
        if length is None:
            raise ClientError(411, "'Content-Length' header is required")
        try:
            n = int(length)
        except (TypeError, ValueError):
            raise ClientError(400, "'Content-Length' must be a number",
                              param="Content-Length")
        if n < 0:
            raise ClientError(400, "'Content-Length' must not be negative",
                              param="Content-Length")
        if n > MAX_BODY_BYTES:
            raise ClientError(
                413,
                f"the request body is {n:,} bytes, and this server reads up to "
                f"{MAX_BODY_BYTES:,}",
                param="Content-Length")
        return self.rfile.read(n)

    def acquire(size):
        """Wait for a slot. Returns None to proceed, or the error to send."""
        deadline = time.monotonic() + ADMIT_WAIT_SECONDS
        with cv:
            queued[0] += 1
            try:
                while True:
                    if stopping[0]:
                        # A stop that arrives while queueing is not congestion,
                        # and saying "too busy" would send the client back.
                        return ClientError(
                            503,
                            "the server is stopping; retry once it is started again",
                            retry_after=10)
                    if admit(size, running[0], running_tokens[0]):
                        running[0] += 1
                        running_tokens[0] += size
                        return None
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return ClientError(
                            503,
                            f"the server is already answering {MAX_CONCURRENT} "
                            f"requests and could not take another within "
                            f"{ADMIT_WAIT_SECONDS} s. More than {MAX_CONCURRENT} "
                            "at once is what kills it, so they wait rather "
                            "than run",
                            code="server_busy", retry_after=30)
                    cv.wait(timeout=min(left, 1.0))
            finally:
                queued[0] -= 1

    def release(size):
        with cv:
            running[0] -= 1
            running_tokens[0] -= size
            cv.notify_all()

    def do_POST(self):
        self._ns_headers_sent = False
        if self.path not in CHAT_PATHS + TEXT_PATHS:
            return orig_post(self)  # the stock 404 for an unknown path

        # Everything that can be refused is refused here: before a slot is
        # taken, before the model is touched, and while a reply can still be
        # written.
        try:
            raw = read_body(self)
            body = parse_body(raw)
            cli = self.response_generator.cli_args
            err = check_request(self.path, body, model_aliases(repo, cli.model), repo)
            if err is not None:
                raise err
            size, counted = measure_prompt(self.path, body, token_counter(self))
            err = check_size(size, counted, self.path)
            if err is not None:
                raise err
            if stopping[0]:
                raise ClientError(
                    503, "the server is stopping; retry once it is started again",
                    retry_after=10)
            if not self.response_generator.generation_available():
                raise ClientError(
                    503,
                    "the model server ran out of memory and is shutting down. "
                    "Run 'nightsmith start' to bring it back",
                    retry_after=30)
        except ClientError as e:
            fail(self, e)
            return

        turned_away = acquire(size)
        if turned_away is not None:
            fail(self, turned_away)
            return

        # The body has been read off the socket, so hand the stock parser a
        # copy of it. Restored afterwards: the real stream is where the next
        # request on this connection starts.
        real_rfile = self.rfile
        self.rfile = BytesIO(raw)
        try:
            orig_post(self)
        except ClientError as e:
            if not self._ns_headers_sent:
                fail(self, e)
        except (ValueError, TypeError, KeyError, AssertionError) as e:
            logging.warning(f"rejected request: {e!r}")
            if not self._ns_headers_sent:
                reply(self, 400, error_body(ClientError(400, str(e) or repr(e))))
        except Exception as e:
            logging.exception("request failed")
            if not self._ns_headers_sent:
                reply(self, 500, error_body(ClientError(500, f"server error: {e}")))
        finally:
            self.rfile = real_rfile
            release(size)

    def handle_completion(self, request, stop_words):
        # The stock server answers anything raised while starting a generation
        # with 404 {"error": "..."} — wrong code, wrong shape, and the one
        # thing it is actually used for is the generation thread being dead.
        try:
            return orig_handle_completion(self, request, stop_words)
        except Exception as e:
            if self._ns_headers_sent:
                raise
            if not self.response_generator.generation_available():
                raise ClientError(
                    503,
                    "the model server ran out of memory and is shutting down. "
                    "Run 'nightsmith start' to bring it back",
                    retry_after=30) from e
            raise

    def handle_health_check(self):
        rg = self.response_generator
        state = health_state(rg.generation_available(),
                             rg.model_provider.model is not None, stopping[0])
        body = {"status": state, "model": repo,
                "in_flight": running[0], "queued": queued[0]}
        if state == "ready":
            reply(self, 200, body)
        elif state == "loading":
            reply(self, 503, body, headers=[("Retry-After", "5")])
        elif state == "stopping":
            reply(self, 503, body, headers=[("Retry-After", "10")])
        else:
            reply(self, 503, body)

    def handle_models_request(self):
        model = {"id": repo, "object": "model", "created": created,
                 "owned_by": "nightsmith"}
        parts = self.path.split("?")[0].rstrip("/").split("/")
        if len(parts) > 3:  # /v1/models/<id>, where the id may contain "/"
            wanted = "/".join(parts[3:])
            if wanted == repo:
                reply(self, 200, model)
            else:
                reply(self, 404, error_body(ClientError(
                    404, f"model {wanted!r} is not served here", param="model",
                    code="model_not_found")))
            return
        reply(self, 200, {"object": "list", "data": [model]})

    H.send_response = send_response
    H.validate_model_parameters = validate_model_parameters
    H.do_POST = do_POST
    H.handle_completion = handle_completion
    H.handle_health_check = handle_health_check
    H.handle_models_request = handle_models_request

    # The main thread is inside serve_forever, which is where this handler
    # runs, so the server cannot be shut down from here. Nothing needs it to
    # be: new completions are refused above, and once the ones in progress
    # are answered the process exits.
    def drain_then_exit():
        deadline = time.monotonic() + DRAIN_SECONDS
        while running[0] > 0 and time.monotonic() < deadline:
            time.sleep(0.1)
        if running[0] > 0:
            logging.warning(f"stopping with {running[0]} request(s) unanswered "
                            f"after {DRAIN_SECONDS} s")
        os._exit(0)

    def on_sigterm(signum, frame):
        if stopping[0]:
            return
        with cv:
            stopping[0] = True
            cv.notify_all()  # anything still queued is turned away, not waited on
        logging.info(f"stopping: finishing {running[0]} request(s) in progress")
        threading.Thread(target=drain_then_exit, daemon=True).start()

    signal.signal(signal.SIGTERM, on_sigterm)

    # Once the generation thread dies the process can never answer again: the
    # Metal allocator has failed and mlx-lm has no way to start another. It
    # used to stay up and answer 404 to everything for ever, which is a state
    # nothing detects and nobody asked for. Say why, in the log `nightsmith
    # status` reads, and go.
    def watch_generation(rg):
        while True:
            time.sleep(1.0)
            if stopping[0]:
                return
            if not rg.generation_available():
                # Short, and the last thing in the log: `nightsmith status`
                # prints whatever follows the marker, on one line, to someone
                # who is only trying to find out why nothing is running.
                logging.error("the model ran out of memory answering a request; "
                              "exiting so 'nightsmith start' can bring it back")
                logging.error(f"{FATAL_MARKER} it ran out of memory answering a "
                              "request")
                logging.shutdown()
                sys.stderr.flush()
                os._exit(70)

    def _run_http_server(host, port, response_generator, **kwargs):
        threading.Thread(target=watch_generation, args=(response_generator,),
                         daemon=True).start()
        return orig_run_http_server(host, port, response_generator, **kwargs)

    server._run_http_server = _run_http_server


def _check_pin(mlx_lm, server):
    if mlx_lm.__version__ != PINNED_MLX_LM:
        return f"mlx-lm {mlx_lm.__version__} is installed, but this server was written for {PINNED_MLX_LM}"
    needed = {
        server.APIHandler: ("do_POST", "validate_model_parameters",
                            "handle_completion", "handle_health_check",
                            "handle_models_request", "_set_cors_headers"),
        server.ResponseGenerator: ("generation_available",),
        server.ModelProvider: ("load",),
    }
    for cls, names in needed.items():
        for name in names:
            if not hasattr(cls, name):
                return f"mlx-lm has no {cls.__name__}.{name}"
    if not hasattr(server, "_run_http_server"):
        return "mlx-lm has no server._run_http_server"
    return None


def split_args(argv):
    """nightsmith's own flags, and the rest for mlx_lm.server.

    Returns (repo, drain_seconds, rest) or raises ValueError.
    """
    repo = None
    drain = None
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--served-model-id", "--drain-seconds"):
            if i + 1 >= len(argv):
                raise ValueError(f"{a} needs a value")
            if a == "--served-model-id":
                repo = argv[i + 1]
            else:
                try:
                    drain = int(argv[i + 1])
                except ValueError:
                    raise ValueError("--drain-seconds must be a whole number")
                if drain < 0:
                    raise ValueError("--drain-seconds must not be negative")
            i += 2
            continue
        rest.append(a)
        i += 1
    if not repo:
        raise ValueError("--served-model-id is required")
    return repo, drain, rest


def main(argv):
    global DRAIN_SECONDS
    try:
        repo, drain, rest = split_args(argv)
    except ValueError as e:
        print(f"usage: serve.py --served-model-id <repo> [--drain-seconds N] "
              f"<mlx_lm server flags...>\nserve.py: {e}", file=sys.stderr)
        return 2
    if drain is not None:
        DRAIN_SECONDS = drain

    import mlx_lm
    from mlx_lm import server

    problem = _check_pin(mlx_lm, server)
    if problem:
        # Never fall through to the unpatched server: it would look healthy.
        print(f"nightsmith: {problem}. Run 'nightsmith' to reinstall the runtime.",
              file=sys.stderr)
        return 1

    _patch(server, repo)
    sys.argv = ["mlx_lm.server"] + rest
    server.main()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
