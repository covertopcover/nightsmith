"""The model server nightsmith runs: the pinned mlx_lm server, with its HTTP
edges made honest for clients.

Nightsmith embeds this file and runs it as

    python3 serve.py --served-model-id <repo> <mlx_lm server flags...>

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

/health is a hint for clients. Nightsmith itself never trusts it: every
checkmark it prints still comes from a real completion.

The pure checks at the top import nothing from mlx, so they can be tested with
any Python. The patching below them depends on the exact pinned version, and
refuses to run against any other.
"""

import os
import sys

PINNED_MLX_LM = "0.32.0"

CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")
TEXT_PATHS = ("/v1/completions",)
DEFAULT = "default_model"
# How long a stop waits for requests in progress. `nightsmith stop` waits
# longer than this before it resorts to SIGKILL.
DRAIN_SECONDS = 30


class ClientError(Exception):
    def __init__(self, status, message, param=None, code=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code


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


def health_state(generation_available, model_loaded, stopping=False):
    if stopping:
        return "stopping"  # finishing what it has, taking nothing new
    if not generation_available:
        return "failed"  # the generation thread died; a person must restart it
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
    import json
    import logging
    import signal
    import threading
    import time

    H = server.APIHandler
    orig_post = H.do_POST
    orig_validate = H.validate_model_parameters
    orig_send_response = H.send_response
    in_flight = [0]
    stopping = [False]
    lock = threading.Lock()
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

    def send_response(self, *args, **kwargs):
        self._ns_headers_sent = True
        return orig_send_response(self, *args, **kwargs)

    def validate_model_parameters(self):
        cli = self.response_generator.cli_args
        err = check_request(self.path, self.body,
                            model_aliases(repo, cli.model), repo)
        if err is not None:
            raise err
        self.requested_model = DEFAULT
        self.requested_draft_model = DEFAULT
        return orig_validate(self)

    def do_POST(self):
        self._ns_headers_sent = False
        counted = self.path in CHAT_PATHS + TEXT_PATHS
        if counted and stopping[0]:
            reply(self, 503, error_body(ClientError(
                503, "the server is stopping; retry once it is started again")),
                headers=[("Retry-After", "10")])
            return
        if counted:
            with lock:
                in_flight[0] += 1
        try:
            orig_post(self)
        except ClientError as e:
            if not self._ns_headers_sent:
                reply(self, e.status, error_body(e))
        except (ValueError, TypeError, KeyError, AssertionError) as e:
            logging.warning(f"rejected request: {e!r}")
            if not self._ns_headers_sent:
                reply(self, 400, error_body(ClientError(400, str(e) or repr(e))))
        except Exception as e:
            logging.exception("request failed")
            if not self._ns_headers_sent:
                reply(self, 500, error_body(ClientError(500, f"server error: {e}")))
        finally:
            if counted:
                with lock:
                    in_flight[0] -= 1

    def handle_health_check(self):
        rg = self.response_generator
        state = health_state(rg.generation_available(),
                             rg.model_provider.model is not None, stopping[0])
        body = {"status": state, "model": repo, "in_flight": in_flight[0]}
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
    H.handle_health_check = handle_health_check
    H.handle_models_request = handle_models_request

    # The main thread is inside serve_forever, which is where this handler
    # runs, so the server cannot be shut down from here. Nothing needs it to
    # be: new completions are refused above, and once the ones in progress
    # are answered the process exits.
    def drain_then_exit():
        deadline = time.monotonic() + DRAIN_SECONDS
        while in_flight[0] > 0 and time.monotonic() < deadline:
            time.sleep(0.1)
        if in_flight[0] > 0:
            logging.warning(f"stopping with {in_flight[0]} request(s) unanswered "
                            f"after {DRAIN_SECONDS} s")
        os._exit(0)

    def on_sigterm(signum, frame):
        if stopping[0]:
            return
        stopping[0] = True
        logging.info(f"stopping: finishing {in_flight[0]} request(s) in progress")
        threading.Thread(target=drain_then_exit, daemon=True).start()

    signal.signal(signal.SIGTERM, on_sigterm)


def _check_pin(mlx_lm, server):
    if mlx_lm.__version__ != PINNED_MLX_LM:
        return f"mlx-lm {mlx_lm.__version__} is installed, but this server was written for {PINNED_MLX_LM}"
    needed = {
        server.APIHandler: ("do_POST", "validate_model_parameters",
                            "handle_health_check", "handle_models_request",
                            "_set_cors_headers"),
        server.ResponseGenerator: ("generation_available",),
        server.ModelProvider: ("load",),
    }
    for cls, names in needed.items():
        for name in names:
            if not hasattr(cls, name):
                return f"mlx-lm has no {cls.__name__}.{name}"
    return None


def main(argv):
    if len(argv) < 2 or argv[0] != "--served-model-id":
        print("usage: serve.py --served-model-id <repo> <mlx_lm server flags...>",
              file=sys.stderr)
        return 2
    repo, rest = argv[1], argv[2:]

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
