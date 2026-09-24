# nightsmith

Local AI that runs in the background on your own Mac, set up with one
terminal command.

```sh
curl -fsSL https://nightsmith.sh/install | sh
```

It installs to `~/.local/bin`, adds that to your PATH if it isn't there, and
goes straight into setup. When it has to change your PATH it hands you a fresh
shell at the end, so `nightsmith` works immediately — there is nothing to
source and no terminal to restart.

Nightsmith looks at your Mac, picks a model that fits, installs the runtime,
downloads the model, and **proves it works by asking it a real question**.
Then you can talk to it — type `nightsmith` and it answers, on your own
machine, with nothing leaving it.

- **No password, ever.** Everything lives in `~/.nightsmith` and
  `~/.local/bin`. No `sudo`, no Homebrew, no Xcode.
- **Nothing it doesn't own is touched.** Nightsmith uses its own copy of Python,
  not yours. A model already in `~/.cache/huggingface` (shared with Ollama and
  LM Studio) is reused, never modified, and never deleted.
- **A checkmark means it happened.** Setup is done when the model has answered,
  not when a health check returns 200.
- **Nothing you type leaves the machine.** There is no account, no telemetry
  and no network call once the model is downloaded.

Apple Silicon only (M1 or newer). Intel Macs get a clear refusal instead of a
slow fallback.

## What it looks like

```
  Looking at this Mac…

  ✓  Apple M4 · 10 GPU cores · ~120 GB/s memory
  ✓  macOS 27.0
  ✓  16 GB memory — the GPU can use 12.7 GB of it
  ✓  33 GB free on disk — setup needs 7.6 GB

  Here's what fits, and what it'll be like:

     gemma-4-12B-it-4bit  ·  6.8 GB download

     Uses 10.2 GB while working, leaving 2.5 GB spare.
     Writes about 9 words a second — steady, not fast.

  Set it up? [Y/n]

  ✓  Fetched uv                       1s
  ✓  Installed Python 3.12.14         1s
  ✓  Installed MLX (mlx-lm 0.32.0)    7s
  ✓  Downloaded gemma-4-12B-it-4bit   6.8 GB   2m52s
  ✓  Started the model                1s
  ✓  Asked it something — it answered:
        "Hello!"
  ✓  Thinking stayed off
  ✓  Measured it here:  about 9 words a second · 7.9 GB peak · no swap

  Ready. It's running now.

    nightsmith           talk to it
    nightsmith start     turn it on
    nightsmith stop      turn it off
    nightsmith status    is it running, and how much memory
    nightsmith remove    take it back off this Mac

  While it's on, it's at  http://127.0.0.1:8080/v1
  OpenAI-compatible · model "mlx-community/gemma-4-12B-it-4bit", or leave it out

  Settings: ~/.nightsmith/config.toml — each one explained.
  'nightsmith config check' shows what they cost.
```

## Talking to it

Type `nightsmith` and you are in a conversation. Answers stream as they are
written, which matters at about nine words a second.

```
  nightsmith — gemma-4-12B-it-4bit, on this Mac.
  Nothing you type here leaves it.
  /help for commands · Ctrl-D to leave · no arrow-key history yet

  › what's a spring tide?

  A spring tide is a tide cycle that occurs when the moon and sun are
  aligned, resulting in the highest high tides and lowest low tides.

  5s · about 10 words a second

  ›
```

- **Ctrl-C** stops an answer without leaving. **Ctrl-D** leaves.
- **`/help /new /sessions /resume ID /context /paste /exit`** inside.
- Every conversation is **written down as it happens**, to
  `~/.nightsmith/chats`. `nightsmith -c` picks up the last one,
  `nightsmith -r ID` a particular one, and `nightsmith -r` lists them.
  `nightsmith remove` deletes them along with everything else.
- **A long conversation gets *faster* off the mark**, not slower: each turn
  reuses the previous turn's prompt from the server's cache. Measured below.
- `/context` shows how much is being re-read each turn. Past about 8,000
  tokens the oldest exchanges stop being sent — it says so when that happens,
  and `chat_context_tokens` in your settings is the dial.
- **The transcript on disk is never trimmed**, only what gets sent. A
  conversation shortened today is still complete when you read it tomorrow.

### One question, one answer

For scripts, and for another program driving it:

```sh
nightsmith -p "what's a spring tide?"
cat notes.txt | nightsmith -p                 # the question is stdin
cat notes.txt | nightsmith -p "sum this up" -  # instruction, then material
```

Standard output is the answer and nothing else — timings and warnings go to
standard error, and only when it is a terminal. Exit codes are `status`'s:
**0** answered, **3** not running, **4** running but could not answer. An answer
cut short mid-stream is a **4**, never a 0.

`-p` never starts the server; `nightsmith start` is one word and says what it
is doing. It is stateless, too — it does not add to any conversation.

## Using it from a program

The server speaks the OpenAI chat-completions API, so anything that can point
at a custom base URL can use it.

- **Base URL** `http://127.0.0.1:8080/v1`. If 8080 was taken, setup picked the
  next free port; `nightsmith status` shows the one in use.
- **Model id** is the Hub repo, as `/v1/models` lists it. Leaving `model` out
  works too. Any other id is refused with a 404 — it never unloads the model
  that is serving.
- **`GET /health`** says `ready` (200), `loading` (503, with `Retry-After`),
  `stopping` (503, with `Retry-After`) or `failed` (503: run `nightsmith
  stop`, then `nightsmith start`). `nightsmith
  start` returns only once the model has answered, so a client started after
  it will not normally see `loading`.
- **Streaming works** (`"stream": true`). Ask for
  `"stream_options": {"include_usage": true}` and the last frame before
  `[DONE]` carries exact token counts plus `prompt_tokens_details.cached_tokens`
  — how much of your prompt the cache reused. A `stream_options` without
  `include_usage` is refused with a 400, because the server it wraps would
  otherwise fail *after* the headers were sent, reaching you as a stream that
  simply stops.
- **Treat a stream that ends without `[DONE]` or a `finish_reason` as a
  failure.** It is a well-formed 200 either way: once a stream has started,
  nothing can write an error into it. That is how nightsmith reads it too.
- **Bad requests get a 400** that names the field, in the OpenAI error shape.
- **`response_format` is refused with a 400.** Nothing here constrains output
  to JSON or a schema, so accepting it would be a promise not kept. Ask for
  JSON in the prompt and parse the reply.
- **`nightsmith stop` lets requests in progress finish** (up to 30 s). New
  requests during that time get a 503 with `Retry-After`.
- **Defaults come from your settings.** A request that leaves out
  `max_tokens` or `temperature` gets the values in `~/.nightsmith/config.toml`
  (1500 and 0 unless you changed them).
- In Activity Monitor the server is **`nightsmith-model`**.
- `/health` is a hint. The only proof that a request will work is a request
  that worked — that is how nightsmith checks itself.

## Commands

| Command | What it does |
|---|---|
| `nightsmith` | Talk to it. Before setup, set it up. With no terminal (a script, cron, CI) it reports status instead, with the same exit codes as `status` |
| `nightsmith -p "…"` | One question, one answer, for scripts. Reads stdin when given no question, or a lone `-` |
| `nightsmith -c` | Pick up the last conversation |
| `nightsmith -r [ID]` | Pick up that conversation — or list them all |
| `nightsmith start` / `stop` | Turn the model server on and off |
| `nightsmith status` | Ask it a real question, and show its memory use |
| `nightsmith status --json` | The same, for programs: `state`, `model`, `port`, `url`, `pid`, `memory_bytes`. Exit code either way: 0 answered, 3 not running, 4 running but not answering |
| `nightsmith remove` | List everything it installed, then delete it (`uninstall` works too) |
| `nightsmith config check` | Estimate peak memory for the settings in `~/.nightsmith/config.toml` against this Mac's limit; flags misspelled settings, and ones the running server hasn't picked up yet |
| `nightsmith model list` | What fits on this Mac, with real sizes |
| `nightsmith model use <repo>` | Switch model, and prove the new one answers |

## Measured, not guessed

Measured on a base M4 / 16 GB running Gemma 4 12B (4-bit) on `mlx_lm.server`,
over 30 background-style requests (20 prompts of ~2,200 tokens, 10 of ~300):

| Measurement | Result |
|---|---|
| Writing speed | ~12 tokens/s (about 9 words a second), flat over a 92-minute soak |
| Reading speed | ~131 tokens/s cold, ~1,900 warm (prompt cache hit) |
| Peak memory | 10.2 GB, against Metal's 12.7 GB GPU limit |
| Idle memory | 7.7 GB |

In a conversation, measured across four turns on the same Mac:

| Measurement | Result |
|---|---|
| Prompt, turn 1 → 4 | 24 → 225 tokens |
| Reused from cache | 4 → 157 tokens |
| Time to first word | 1.40s → **0.71s** — it speeds up as it grows |
| Writing speed | 12.4–13.3 tokens/s throughout |

That last row is why `/context` warns you about `nightsmith status`: it sends
a real question of its own, and the cache has one slot. Running it mid-
conversation dropped the reuse from 157 tokens back to 4 and pushed the next
answer's first word from 0.71s to 2.15s.

Speed is set by memory bandwidth, not RAM. The same model is about 2× faster on
an M-series Pro and about 4× faster on a Max. Setup measures your Mac and
reports its own numbers rather than quoting these.

Nightsmith handles the things that make local models fail silently:

- **Thinking is sent explicitly.** Left unset, `mlx_lm` turns reasoning on for
  any model that supports it. A short prompt can then burn its whole token
  budget thinking and return nothing.
- **The runtime is pinned to a git commit.** The PyPI release of `mlx-lm`
  cannot load Gemma 4. The server starts, answers `/v1/models`, and then hangs
  on the first real request.
- **Server flags known to break a model are never passed.** `--kv-bits` and
  `--draft-model` are accepted without complaint and fail only when work
  arrives.
- **Models are pinned to a Hub commit and checked by size**, so a partial
  download is never mistaken for a model.
- **The server's edges are made honest.** Stock `mlx_lm.server` drops the
  connection on a request with no `messages`, says `ok` while the weights are
  still loading, unloads the model it is serving when asked for one it
  doesn't have, and crashes mid-stream on a `stream_options` that omits
  `include_usage`. Nightsmith runs it through a small wrapper (`serve.py`)
  that fixes those and leaves generation untouched.
- **A stream that stops early is treated as a failure**, not a short answer.
  It arrives as a well-formed 200, so anything that trusts the connection
  closing will report half an answer as a whole one.

## Status

Early. It works end to end on the machine above. Only the 16 GB rung has been
measured. Every other size in `models.toml` is arithmetic, and is labelled as
such.

You can talk to it, and it remembers a conversation. It has **no tools, no
skills and no ability to do anything but answer** — it cannot read your
files, run commands or look anything up, and it says so when asked. Whether
it should is undecided, and depends on how well a 12B 4-bit model calls tools
at all, which has not been measured. Scheduling, job queues, supervision and
agent integrations are deliberately out of scope for now.

## Building

```sh
go test ./...
GOOS=darwin GOARCH=arm64 go build -o nightsmith .
```

## License

MIT
