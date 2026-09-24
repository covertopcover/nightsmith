# nightsmith

Local AI on your own Mac, set up with one terminal command.

```sh
curl -fsSL https://nightsmith.sh/install | sh
```

Nightsmith looks at your Mac, picks a model that fits, installs the runtime,
downloads the model, and **proves it works by asking it a real question**.
Then you talk to it — type `nightsmith` — or point any OpenAI-compatible
client at `http://127.0.0.1:8080/v1`.

It installs to `~/.local/bin`, adds that to your PATH if it isn't there, and
hands you a fresh shell at the end when it had to — there is nothing to source
and no terminal to restart. Setup takes about five minutes and 8 GB of disk on
a 16 GB M4.

- **No password, ever.** Everything lives in `~/.nightsmith` and
  `~/.local/bin`. No `sudo`, no Homebrew, no Xcode.
- **It uses its own copy of Python**, not yours. A model already in
  `~/.cache/huggingface` (shared with Ollama and LM Studio) is reused, never
  modified, never deleted.
- **A checkmark means it happened.** Setup is done when the model has
  answered, not when a health check returned 200.
- **Nothing you type leaves the machine.** No account, no telemetry, and no
  network call once the model is downloaded.

Apple Silicon only (M1 or newer). Intel Macs get a clear refusal rather than a
slow fallback.

## Using it

```
  › what's a spring tide?

  A spring tide is a tide cycle that occurs when the moon and sun are
  aligned, resulting in the highest high tides and lowest low tides.

  5s · about 10 words a second
```

Answers stream as they are written, which matters at nine words a second.
**Ctrl-C** stops an answer without leaving, **Ctrl-D** leaves, and
`/help /new /sessions /resume ID /context /paste /exit` work inside.
Conversations are written to `~/.nightsmith/chats` as they happen.

| Command | What it does |
|---|---|
| `nightsmith` | Talk to it. Before setup, set it up. With no terminal it reports status instead, with `status`'s exit codes |
| `nightsmith -p "…"` | One question, one answer. Reads stdin when given no question, or a lone `-` |
| `nightsmith -c` | Pick up the last conversation |
| `nightsmith -r [ID]` | Pick up that conversation — or list them all |
| `nightsmith start` / `nightsmith stop` | Turn the model server on and off |
| `nightsmith status` | Ask it a real question, and show its memory use. `--json` for programs |
| `nightsmith remove` | List everything it installed, then delete it (`uninstall` works too) |
| `nightsmith config check` | What your settings will cost, against what this Mac can give them |
| `nightsmith model list` | What fits on this Mac, with real sizes |
| `nightsmith model use <repo>` | Switch model, and prove the new one answers |
| `nightsmith --yes` | Set up without asking, for a script |

```sh
nightsmith -p "what's a spring tide?"
cat notes.txt | nightsmith -p "sum this up" -   # instruction, then material
```

Standard output is the answer and nothing else; timings and warnings go to
standard error, and only when it is a terminal.

## For a program or an agent

The server speaks the OpenAI chat-completions API, so anything that can point
at a custom base URL can use it. **Base URL** `http://127.0.0.1:8080/v1` —
`nightsmith status` prints the port in use if 8080 was taken. **Model id** is
the Hub repo as `/v1/models` lists it, or leave `model` out.

```sh
curl -fsSL https://nightsmith.sh/install | NIGHTSMITH_YES=1 sh   # unattended
nightsmith status --json    # 0 answered · 3 not running · 4 not answering
nightsmith start            # returns only once the model has answered
```

Five things to know before you drive it:

- **Three requests at a time.** Concurrency is batched and genuinely faster,
  but past three it killed the server, so a fourth waits for a slot (up to two
  minutes) and then gets a 503 with `Retry-After`. `GET /health` reports
  `in_flight` and `queued`.
- **A prompt over 40,000 tokens is a 400**, counted with the model's own
  tokenizer. Past the ceiling one request takes the whole server down after
  eight minutes of trying. A refusal takes 0.07 s.
- **Treat a stream that ends without `[DONE]` or a `finish_reason` as a
  failure.** It is a well-formed 200 either way — once a stream has started,
  nothing can write an error into it. Nightsmith reads it that way too.
- **`response_format` is refused with a 400.** Nothing here constrains output
  to JSON or a schema, so accepting it would be a promise not kept. Ask for
  JSON in the prompt and parse the reply; 27 of 30 varied requests parsed
  first try.
- **`/health` is a hint.** The only proof that a request will work is a
  request that worked — that is how nightsmith checks itself, and why every
  checkmark it prints costs a real completion.

Errors are the OpenAI shape on every path (`error.message`, `type`, `param`,
`code`). `nightsmith stop` lets answers in progress finish — a whole answer's
worth of writing, about 2.5 minutes at the defaults. In Activity Monitor the
server is `nightsmith-model`.

## Measured on a base M4 / 16 GB

Gemma 4 12B (4-bit) on `mlx_lm.server`. Setup measures your Mac and reports
its own numbers rather than quoting these.

| | |
|---|---|
| Writing | ~12 tokens/s (about 9 words a second), flat over a 92-minute soak |
| Reading a prompt | ~131 tokens/s cold, ~1,900 warm · **~19 s per 2,048 tokens**, so 40,000 tokens is six minutes before the first word |
| Memory | 10.2 GB peak, 7.7 GB idle, against Metal's 12.7 GB GPU limit |
| Endurance | 953 requests over 3 hours, zero failures, memory flat |
| Context ceiling | 38,009 tokens answered in 349 s · 43,389 in 406 s · 62,009 fatal. Capped at 40,000 |
| Concurrency ceiling | 3 × 8,000 tokens fine · 4 × 4,000 fatal · 5 × 2,000 fatal. Capped at 3 |
| A conversation | Gets *faster* as it grows: 1.40 s to first word on turn 1, 0.71 s by turn 4, as the cache reuses the previous turn |

Speed is set by memory bandwidth, not RAM: the same model is about 2× faster
on an M-series Pro and 4× on a Max.

The cache has one slot, which is why `nightsmith status` mid-conversation
costs you something: it sends a real question of its own, dropping reuse from
157 tokens to 4 and the next first word from 0.71 s to 2.15 s.

## Known limits

- **No tools, no skills, no system prompt.** It cannot read your files, run
  commands or look anything up, and it says so when asked. The model also
  introduces itself as Gemma and does not know it is running here.
- **Measured in English only.** A model this size can write fluent-looking
  nonsense in a smaller language — inventing words, reversing meanings — with
  nothing in the output to show it has.
- **It invents academic citations**, while correctly refusing to invent APIs.
- **At `temperature = 0` a prompt that fails fails identically for ever**, so
  retrying cannot help. Raise the temperature for that one prompt.
- **Scheduling, job queues and supervision are out of scope for now.** If the
  model runs out of memory the server says so in its log and exits; `status`
  explains it and `nightsmith start` brings it back. Nothing restarts it for
  you.
- Only the 16 GB rung has been measured. Every other size in `models.toml` is
  arithmetic, and is labelled as such.

## Building

```sh
go test ./...
GOOS=darwin GOARCH=arm64 go build -o nightsmith .
```

## License

MIT
