# nightsmith

Local AI that runs in the background on your own Mac, set up with one
terminal command.

```sh
curl -fsSL https://nightsmith.sh/install | sh
```

Nightsmith looks at your Mac, picks a model that fits, installs the runtime,
downloads the model, and **proves it works by asking it a real question**.
Then it leaves you with one-word commands to start it, stop it, and remove it.

- **No password, ever.** Everything lives in `~/.nightsmith` and
  `~/.local/bin`. No `sudo`, no Homebrew, no Xcode.
- **Nothing it doesn't own is touched.** Nightsmith uses its own copy of Python,
  not yours. A model already in `~/.cache/huggingface` (shared with Ollama and
  LM Studio) is reused, never modified, and never deleted.
- **A checkmark means it happened.** Setup is done when the model has answered,
  not when a health check returns 200.

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

  ✓  Runtime ready                    11s
  ✓  Downloaded gemma-4-12B-it-4bit   6.8 GB   2m52s
  ✓  Started the model                1s
  ✓  Asked it something — it answered:
        "Hello!"
  ✓  Thinking stayed off
  ✓  Measured it here:  about 9 words a second · 7.9 GB peak · no swap

  Ready. It's running now.

    nightsmith start     turn it on
    nightsmith stop      turn it off
    nightsmith status    is it running, and how much memory
    nightsmith remove    take it back off this Mac

  While it's on, it's at  http://127.0.0.1:8080
```

The server speaks the OpenAI chat-completions API, so anything that can point
at a custom base URL can use it.

## Commands

| | |
|---|---|
| `nightsmith` | Set up, or report that setup is already done |
| `nightsmith start` / `stop` | Turn the model server on and off |
| `nightsmith status` | Ask it a real question, and show its memory use |
| `nightsmith remove` | List everything it installed, then delete it (`uninstall` works too) |
| `nightsmith config check` | Estimate peak memory for your settings against this Mac's limit |
| `nightsmith model list` | What fits on this Mac, with real sizes |
| `nightsmith model use <repo>` | Switch model, and prove the new one answers |

## Measured, not guessed

Measured on a base M4 / 16 GB running Gemma 4 12B (4-bit) on `mlx_lm.server`,
over 30 background-style requests (20 prompts of ~2,200 tokens, 10 of ~300):

| | |
|---|---|
| Writing speed | ~12 tokens/s (about 9 words a second), flat over a 92-minute soak |
| Reading speed | ~131 tokens/s cold, ~1,900 warm (prompt cache hit) |
| Peak memory | 10.2 GB, against Metal's 12.7 GB GPU limit |
| Idle memory | 7.7 GB |

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

## Status

Early. It works end to end on the machine above. Only the 16 GB rung has been
measured. Every other size in `models.toml` is arithmetic, and is labelled as
such. Scheduling, job queues, and agent integrations are deliberately out of
scope for now.

## Building

```sh
go test ./...
GOOS=darwin GOARCH=arm64 go build -o nightsmith .
```

## License

MIT
