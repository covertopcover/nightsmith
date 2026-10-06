# nightsmith

Local AI that runs in the background on your own Mac, set up with one terminal
command.

```sh
curl -fsSL https://nightsmith.sh/install | sh
```

## Use cases

The model runs on your Mac, so the text that you give it stays on your Mac.
That makes nightsmith a fit for text that you cannot paste into a cloud
service: contracts, customer data, medical letters, private code. There is
also no cost for each request and no rate limit.

- Extract data. Pull names, dates and amounts out of invoices, emails or
  contracts, and ask for the result as JSON.
- Summarize text. Shorten meeting notes, long email threads and reports.
- Sort and label. Tag support tickets, emails or survey answers by topic,
  urgency or sentiment.
- Redact. Remove names, email addresses and phone numbers from a text before
  you share it or send it to a cloud model.
- Ask questions about a document. Give it a contract or a policy, and ask
  what the document says about one point.
- Explain a hard letter. Paste a medical, legal or tax letter and ask for
  plain words.
- Rewrite and proofread. Correct grammar, change the tone, or make a text
  shorter.
- Draft replies. Write a first version of an answer to an email or a message.
- Clean up messy text. Turn rough notes into a table, or put dates and
  addresses into one format.
- Read logs and errors. Ask what an error means, or summarize a log that
  contains internal hostnames and customer IDs.
- Work with private code. Ask it to explain a script, or to write a commit
  message from a diff.
- Run a batch from a script. Loop over a folder of files with
  `nightsmith -p` and collect the answers.

One prompt can hold up to 40,000 tokens, which is roughly 30,000 words. The
model does not look anything up, so do not use it as a source of facts. See
Known limits.

## What setup does

Nightsmith looks at your Mac and picks a model that fits. It installs the
runtime (the program that runs the model) and downloads the model. Then it
asks the model a real question, to prove that the model works.

To talk to the model, type `nightsmith`. You can also point any
OpenAI-compatible client at `http://127.0.0.1:8080/v1`. The model server keeps
running after you close the terminal.

Nightsmith installs to `~/.local/bin`. If that directory is not in your PATH,
nightsmith adds it and opens a fresh shell at the end. You do not source a
file or restart the terminal. On a 16 GB M4, setup takes about five minutes
and 8 GB of disk.

- Nightsmith never asks for a password. Everything is in `~/.nightsmith` and
  `~/.local/bin`. It does not use `sudo`, Homebrew or Xcode.
- Nightsmith uses its own copy of Python, not yours. If a model is already in
  `~/.cache/huggingface`, nightsmith reuses it. Ollama and LM Studio share
  that directory, and nightsmith does not change or delete the models in it.
- A checkmark means that the step happened. Setup is done when the model
  answers a question. A health check that returns 200 is not enough.
- Nothing that you type leaves the machine. There is no account and no
  telemetry. After the model downloads, nightsmith makes no network calls.

Nightsmith runs on Apple Silicon only (M1 or newer). On an Intel Mac,
nightsmith refuses clearly. It does not fall back to a slow mode.

## Using it

Type `nightsmith` to start a conversation. Nightsmith shows each answer as the model writes it. At nine words a second,
you do not want to wait for the whole answer. Press Ctrl-C to stop an answer
and stay in the conversation. Press Ctrl-D to leave. These commands work
inside a conversation:
`/help /new /sessions /resume ID /context /paste /exit`.
Nightsmith writes each conversation to `~/.nightsmith/chats` as it happens.

| Command | What it does |
|---|---|
| `nightsmith` | Talk to the model. Before setup, this command runs setup. With no terminal, it reports status and uses the exit codes of `status` |
| `nightsmith -p "…"` | One question, one answer. With no question, or with a lone `-`, it reads stdin |
| `nightsmith -c` | Continue the last conversation |
| `nightsmith -r [ID]` | Continue that conversation. With no ID, list all conversations |
| `nightsmith start` / `nightsmith stop` | Turn the model server on and off |
| `nightsmith status` | Ask the model a real question and show its memory use. Add `--json` for programs |
| `nightsmith remove` | List everything that nightsmith installed, then delete it (`uninstall` also works) |
| `nightsmith config check` | Show what your configuration costs, against what this Mac can give |
| `nightsmith model list` | Show the models that fit on this Mac, with real sizes |
| `nightsmith model use <repo>` | Switch model, and prove that the new model answers |
| `nightsmith --yes` | Set up without questions, for a script |

```sh
nightsmith -p "what's a spring tide?"
cat notes.txt | nightsmith -p "sum this up" -   # instruction, then material
cat invoice.txt | nightsmith -p "list the supplier, date and total as JSON" -
git diff | nightsmith -p "write a commit message for this diff" -
```

Standard output contains the answer and nothing else. Timings and warnings go
to standard error, and only when standard error is a terminal.

## For a program or an agent

The server speaks the OpenAI chat-completions API. Any program that accepts a
custom base URL can use it. The base URL is `http://127.0.0.1:8080/v1`. If
port 8080 was taken, `nightsmith status` prints the port in use. The model id
is the Hugging Face repo as `/v1/models` lists it. You can also leave `model`
out.

```sh
curl -fsSL https://nightsmith.sh/install | NIGHTSMITH_YES=1 sh   # unattended
nightsmith status --json    # 0 answered · 3 not running · 4 not answering
nightsmith start            # returns only once the model has answered
```

Know these five things before you send requests:

- The server runs three requests at a time. It batches concurrent requests,
  and that is faster. More than three at once killed the server. A fourth
  request waits up to two minutes for a slot, then gets a 503 with
  `Retry-After`. `GET /health` reports `in_flight` and `queued`.
- A prompt over 40,000 tokens gets a 400. A token is a piece of a word.
  Nightsmith counts tokens with the model's own tokenizer. Above the limit,
  one request takes down the whole server after eight minutes of trying. A
  refusal takes 0.07 s.
- If a stream ends without `[DONE]` or a `finish_reason`, treat it as a
  failure. The response is a well-formed 200 in both cases, because nothing
  can write an error into a stream after it starts. Nightsmith reads its own
  streams the same way.
- `response_format` gets a 400. Nothing in nightsmith constrains output to
  JSON or a schema, so nightsmith does not make a promise that it cannot keep.
  Ask for JSON in the prompt and parse the reply. 27 of 30 varied requests
  parsed on the first try.
- `/health` is a hint. The only proof that a request will work is a request
  that worked. Nightsmith checks itself that way, so each checkmark that it
  prints costs a real completion.

Errors use the OpenAI shape on every path (`error.message`, `type`, `param`,
`code`). `nightsmith stop` lets answers in progress finish. That can take the
time of one whole answer, about 2.5 minutes at the defaults. In Activity
Monitor, the server is named `nightsmith-model`.

## Measured on a base M4 / 16 GB

These numbers are for Gemma 4 12B (4-bit) on `mlx_lm.server`. Setup measures
your Mac and reports its own numbers. It does not quote these.

| What | Measured |
|---|---|
| Writing | ~12 tokens/s (about 9 words a second), flat over a 92-minute run |
| Reading a prompt | ~131 tokens/s cold, ~1,900 warm · ~19 s per 2,048 tokens, so 40,000 tokens is six minutes before the first word |
| Memory | 10.2 GB peak, 7.7 GB idle, against Metal's 12.7 GB GPU limit |
| Endurance | 953 requests over 3 hours, zero failures, memory flat |
| Context ceiling | 38,009 tokens answered in 349 s · 43,389 in 406 s · 62,009 fatal. Capped at 40,000 |
| Concurrency ceiling | 3 × 8,000 tokens fine · 4 × 4,000 fatal · 5 × 2,000 fatal. Capped at 3 |
| A conversation | Gets faster as it grows. The first word takes 1.40 s on turn 1 and 0.71 s by turn 4, because the cache reuses the previous turn |

Memory bandwidth sets the speed, not the amount of RAM. The same model is
about 2× faster on an M-series Pro and 4× faster on a Max.

The cache (the saved work from the last prompt) has one slot.
`nightsmith status` sends a real question of its own, and that question takes
the slot. In the middle of a conversation, reuse drops from 157 tokens to 4.
The next first word then takes 2.15 s, up from 0.71 s.

## Known limits

- Nightsmith has no tools, no skills and no system prompt. The model cannot
  read your files, run commands or look anything up, and it says so when you
  ask. The model also introduces itself as Gemma and does not know that it
  runs inside nightsmith.
- Only English was measured. In a smaller language, a model this size can
  write nonsense that looks fluent. It invents words and reverses meanings,
  and nothing in the output shows that.
- The model invents academic citations. It correctly refuses to invent APIs.
- At `temperature = 0`, a prompt that fails will fail the same way every time,
  so a retry cannot help. Raise the temperature for that one prompt.
- Scheduling, job queues and supervision are out of scope for now. If the
  model runs out of memory, the server writes that to its log and exits.
  `nightsmith status` explains it, and `nightsmith start` brings the server
  back. Nothing restarts the server for you.
- Only the 16 GB size was measured. Every other size in `models.toml` is
  arithmetic, and is labeled as arithmetic.

## Building

```sh
go test ./...
GOOS=darwin GOARCH=arm64 go build -o nightsmith .
```

## License

MIT
