package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
)

// Talking to it.
//
// Setup ends by proving a model on this Mac answered a real question, and
// then hands over a port number. This file is the rest of that sentence: a
// way to ask it something without writing an HTTP client first.
//
// `-p` is the machine-facing half, so it behaves like a unix tool rather than
// like a screen: the answer is stdout and nothing else, every remark goes to
// stderr, and it never starts the server on its own — a script that quietly
// spends 45 seconds and 7 GB is a surprise, and `nightsmith start` is one
// word.

// askQuestion reads the question from argv, from stdin, or from both.
//
// stdin is read only when it was asked for — a lone "-" among the arguments,
// or no question in argv at all. The obvious-looking rule, "read stdin
// whenever it isn't a terminal", hangs forever wherever stdin is a pipe that
// nobody ever writes to or closes: a CI runner, a cron job, an agent's shell.
// Found by hanging exactly that way. An ignored pipe is a disappointment; a
// command that never returns is a bug report.
//
//	nightsmith -p "what's a spring tide?"      the question is argv
//	cat notes.txt | nightsmith -p              the question is stdin
//	cat notes.txt | nightsmith -p "sum up" -   the instruction, then the material
func askQuestion(args []string) (string, error) {
	return buildQuestion(args, stdin)
}

func buildQuestion(args []string, in io.Reader) (string, error) {
	var words []string
	wantStdin := false
	for _, a := range args {
		if a == "-" {
			wantStdin = true
			continue
		}
		words = append(words, a)
	}
	argv := strings.TrimSpace(strings.Join(words, " "))

	if !wantStdin && argv != "" {
		return argv, nil
	}

	b, err := io.ReadAll(in)
	if err != nil {
		return "", fmt.Errorf("couldn't read the question from stdin: %w", err)
	}
	piped := strings.TrimSpace(string(b))

	switch {
	case argv != "" && piped != "":
		return argv + "\n\n" + piped, nil
	case piped != "":
		return piped, nil
	default:
		return argv, nil
	}
}

func cmdAsk(args []string) error {
	question, err := askQuestion(args)
	if err != nil {
		return err
	}
	if question == "" {
		return errors.New(`usage: nightsmith -p "your question"  (or pipe one in)`)
	}

	cfg, _, m, err := loaded()
	if err != nil {
		return err
	}
	if _, running := ServerPID(); !running {
		// Deliberately not offered here. See the note at the top of the file.
		fmt.Fprint(os.Stderr, "\n  Not running. 'nightsmith start' turns it on.\n\n")
		return statusNotRunning
	}

	// Ctrl-C stops the answer, not the process mid-write.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// stdout is the answer. Anything that isn't the answer — timings,
	// warnings, the reason it failed — goes to stderr, so a pipe gets clean
	// text and a terminal still gets told what happened.
	remarks := isTerminal(os.Stderr)
	wroteAny := false

	res, err := StreamChat(ctx, cfg.Port, cfg, m, Turn{
		Messages:    []chatMessage{{Role: "user", Content: question}},
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		Kwargs:      ThinkingKwargs(m, cfg.Thinking),
	}, func(s string) {
		wroteAny = true
		fmt.Print(s)
	}, nil)

	if wroteAny && !strings.HasSuffix(res.Answer, "\n") {
		fmt.Println()
	}

	switch {
	case res.Interrupted:
		fmt.Fprint(os.Stderr, "\n  ⚠  Stopped. What's above is all that arrived.\n\n")
		return errors.New("stopped")

	case errors.Is(err, errStreamTruncated):
		// A stream that ends early is a well-formed 200, and the server could
		// not write its own error once it had started. The log is the only
		// evidence there is, so show it.
		fmt.Fprintf(os.Stderr, "\n  ✗  %s — what's above is all that arrived.\n\n%s\n",
			err, indentTail(readLog(), 8))
		return statusNotAnswering

	case err != nil:
		fmt.Fprintf(os.Stderr, "\n  ✗  %s\n\n", err)
		return statusNotAnswering
	}

	if remarks {
		fmt.Fprint(os.Stderr, "\n"+turnFooter(res, cfg)+"\n")
	}
	return nil
}

// turnFooter is what a turn cost, and only what was actually measured. The
// speed is left off a short answer on purpose: dividing two tokens by a cold
// start once had setup reporting a 9 word/s model as a 5 word/s one.
func turnFooter(res StreamResult, c Config) string {
	line := "  " + humanDuration(res.Elapsed.Seconds())
	if res.SpeedWorthPrinting() {
		line += " · " + humanSpeed(res.TokPerSec())
	}
	if res.FinishReason == "length" {
		line += fmt.Sprintf("\n  ⚠  Stopped at the max_tokens limit (%d). Raise it in %s.",
			c.MaxTokens, shortPath(configPath()))
	}
	return line
}

// ── The conversation ────────────────────────────────────────────────────────

// asker is the seam the REPL is tested through: a fake one lets the whole
// loop run headless, with no server, no model and no terminal.
type asker func(ctx context.Context, msgs []chatMessage, onDelta func(string)) (StreamResult, error)

// sessionRecorder writes the conversation down as it happens. A nil one
// records nothing, which is what the tests run against — the loop must not
// need a disk to be exercised.
//
// Nothing it can fail at is worth ending a conversation over: if the disk
// refuses, it says so once and the talking carries on.
type sessionRecorder struct {
	model  string
	out    io.Writer
	cur    *Session
	warned bool
}

func (r *sessionRecorder) record(m SessionMessage) {
	if r == nil || r.cur == nil {
		return
	}
	if err := r.cur.Append(m); err != nil {
		r.warn(err)
	}
}

func (r *sessionRecorder) fresh() {
	if r == nil {
		return
	}
	s, err := NewSession(r.model)
	if err != nil {
		r.cur = nil
		r.warn(err)
		return
	}
	r.cur = s
}

func (r *sessionRecorder) use(s *Session) {
	if r != nil {
		r.cur = s
	}
}

func (r *sessionRecorder) warn(err error) {
	if r.warned {
		return
	}
	r.warned = true
	fmt.Fprintf(r.out, "\n  ⚠  Couldn't write this conversation down: %v\n"+
		"     Talking still works — it just won't be here tomorrow.\n\n", err)
}

const chatBanner = `
  nightsmith — %s, on this Mac.
  Nothing you type here leaves it.
  /help for commands · Ctrl-D to leave · no arrow-key history yet

`

const chatHelp = `
    /help       this
    /new        start a fresh conversation
    /sessions   the conversations on this Mac
    /resume ID  pick one up again
    /context    how much of the conversation is re-read each turn
    /paste      paste several lines as one message, ending with a lone .
    /exit       leave  (Ctrl-D does the same)

  Anything else is a message. To send a line that starts with /, put a
  space in front of it.

`

// cmdChat opens the conversation. A nil `start` means a fresh one; anything
// else is a transcript being picked up where it was left.
func cmdChat(start *Session) error {
	cfg, _, m, err := loaded()
	if err != nil {
		return err
	}
	if _, running := ServerPID(); !running {
		printf("\n  Not running.\n\n")
		if !confirm("  Start it?", true) {
			fmt.Println()
			return statusNotRunning
		}
		if err := startForChat(&cfg, m); err != nil {
			return err
		}
	}

	rec := &sessionRecorder{model: cfg.Model, out: os.Stdout}
	var history []chatMessage
	if start != nil {
		rec.use(start)
		history = start.Conversation()
	} else {
		rec.fresh()
	}

	return runChat(stdin, os.Stdout, shortRepo(cfg.Model),
		func(ctx context.Context, msgs []chatMessage, onDelta func(string)) (StreamResult, error) {
			return StreamChat(ctx, cfg.Port, cfg, m, Turn{
				Messages:  msgs,
				MaxTokens: cfg.MaxTokens,
				// Not cfg.Temperature: 0 is right for the batch workload the
				// default was chosen for, and wrong for a conversation, where
				// rephrasing a question would return the same answer worded
				// identically. `-p` keeps 0, because a script wants the
				// reproducible one.
				Temperature: cfg.ChatTemperature,
				Kwargs:      ThinkingKwargs(m, cfg.Thinking),
			}, onDelta, nil)
		}, rec, history, WindowPolicy{BudgetTokens: cfg.ChatContext})
}

// cmdResume picks up a past conversation. With no id it takes the most
// recent; with an ambiguous or missing one it says so rather than guessing.
func cmdResume(id string) error {
	if !ConfigExists() {
		return errors.New("nightsmith isn't set up on this Mac yet — run 'nightsmith'")
	}
	if id == "" {
		s, err := LatestSession()
		if err != nil {
			return err
		}
		return cmdChat(s)
	}
	s, err := OpenSession(id)
	if err != nil {
		return err
	}
	return cmdChat(s)
}

// cmdSessions lists what is on this Mac. It is a flag, not a verb: the
// surface stays at eight.
func cmdSessions() error {
	list, err := ListSessions()
	if err != nil {
		return err
	}
	fmt.Print(sessionList(list))
	return nil
}

func sessionList(list []SessionInfo) string {
	if len(list) == 0 {
		return "\n  No conversations yet. 'nightsmith' starts one.\n\n"
	}
	var b strings.Builder
	b.WriteString("\n")
	for _, s := range list {
		b.WriteString(fmt.Sprintf("    %s  %s  %-4s  %s\n",
			s.ID, s.When.Format("2 Jan 15:04"),
			fmt.Sprintf("%dt", s.Turns), s.Title))
	}
	b.WriteString("\n  'nightsmith -r ID' picks one up.\n\n")
	return b.String()
}

// startForChat starts the server the way `start` does, including the rule
// that matters: the prompt is not drawn until a real answer has come back. An
// open port is not a started model, and a conversation that begins with 45
// seconds of silence looks broken.
func startForChat(cfg *Config, m Model) error {
	port := cfg.Port
	pid, err := StartServer(cfg, m)
	if err != nil {
		return err
	}
	if cfg.Port != port {
		if err := WriteConfig(*cfg); err != nil {
			return err
		}
	}
	printf("\n  Loading the model — about 45 s the first time…\n")
	if _, err := Probe(cfg.Port, *cfg, m, "Reply with the single word: ready"); err != nil {
		StopServer() // never leave 7 GB resident with nothing working
		return fmt.Errorf("the model started but couldn't answer: %w\n\n%s", err, indentTail(readLog(), 8))
	}
	printf("  ✓  Started, and it answered. pid %d\n", pid)
	return nil
}

func runChat(in *bufio.Reader, out io.Writer, modelName string, ask asker,
	rec *sessionRecorder, history []chatMessage, policy WindowPolicy) error {
	fmt.Fprintf(out, chatBanner, modelName)
	if len(history) > 0 {
		fmt.Fprint(out, recap(history))
	}

	var lastCached int

	for {
		line, err := readMessage(in, out)
		if err == io.EOF {
			fmt.Fprint(out, "\n")
			return nil
		}
		if err != nil {
			return err
		}
		if strings.TrimSpace(line) == "" {
			continue
		}

		if cmd, arg, isCmd := parseMeta(line); isCmd {
			switch cmd {
			case "/exit", "/quit":
				return nil
			case "/help":
				fmt.Fprint(out, chatHelp)
			case "/new":
				history, lastCached = nil, 0
				rec.fresh()
				fmt.Fprint(out, "\n  A fresh conversation.\n\n")
			case "/sessions":
				list, err := ListSessions()
				if err != nil {
					fmt.Fprintf(out, "\n  ✗  %s\n\n", err)
					continue
				}
				fmt.Fprint(out, sessionList(list))
			case "/resume":
				s, err := OpenSession(arg)
				if err != nil {
					fmt.Fprintf(out, "\n  ✗  %s\n\n", err)
					continue
				}
				history, lastCached = s.Conversation(), 0
				rec.use(s)
				fmt.Fprint(out, recap(history))
			case "/context":
				fmt.Fprint(out, contextReport(history, lastCached, policy))
			case "/paste":
				pasted, err := readPaste(in, out)
				if err == io.EOF {
					fmt.Fprint(out, "\n")
					return nil
				}
				if strings.TrimSpace(pasted) == "" {
					continue
				}
				line = pasted
				goto send
			default:
				fmt.Fprintf(out, "\n  %s isn't a command. /help lists them; put a space in\n"+
					"  front of the line to send it as a message.\n\n", cmd)
			}
			_ = arg
			continue
		}
		// A leading space is how you send a line that starts with a slash.
		line = strings.TrimPrefix(line, " ")

	send:
		history = append(history, chatMessage{Role: "user", Content: line})

		// Ctrl-C belongs to the turn, not to the prompt. At an idle prompt the
		// default action stands, so Ctrl-C leaves — which is what it should
		// do, and what a handler could not achieve anyway while a cooked-mode
		// read is blocking.
		// What is actually sent. The transcript on disk keeps everything;
		// this is only what fits in one request.
		window, trim := BuildWindow(history, policy)
		fmt.Fprint(out, trim.Notice())

		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		ind := &indenter{w: out, prefix: "  "}
		fmt.Fprint(out, "\n")
		res, err := ask(ctx, window, func(s string) { ind.WriteString(s) })
		stop()
		ind.finish()

		switch {
		case res.Interrupted:
			fmt.Fprint(out, "\n  ⚠  Stopped. Keeping what arrived.\n\n")
			// Kept on purpose: it is on the screen, and dropping it would
			// break the prefix the next turn extends.
			history = append(history, chatMessage{Role: "assistant", Content: res.Answer})
			rec.record(SessionMessage{Role: "user", Content: line})
			rec.record(SessionMessage{Role: "assistant", Content: res.Answer,
				Interrupted: true, MS: res.Elapsed.Milliseconds()})

		case errors.Is(err, errStreamTruncated):
			fmt.Fprintf(out, "\n  ✗  %s — what's above is all that arrived.\n\n%s\n\n",
				err, indentTail(readLog(), 8))
			history = history[:len(history)-1] // the turn didn't happen

		case err != nil:
			fmt.Fprintf(out, "\n  ✗  %s\n\n", err)
			history = history[:len(history)-1]

		default:
			history = append(history, chatMessage{Role: "assistant", Content: res.Answer})
			lastCached = res.CachedTokens
			// Written down only now: a turn that failed did not happen, and
			// a half-recorded exchange would come back as context tomorrow.
			rec.record(SessionMessage{Role: "user", Content: line})
			rec.record(SessionMessage{Role: "assistant", Content: res.Answer,
				Tokens: res.TokensOut, MS: res.Elapsed.Milliseconds()})
			fmt.Fprintf(out, "\n%s\n\n", turnFooterNoConfig(res))
		}
	}
}

// readMessage reads one line, joining any that end in a backslash.
func readMessage(in *bufio.Reader, out io.Writer) (string, error) {
	var b strings.Builder
	for {
		fmt.Fprint(out, "  › ")
		line, err := in.ReadString('\n')
		if err != nil && line == "" {
			return "", io.EOF
		}
		line = strings.TrimRight(line, "\r\n")
		if cont, ok := strings.CutSuffix(line, "\\"); ok {
			b.WriteString(cont + "\n")
			continue
		}
		b.WriteString(line)
		return b.String(), nil
	}
}

// readPaste collects lines until one containing only a dot. It exists because
// a cooked-mode terminal delivers a pasted block as separate lines, which
// would otherwise fire one turn per line. Raw mode and bracketed paste would
// fix that properly; this is the honest workaround until then.
func readPaste(in *bufio.Reader, out io.Writer) (string, error) {
	fmt.Fprint(out, "\n  Paste away. A line with a single . on it ends the message.\n\n")
	var b strings.Builder
	for {
		fmt.Fprint(out, "  │ ")
		line, err := in.ReadString('\n')
		if err != nil && line == "" {
			return b.String(), io.EOF
		}
		line = strings.TrimRight(line, "\r\n")
		if strings.TrimSpace(line) == "." {
			return strings.TrimRight(b.String(), "\n"), nil
		}
		b.WriteString(line + "\n")
	}
}

// parseMeta treats a line as a command only when it starts with a slash. An
// unknown slash-word is reported rather than sent, because a typo'd command
// silently becoming a question is worse than being told.
func parseMeta(line string) (cmd, arg string, ok bool) {
	if !strings.HasPrefix(line, "/") {
		return "", "", false
	}
	cmd, arg, _ = strings.Cut(strings.TrimSpace(line), " ")
	return cmd, strings.TrimSpace(arg), true
}

// recap is what Hermes Agent gets right about resuming: dropping someone back
// at a bare prompt tells them nothing about which conversation they are in.
// The last exchange is enough to recognise it.
func recap(history []chatMessage) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("  Picking up where you left off — %d message%s.\n",
		len(history), plural(len(history))))
	from := len(history) - 2
	if from < 0 {
		from = 0
	}
	for _, m := range history[from:] {
		who := "you"
		if m.Role == "assistant" {
			who = "it"
		}
		b.WriteString(fmt.Sprintf("    %-3s %s\n", who, firstLine(m.Content, 60)))
	}
	b.WriteString("\n")
	return b.String()
}

func contextReport(history []chatMessage, lastCached int, policy WindowPolicy) string {
	turns := 0
	words := 0
	for _, m := range history {
		if m.Role == "user" {
			turns++
		}
		words += len(strings.Fields(m.Content))
	}
	s := fmt.Sprintf("\n  %d turn%s · about %d words in this conversation.\n", turns, plural(turns), words)

	if policy.BudgetTokens > 0 {
		_, trim := BuildWindow(history, policy)
		// Marked as an estimate wherever it appears: this is characters ÷ 4,
		// not the model's tokenizer.
		switch {
		case trim.DroppedExchanges > 0 && trim.OverBudget:
			s += fmt.Sprintf("  The last exchange alone is about %d tokens, over the %d\n"+
				"  budget, so it is being sent on its own — the %d before it are\n"+
				"  not. (Estimated from characters, not counted.)\n",
				trim.EstimatedTokens, policy.BudgetTokens, trim.DroppedExchanges)
		case trim.DroppedExchanges > 0:
			s += fmt.Sprintf("  About %d tokens sent each turn, against a %d budget; the\n"+
				"  oldest %d exchange%s no longer fit. (Estimated, not counted.)\n",
				trim.EstimatedTokens, policy.BudgetTokens,
				trim.DroppedExchanges, plural(trim.DroppedExchanges))
		default:
			s += fmt.Sprintf("  All of it is re-read each turn: roughly %d tokens of a %d\n"+
				"  budget. (Estimated from characters, not counted.)\n",
				trim.EstimatedTokens, policy.BudgetTokens)
		}
	}
	if lastCached > 0 {
		s += fmt.Sprintf("  Last turn, the server reused %d cached tokens of the prompt.\n", lastCached)
	}
	// One slot, and every client on the port shares it — including `status`,
	// which sends a real completion of its own.
	s += "  The prompt cache has one slot, shared with anything else using\n" +
		"  the port: a 'nightsmith status' between turns costs this\n" +
		"  conversation its head start.\n\n"
	return s
}

func turnFooterNoConfig(res StreamResult) string {
	line := "  " + humanDuration(res.Elapsed.Seconds())
	if res.SpeedWorthPrinting() {
		line += " · " + humanSpeed(res.TokPerSec())
	}
	if res.FinishReason == "length" {
		line += "\n  ⚠  Stopped at the token limit. Raise max_tokens in " + shortPath(configPath()) + "."
	}
	return line
}

// indenter puts the house two-space margin on streamed text, which arrives a
// few characters at a time and knows nothing about lines.
type indenter struct {
	w       io.Writer
	prefix  string
	started bool
	pending bool // a newline was written; indent before the next character
}

func (i *indenter) WriteString(s string) {
	for _, r := range s {
		if !i.started || i.pending {
			fmt.Fprint(i.w, i.prefix)
			i.started, i.pending = true, false
		}
		fmt.Fprintf(i.w, "%c", r)
		if r == '\n' {
			i.pending = true
		}
	}
}

func (i *indenter) finish() {
	if i.started && !i.pending {
		fmt.Fprint(i.w, "\n")
	}
}
