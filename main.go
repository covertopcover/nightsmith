package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"
)

// The command surface:
//
//	nightsmith                 set up · talk to it · or report setup is done
//	nightsmith -p "…"          one question, one answer — for scripts
//	nightsmith -c | -r [ID]    pick a past conversation back up
//	nightsmith start|stop      the server
//	nightsmith status [--json] proved by a real completion, never a ping
//	nightsmith remove          take it back off this Mac  (alias: uninstall)
//	nightsmith config check    estimate peak memory against this Mac's ceiling
//	nightsmith model list|use  what fits, with sizes
//
// Nothing else. Every feature outside setup would add verbs here.

// Set at release time: go build -ldflags "-X main.version=v0.1.0".
var version = "0.0.0-dev"

func main() {
	args := os.Args[1:]
	if err := run(args); err != nil {
		var code exitCode
		if errors.As(err, &code) {
			os.Exit(int(code)) // already reported; the code is the message
		}
		fmt.Fprint(os.Stderr, colorize(fmt.Sprintf("\n  ✗ %s\n\n", err), colorOn(os.Stderr)))
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return cmdSetup()
	}
	switch args[0] {
	case "-p", "--print":
		return cmdAsk(args[1:])
	case "-c", "--continue":
		return cmdResume("")
	case "-r", "--resume":
		// With no id: show what there is, rather than guessing which one.
		if len(args) > 1 {
			return cmdResume(args[1])
		}
		return cmdSessions()
	case "start":
		return cmdStart()
	case "stop":
		return cmdStop()
	case "status":
		return cmdStatus(hasFlag(args, "--json"))
	case "remove", "uninstall":
		// `remove` is the name; `uninstall` is what people type, because it is
		// the verb every other tool uses. Making them guess right is a
		// pointless tax, so it is accepted silently.
		return cmdRemove(hasFlag(args, "--yes", "-y"))
	case "config":
		if len(args) > 1 && args[1] == "check" {
			return cmdConfigCheck()
		}
		return fmt.Errorf("unknown: nightsmith config %s (did you mean 'config check'?)", strings.Join(args[1:], " "))
	case "model":
		if len(args) > 1 && args[1] == "list" {
			return cmdModelList()
		}
		if len(args) > 2 && args[1] == "use" {
			return cmdModelUse(args[2])
		}
		return errors.New("usage: nightsmith model list | nightsmith model use <repo>")
	case "version", "--version", "-v":
		printf("nightsmith %s\n", version)
		return nil
	case "help", "--help", "-h":
		printHelp()
		return nil
	default:
		printHelp()
		return fmt.Errorf("unknown command: %s", args[0])
	}
}

func printHelp() {
	fmt.Print(`
  nightsmith — AI that runs on your own Mac.

    nightsmith                set it up — or, once it is, talk to it
                              (with no terminal: report that setup is done)
    nightsmith start          turn it on
    nightsmith stop           turn it off
    nightsmith status         is it running, and how much memory
    nightsmith remove         take it back off this Mac

    nightsmith -p "…"         ask one question, print the answer, exit.
                              With no question, or with a lone '-', the
                              question is read from stdin instead:
                                cat notes.txt | nightsmith -p "sum this up" -
                              Same exit codes as status

    nightsmith -c             pick up the last conversation
    nightsmith -r [ID]        pick up that one — or list them all

    nightsmith status --json  the same, for programs. Exit code, either way:
                              0 it answered · 3 not running ·
                              4 running but not answering · 1 anything else

    nightsmith config check   what the settings in ~/.nightsmith/config.toml
                              will cost, against what this Mac can give them
    nightsmith model list     what fits, what doesn't, with sizes
    nightsmith model use R    switch model, and prove the new one answers

`)
}

// ── Screen 1: inspect, decide, ask once ─────────────────────────────────────
// Everything above the question is work, not interrogation. The tool earns the
// right to ask by doing something first.

func cmdSetup() error {
	cat, err := LoadCatalog()
	if err != nil {
		return err
	}
	machine, err := Detect()
	if err != nil {
		return err
	}
	if refusal := machine.Refusal(); refusal != "" {
		return errors.New(refusal)
	}

	if ConfigExists() {
		// Set up already. At a terminal that means: talk to it — the bare
		// command is the conversation, which is what a person who just
		// installed this wants and what every other terminal agent does.
		//
		// With no terminal it keeps its old meaning exactly: re-running a
		// command to check it worked is what non-developers do, it must be
		// instant and idempotent, and install.sh's non-interactive ending and
		// every script depend on the exit codes. A conversation needs a
		// terminal; nothing else here does, which is what makes the split
		// safe rather than clever.
		var code exitCode
		if isConsole(os.Stdin) && isConsole(os.Stdout) {
			if err := cmdChat(nil); err != nil && !errors.As(err, &code) {
				return err
			}
			return nil
		}
		// Not running is a fine answer here, not a failure.
		if err := cmdStatus(false); err != nil && !errors.As(err, &code) {
			return err
		}
		return nil
	}

	printf("\n  Nightsmith — AI that runs on your own Mac.\n\n  Looking at this Mac…\n\n")
	printf("  ✓  %s", machine.Chip)
	if machine.GPUCores > 0 {
		printf(" · %d GPU cores", machine.GPUCores)
	}
	printf(" · ~%.0f GB/s memory\n", machine.BandwidthGB)
	printf("  ✓  macOS %s\n", machine.MacOS)
	printf("  ✓  %d GB memory — the GPU can use %.1f GB of it", machine.RAMGB, machine.CeilingGB)
	if !machine.CeilingIsMeasured {
		printf("   (estimated)")
	}
	fmt.Println()
	m, ok := cat.PickFor(machine.RAMGB)
	if !ok {
		return fmt.Errorf("no model in the table fits %d GB of memory", machine.RAMGB)
	}

	// A checkmark on disk space has to mean the space was compared with what
	// setup is about to use, not merely that it was read.
	need := DiskNeeded(m, DefaultConfig(cat, m))
	// Plus room to spare: a Mac with its disk filled to the last megabyte by
	// this download is a broken Mac, whatever the model does.
	const spare = 2e9
	if short := float64(need) + spare - machine.DiskFreeGB*bytesPerGB; short > 0 {
		return fmt.Errorf("setup needs %s on disk, plus 2 GB left free for macOS.\n"+
			"    This Mac has %.1f GB free.\n\n"+
			"    Free up %.1f GB and run 'nightsmith' again. Nothing was installed.",
			humanBytes(need), machine.DiskFreeGB, short/bytesPerGB)
	}
	printf("  ✓  %.0f GB free on disk — setup needs %s\n", machine.DiskFreeGB, humanBytes(need))
	est := EstimatePeak(m, cat.Defaults.PromptCacheMB, machine.RAMGB, machine.CeilingGB)
	if !est.Fits {
		return errors.New(RefuseReason(m, est, machine.RAMGB))
	}

	if machine.RAMGB <= 8 {
		// Say the quiet part. It is supported precisely because the workload is
		// many small tasks; that would not hold for agent work.
		printf("\n  ⚠  16 GB is where this gets comfortable. You have %d.\n\n", machine.RAMGB)
		printf("     I'll set up a smaller model. For short, simple tasks it's\n")
		printf("     fine. It will compete with your browser for memory.\n")
	}

	printf("\n  Here's what fits, and what it'll be like:\n\n")
	if _, _, here := LocateModel(m, DefaultConfig(cat, m)); here {
		printf("     %s  ·  %s, already on this Mac\n\n",
			shortRepo(m.Repo), humanBytes(m.TotalBytes()))
	} else {
		printf("     %s  ·  %s download\n\n",
			shortRepo(m.Repo), humanBytes(m.TotalBytes()))
	}
	printf("     Uses %.1f GB while working, leaving %.1f GB spare.\n",
		est.PeakGB, est.HeadroomGB)

	speed := 12.6 * SpeedFactor(machine.BandwidthGB)
	if m.IsMeasured() && machine.BandwidthGB == 120 {
		printf("     Writes %s — steady, not fast.\n", humanSpeed(m.Measured.DecodeTokS))
	} else {
		// Everything here is extrapolated from one measured machine, and says so.
		printf("     Should write %s — estimated from this Mac's\n", humanSpeed(speed))
		printf("     memory bandwidth, then measured for real below.\n")
	}

	printf("\n  Nothing here needs your password, and nothing touches the\n")
	printf("  Python or Homebrew already on this Mac.\n\n")

	if !confirm("  Set it up?", true) {
		fmt.Println("\n  Nothing was installed.")
		return nil
	}
	return runSetup(cat, m, machine)
}

// ── Screen 2: install, and earn the checkmarks ──────────────────────────────

func runSetup(cat *Catalog, m Model, machine Machine) error {
	cfg := DefaultConfig(cat, m)
	port, err := FreePort(cfg.Port)
	if err != nil {
		return err
	}
	cfg.Port = port
	fmt.Println()

	if err := ensureRuntime(); err != nil {
		return err
	}
	if _, err := ensureModel(m, cfg); err != nil {
		return err
	}

	started := time.Now()
	pid, err := StartServer(&cfg, m)
	if err != nil {
		return err
	}
	// If anything below fails, setup is not done, and a 7 GB process must not
	// be left resident with no config pointing at it.
	ok := false
	defer func() {
		if !ok {
			StopServer()
		}
	}()
	printf("%s", stepLine("Started the model", since(started)))

	// The most important line in the product. A checkmark may only appear
	// after a real completion came back — /v1/models returned 200 while the
	// server could serve nothing, in three different ways.
	mem := WatchMemory(pid)
	res, err := Probe(cfg.Port, cfg, m, "Say hello in one short sentence.")
	peak := mem.Stop()
	if err != nil {
		return fmt.Errorf("the model started but couldn't answer: %w\n\n%s", err, indentTail(readLog(), 8))
	}
	printf("  ✓  Asked it something — it answered:\n        %q\n", res.Answer)
	if res.ThinkingKnown && res.ThinkingOff {
		printf("  ✓  Thinking stayed off\n")
	}
	// Speed is measured on a second, longer answer. The first one's time is
	// mostly loading the model, and "Hello!" is two tokens: dividing one by
	// the other printed "5 words a second" for a model that writes 9.
	speed, err := Probe(cfg.Port, cfg, m, "Describe a quiet harbour at dawn in about eighty words.")
	if err != nil {
		return fmt.Errorf("the model answered once, then couldn't answer again: %w", err)
	}
	peak = mem.Stop()
	printf("  ✓  Measured it here:  %s · %s\n", humanSpeed(speed.TokPerSec), peak)

	if err := WriteConfig(cfg); err != nil {
		return err
	}
	ok = true
	printReady(cfg)
	return nil
}

// ── Screen 3: the handoff ───────────────────────────────────────────────────

func printReady(c Config) {
	n := CommandName(os.Getenv("NIGHTSMITH_PATH_STATE"))
	printf(`
  Ready. It's running now.

    %[1]s           talk to it
    %[1]s start     turn it on
    %[1]s stop      turn it off
    %[1]s status    is it running, and how much memory
    %[1]s remove    take it back off this Mac

`, n)
	printEndpoint(c)
	printf(`
  Settings: %s — each one explained.
  '%s config check' shows what they cost.

`, shortPath(configPath()), n)
}

// printEndpoint is what a program pointed at the server needs: where it is,
// and the model id to send. /v1/models says the same, but only to someone who
// already knows to ask.
func printEndpoint(c Config) {
	printf("  While it's on, it's at  http://127.0.0.1:%d/v1\n", c.Port)
	printf("  OpenAI-compatible · model %q, or leave it out\n", c.Model)
}

// CommandName is how the last screen spells the command, so that every line it
// prints works when typed — without telling anyone to source anything.
//
// A PATH line the installer just added reaches future shells only. The
// installer's answer is to hand over a fresh login shell, and it says so with
// NIGHTSMITH_PATH_STATE:
//
//	fresh-shell  a login shell follows: plain `nightsmith` will work
//	full-path    no shell follows (CI, no terminal): print the full path,
//	             which needs no PATH at all
//
// Unset means PATH already had ~/.local/bin, and the short name is right.
func CommandName(state string) string {
	if state == "full-path" {
		return shortPath(binPath()) // ~/.local/bin/nightsmith — tilde expands
	}
	return "nightsmith"
}

// ── The rest of the surface ─────────────────────────────────────────────────

func cmdStart() error {
	cfg, cat, m, err := loaded()
	if err != nil {
		return err
	}
	port := cfg.Port
	started := time.Now()
	pid, err := StartServer(&cfg, m)
	if err != nil {
		return err
	}
	_ = cat
	if cfg.Port != port {
		// Something else took the port since setup. Moving is quiet;
		// the new one is saved so status and clients find it.
		if err := WriteConfig(cfg); err != nil {
			return err
		}
	}

	// An open port is not a started model: the weights load after it opens,
	// and a client that connects then waits ~45 s with nothing to tell it
	// why. "Started" is printed only once a real answer has come back.
	printf("\n  Loading the model — about 45 s the first time…\n")
	if _, err := Probe(cfg.Port, cfg, m, "Reply with the single word: ready"); err != nil {
		StopServer() // never leave 7 GB resident with nothing working
		return fmt.Errorf("the model started but couldn't answer: %w\n\n%s", err, indentTail(readLog(), 8))
	}
	printf("  ✓  Started, and it answered (%s). pid %d\n\n", humanDuration(time.Since(started).Seconds()), pid)
	printEndpoint(cfg)
	printConfigWarnings(true)
	fmt.Println()
	return nil
}

func printConfigWarnings(running bool) {
	for _, w := range ConfigWarnings(running) {
		printf("\n  ⚠  %s\n", w)
	}
}

func cmdStop() error {
	// A stop lets requests in progress finish. Say so, or a pause of up to
	// half a minute reads as a hang.
	if cfg, err := LoadConfig(); err == nil {
		if _, ok := ServerPID(); ok {
			if n := InFlight(cfg.Port); n > 0 {
				printf("\n  Finishing %d request%s in progress (up to %d s)…\n",
					n, plural(n), drainSeconds)
			}
		}
	}
	stopped, err := StopServer()
	if err != nil {
		return err
	}
	if !stopped {
		fmt.Println("\n  Nothing was running.")
		return nil
	}
	fmt.Println("\n  Stopped.")
	return nil
}

// exitCode ends the program with that code and nothing printed: the command
// has already said what happened.
type exitCode int

func (c exitCode) Error() string { return fmt.Sprintf("exit status %d", int(c)) }

// status's exit codes, the same with or without --json.
const (
	statusNotRunning   exitCode = 3
	statusNotAnswering exitCode = 4
)

// StatusReport is `status --json`. State is "ready" only after a real answer.
type StatusReport struct {
	State       string `json:"state"` // ready | not_running | not_answering
	Model       string `json:"model"`
	Port        int    `json:"port"`
	URL         string `json:"url"`
	PID         int    `json:"pid,omitempty"`
	MemoryBytes int64  `json:"memory_bytes,omitempty"`
	Error       string `json:"error,omitempty"`
}

func newStatusReport(cfg Config, pid int, probeErr error, memory int64) StatusReport {
	r := StatusReport{Model: cfg.Model, Port: cfg.Port,
		URL: fmt.Sprintf("http://127.0.0.1:%d/v1", cfg.Port)}
	switch {
	case pid == 0:
		r.State = "not_running"
	case probeErr != nil:
		r.State, r.PID, r.Error = "not_answering", pid, probeErr.Error()
	default:
		r.State, r.PID, r.MemoryBytes = "ready", pid, memory
	}
	return r
}

func (r StatusReport) exit() error {
	switch r.State {
	case "not_running":
		return statusNotRunning
	case "not_answering":
		return statusNotAnswering
	}
	return nil
}

func printJSON(v any) {
	b, _ := json.MarshalIndent(v, "", "  ")
	fmt.Println(string(b))
}

// cmdStatus must use a real completion, not a ping, or it will cheerfully
// report a post-OOM zombie as healthy.
func cmdStatus(asJSON bool) error {
	cfg, _, m, err := loaded()
	if err != nil {
		return err
	}
	pid, ok := ServerPID()
	if !ok {
		r := newStatusReport(cfg, 0, nil, 0)
		if asJSON {
			printJSON(r)
		} else {
			printf("\n  Set up, but not running.\n\n    nightsmith start    turn it on\n\n")
		}
		return r.exit()
	}

	if !asJSON {
		printf("\n  Running (pid %d). Asking it something…\n", pid)
	}
	res, err := Probe(cfg.Port, cfg, m, "Reply with the single word: ready")
	var now int64
	if err == nil {
		now, _, _ = Footprint(pid)
	}
	r := newStatusReport(cfg, pid, err, now)
	if asJSON {
		printJSON(r)
		return r.exit()
	}
	if err != nil {
		// This is the case a health check gets wrong. Say it plainly.
		printf("\n  ⚠  The server is up but could not answer:\n     %s\n\n", err)
		printf("     That usually means it ran out of memory. 'nightsmith stop'\n")
		printf("     then 'nightsmith start' will clear it.\n\n")
		return r.exit()
	}
	printf("  ✓  It answered: %q\n", res.Answer)
	// No speed here: a one-word reply measures latency, not writing speed.
	// No peak either. The kernel's lifetime peak footprint is a different
	// measure from the GPU ceiling config check compares against, and printed
	// beside it (14.4 GB against 12.7) it read as a limit already broken.
	printf("  ✓  %s · port %d · using %s now\n\n",
		shortRepo(cfg.Model), cfg.Port, humanBytes(now))
	return nil
}

func cmdConfigCheck() error {
	cfg, _, m, err := loaded()
	if err != nil {
		return err
	}
	machine, err := Detect()
	if err != nil {
		return err
	}
	e := EstimatePeak(m, cfg.PromptCacheMB, machine.RAMGB, machine.CeilingGB)

	printf("\n  model               %s weights\n", humanBytes(m.WeightsBytes))
	printf("  prompt_cache_mb     %.1f GB\n", e.CacheGB)
	printf("  overhead            ~%.1f GB\n\n", e.OverheadGB)
	printf("  Estimated peak      %.1f GB\n", e.PeakGB)
	printf("  This Mac's ceiling  %.1f GB", e.CeilingGB)
	if e.Fits {
		printf("          ✓ %.1f GB spare\n", e.HeadroomGB)
	} else {
		printf("          ✗ %.1f GB short\n", -e.HeadroomGB)
	}
	if !machine.CeilingIsMeasured {
		printf("\n  The ceiling is the 75%% rule, not this Mac's reported value,\n")
		printf("  and it runs slightly optimistic. Treat a margin under\n")
		printf("  0.5 GB as no margin.\n")
	}
	if !e.Fits {
		printf("\n  ✗ %s\n", RefuseReason(m, e, machine.RAMGB))
	}
	if w := ThinkingConflict(cfg.Thinking, cfg.MaxTokens); w != "" {
		printf("\n  ⚠  %s\n", w)
	}
	_, running := ServerPID()
	printConfigWarnings(running)
	fmt.Println()
	return nil
}

func cmdModelList() error {
	cat, err := LoadCatalog()
	if err != nil {
		return err
	}
	machine, err := Detect()
	if err != nil {
		return err
	}
	current := ""
	if cfg, err := LoadConfig(); err == nil {
		current = cfg.Model
	}

	fmt.Println()
	for _, m := range cat.Ordered() {
		e := EstimatePeak(m, cat.Defaults.PromptCacheMB, machine.RAMGB, machine.CeilingGB)
		marker, note := "   ", ""
		switch {
		case m.Repo == current:
			marker = "  →"
			note = "current"
			if m.IsMeasured() {
				note += " · measured here"
			}
		case !WeightsFit(m, machine.RAMGB, cat.Defaults.WeightsRAMFraction):
			note = fmt.Sprintf("✗ needs %d GB", m.MinRAMGB)
		default:
			note = fmt.Sprintf("fits · %.1f GB spare", e.HeadroomGB)
		}
		printf("%s  %-40s %9s   %s\n", marker, m.Repo, humanBytes(m.TotalBytes()), note)
	}
	printf("\n  Sizes are shown because the names give no warning: every row\n")
	printf("  above says -4bit or -8bit, and they span %s to %s.\n\n",
		humanBytes(cat.Ordered()[0].TotalBytes()),
		humanBytes(cat.Ordered()[len(cat.Ordered())-1].TotalBytes()))
	return nil
}

func cmdModelUse(repo string) error {
	cat, err := LoadCatalog()
	if err != nil {
		return err
	}
	machine, err := Detect()
	if err != nil {
		return err
	}
	cfg, err := LoadConfig()
	if err != nil {
		return fmt.Errorf("nightsmith isn't set up yet — run 'nightsmith' first")
	}

	m, known := cat.Find(repo)
	if !known {
		// It takes a real repo id, because that is what it is. But nothing
		// about an unlisted model has been checked, and saying so is the
		// difference between this and "just pick any model from Hugging Face".
		printf("\n  ⚠  %s isn't in the checked list.\n\n", repo)
		printf("     Its size, whether thinking can be turned off, and which\n")
		printf("     server flags break on its architecture are all unknown.\n")
		printf("     It may not load at all.\n\n")
		if !confirm("  Try it anyway?", false) {
			return nil
		}
		m = Model{Repo: repo, Status: "unchecked"}
	} else {
		e := EstimatePeak(m, cfg.PromptCacheMB, machine.RAMGB, machine.CeilingGB)
		if !e.Fits {
			return errors.New(RefuseReason(m, e, machine.RAMGB))
		}
	}

	if _, why := ThinkingHonest(m, cfg.Thinking); why != "" {
		printf("\n  ⚠  %s\n", why)
	}

	// A swap is not done until the new model has actually answered.
	if _, err := StopServer(); err != nil {
		return err
	}
	cfg.Model = m.Repo
	if _, err := ensureModel(m, cfg); err != nil {
		return err
	}
	if _, err := StartServer(&cfg, m); err != nil {
		return err
	}
	res, err := Probe(cfg.Port, cfg, m, "Say hello in one short sentence.")
	if err != nil {
		return fmt.Errorf("%s started but couldn't answer, so the switch was not saved: %w",
			shortRepo(repo), err)
	}
	if err := WriteConfig(cfg); err != nil {
		return err
	}
	printf("\n  ✓  %s answered: %q\n\n", shortRepo(repo), res.Answer)
	return nil
}

func cmdRemove(assumeYes bool) error {
	r := PlanRemoval()
	fmt.Print(r.Describe())
	if r.TotalBytes == 0 {
		return nil
	}
	// Default to no. Setup asks to add; this asks to destroy.
	if !assumeYes && !confirm("\n  Remove all of it?", false) {
		fmt.Println("\n  Nothing was removed.")
		return nil
	}
	freed, err := r.Execute()
	if err != nil {
		return err
	}
	printf("\n  ✓ Removed %s. Your Mac is back to how it was.\n\n", humanBytes(freed))
	return nil
}

// ── Helpers ─────────────────────────────────────────────────────────────────

func loaded() (Config, *Catalog, Model, error) {
	cat, err := LoadCatalog()
	if err != nil {
		return Config{}, nil, Model{}, err
	}
	cfg, err := LoadConfig()
	if err != nil {
		return Config{}, nil, Model{},
			errors.New("nightsmith isn't set up on this Mac yet — run 'nightsmith'")
	}
	m, ok := cat.Find(cfg.Model)
	if !ok {
		m = Model{Repo: cfg.Model, Status: "unchecked"}
	}
	return cfg, cat, m, nil
}

// One reader for the whole process, never one per call. A bufio.Reader may
// read further than the line it returns, so a second one built later starts
// with whatever the first swallowed. With a single prompt that never showed;
// the moment a prompt is followed by a conversation, the user's first message
// is the thing that disappears.
var stdin = bufio.NewReader(os.Stdin)

func confirm(prompt string, defaultYes bool) bool {
	suffix := " [y/N] "
	if defaultYes {
		suffix = " [Y/n] "
	}
	fmt.Print(prompt + suffix)
	line, err := stdin.ReadString('\n')
	if err != nil {
		return false // no answer is not consent, whichever way the default points
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true
	case "":
		return defaultYes
	default:
		return false
	}
}

func hasFlag(args []string, names ...string) bool {
	for _, a := range args {
		for _, n := range names {
			if a == n {
				return true
			}
		}
	}
	return false
}
