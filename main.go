package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"
)

// The command surface:
//
//	nightsmith                 set up, or report that setup is done
//	nightsmith start|stop      the server
//	nightsmith status          proved by a real completion, never a ping
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
		fmt.Fprintf(os.Stderr, "\n  ✗ %s\n\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return cmdSetup()
	}
	switch args[0] {
	case "start":
		return cmdStart()
	case "stop":
		return cmdStop()
	case "status":
		return cmdStatus()
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
		fmt.Printf("nightsmith %s\n", version)
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

    nightsmith                set it up, or tell you it already is
    nightsmith start          turn it on
    nightsmith stop           turn it off
    nightsmith status         is it running, and how much memory
    nightsmith remove         take it back off this Mac

    nightsmith config check   what your settings will cost, against
                              what this Mac can give them
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
		// Re-running a command to check it worked is what non-developers do.
		// It must be instant, idempotent, and must never re-download.
		return cmdStatus()
	}

	fmt.Printf("\n  Nightsmith — AI that runs on your own Mac.\n\n  Looking at this Mac…\n\n")
	fmt.Printf("  ✓  %s", machine.Chip)
	if machine.GPUCores > 0 {
		fmt.Printf(" · %d GPU cores", machine.GPUCores)
	}
	fmt.Printf(" · ~%.0f GB/s memory\n", machine.BandwidthGB)
	fmt.Printf("  ✓  macOS %s\n", machine.MacOS)
	fmt.Printf("  ✓  %d GB memory — the GPU can use %.1f GB of it", machine.RAMGB, machine.CeilingGB)
	if !machine.CeilingIsMeasured {
		fmt.Printf("   (estimated)")
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
	fmt.Printf("  ✓  %.0f GB free on disk — setup needs %s\n", machine.DiskFreeGB, humanBytes(need))
	est := EstimatePeak(m, cat.Defaults.PromptCacheMB, machine.RAMGB, machine.CeilingGB)
	if !est.Fits {
		return errors.New(RefuseReason(m, est, machine.RAMGB))
	}

	if machine.RAMGB <= 8 {
		// Say the quiet part. It is supported precisely because the workload is
		// many small tasks; that would not hold for agent work.
		fmt.Printf("\n  ⚠  16 GB is where this gets comfortable. You have %d.\n\n", machine.RAMGB)
		fmt.Printf("     I'll set up a smaller model. For short, simple tasks it's\n")
		fmt.Printf("     fine. It will compete with your browser for memory.\n")
	}

	fmt.Printf("\n  Here's what fits, and what it'll be like:\n\n")
	if _, _, here := LocateModel(m, DefaultConfig(cat, m)); here {
		fmt.Printf("     %s  ·  %s, already on this Mac\n\n",
			shortRepo(m.Repo), humanBytes(m.TotalBytes()))
	} else {
		fmt.Printf("     %s  ·  %s download\n\n",
			shortRepo(m.Repo), humanBytes(m.TotalBytes()))
	}
	fmt.Printf("     Uses %.1f GB while working, leaving %.1f GB spare.\n",
		est.PeakGB, est.HeadroomGB)

	speed := 12.6 * SpeedFactor(machine.BandwidthGB)
	if m.IsMeasured() && machine.BandwidthGB == 120 {
		fmt.Printf("     Writes %s — steady, not fast.\n", humanSpeed(m.Measured.DecodeTokS))
	} else {
		// Everything here is extrapolated from one measured machine, and says so.
		fmt.Printf("     Should write %s — estimated from this Mac's\n", humanSpeed(speed))
		fmt.Printf("     memory bandwidth, then measured for real below.\n")
	}

	fmt.Printf("\n  Nothing here needs your password, and nothing touches the\n")
	fmt.Printf("  Python or Homebrew already on this Mac.\n\n")

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
	fmt.Printf("  ✓  Started the model                %s\n", humanDuration(time.Since(started).Seconds()))

	// The most important line in the product. A checkmark may only appear
	// after a real completion came back — /v1/models returned 200 while the
	// server could serve nothing, in three different ways.
	mem := WatchMemory(pid)
	res, err := Probe(cfg.Port, cfg, m, "Say hello in one short sentence.")
	peak := mem.Stop()
	if err != nil {
		return fmt.Errorf("the model started but couldn't answer: %w\n\n%s", err, indentTail(readLog(), 8))
	}
	fmt.Printf("  ✓  Asked it something — it answered:\n        %q\n", res.Answer)
	if res.ThinkingKnown && res.ThinkingOff {
		fmt.Printf("  ✓  Thinking stayed off\n")
	}
	// Speed is measured on a second, longer answer. The first one's time is
	// mostly loading the model, and "Hello!" is two tokens: dividing one by
	// the other printed "5 words a second" for a model that writes 9.
	speed, err := Probe(cfg.Port, cfg, m, "Describe a quiet harbour at dawn in about eighty words.")
	if err != nil {
		return fmt.Errorf("the model answered once, then couldn't answer again: %w", err)
	}
	peak = mem.Stop()
	fmt.Printf("  ✓  Measured it here:  %s · %s\n", humanSpeed(speed.TokPerSec), peak)

	if err := WriteConfig(cfg); err != nil {
		return err
	}
	ok = true
	printReady(cfg)
	return nil
}

// ── Screen 3: the handoff ───────────────────────────────────────────────────

func printReady(c Config) {
	fmt.Printf(`
  Ready. It's running now.

    nightsmith start     turn it on
    nightsmith stop      turn it off
    nightsmith status    is it running, and how much memory
    nightsmith remove    take it back off this Mac

  While it's on, it's at  http://127.0.0.1:%d

`, c.Port)
}

// ── The rest of the surface ─────────────────────────────────────────────────

func cmdStart() error {
	cfg, cat, m, err := loaded()
	if err != nil {
		return err
	}
	port := cfg.Port
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
	fmt.Printf("\n  Started (pid %d). It's at http://127.0.0.1:%d\n\n", pid, cfg.Port)
	return nil
}

func cmdStop() error {
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

// cmdStatus must use a real completion, not a ping, or it will cheerfully
// report a post-OOM zombie as healthy.
func cmdStatus() error {
	cfg, _, m, err := loaded()
	if err != nil {
		return err
	}
	pid, ok := ServerPID()
	if !ok {
		fmt.Printf("\n  Set up, but not running.\n\n    nightsmith start    turn it on\n\n")
		return nil
	}

	fmt.Printf("\n  Running (pid %d). Asking it something…\n", pid)
	res, err := Probe(cfg.Port, cfg, m, "Reply with the single word: ready")
	if err != nil {
		// This is the case a health check gets wrong. Say it plainly.
		fmt.Printf("\n  ⚠  The server is up but could not answer:\n     %s\n\n", err)
		fmt.Printf("     That usually means it ran out of memory. 'nightsmith stop'\n")
		fmt.Printf("     then 'nightsmith start' will clear it.\n\n")
		return nil
	}
	fmt.Printf("  ✓  It answered: %q\n", res.Answer)
	// No speed here: a one-word reply measures latency, not writing speed.
	now, peak, _ := Footprint(pid)
	fmt.Printf("  ✓  %s · port %d · using %s (peak %s)\n\n",
		shortRepo(cfg.Model), cfg.Port, humanBytes(now), humanBytes(peak))
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

	fmt.Printf("\n  model               %s weights\n", humanBytes(m.WeightsBytes))
	fmt.Printf("  prompt_cache_mb     %.1f GB\n", e.CacheGB)
	fmt.Printf("  overhead            ~%.1f GB\n\n", e.OverheadGB)
	fmt.Printf("  Estimated peak      %.1f GB\n", e.PeakGB)
	fmt.Printf("  This Mac's ceiling  %.1f GB", e.CeilingGB)
	if e.Fits {
		fmt.Printf("          ✓ %.1f GB spare\n", e.HeadroomGB)
	} else {
		fmt.Printf("          ✗ %.1f GB short\n", -e.HeadroomGB)
	}
	if !machine.CeilingIsMeasured {
		fmt.Printf("\n  The ceiling is the 75%% rule, not this Mac's reported value,\n")
		fmt.Printf("  and it runs slightly optimistic. Treat a margin under\n")
		fmt.Printf("  0.5 GB as no margin.\n")
	}
	if !e.Fits {
		fmt.Printf("\n  ✗ %s\n", RefuseReason(m, e, machine.RAMGB))
	}
	if w := ThinkingConflict(cfg.Thinking, cfg.MaxTokens); w != "" {
		fmt.Printf("\n  ⚠  %s\n", w)
	}
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
		fmt.Printf("%s  %-40s %9s   %s\n", marker, m.Repo, humanBytes(m.TotalBytes()), note)
	}
	fmt.Printf("\n  Sizes are shown because the names give no warning: every row\n")
	fmt.Printf("  above says -4bit or -8bit, and they span %s to %s.\n\n",
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
		fmt.Printf("\n  ⚠  %s isn't in the checked list.\n\n", repo)
		fmt.Printf("     Its size, whether thinking can be turned off, and which\n")
		fmt.Printf("     server flags break on its architecture are all unknown.\n")
		fmt.Printf("     It may not load at all.\n\n")
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
		fmt.Printf("\n  ⚠  %s\n", why)
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
	fmt.Printf("\n  ✓  %s answered: %q\n\n", shortRepo(repo), res.Answer)
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
	fmt.Printf("\n  ✓ Removed %s. Your Mac is back to how it was.\n\n", humanBytes(freed))
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

func confirm(prompt string, defaultYes bool) bool {
	suffix := " [y/N] "
	if defaultYes {
		suffix = " [Y/n] "
	}
	fmt.Print(prompt + suffix)
	line, err := bufio.NewReader(os.Stdin).ReadString('\n')
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
