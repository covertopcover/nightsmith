package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

// mlx_lm.server accepts every known-bad flag happily and fails only when real
// work arrives, so the guard has to be here rather than in a runtime check.
func TestKnownBadFlagsAreNeverEmitted(t *testing.T) {
	c, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range c.Models {
		cfg := DefaultConfig(c, m)
		args := strings.Join(ServerArgs(cfg, m, "/snapshot"), " ")
		for _, banned := range []string{"--kv-bits", "--draft-model", "--quantize"} {
			if strings.Contains(args, banned) {
				t.Errorf("%s: %s is a documented way to break the install", shortRepo(m.Repo), banned)
			}
		}
	}
}

// The two flags that hold peak memory at 9.9 GB instead of 12.3.
func TestCacheFlagsCarryTheMeasuredConfiguration(t *testing.T) {
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	args := strings.Join(ServerArgs(DefaultConfig(c, m), m, "/snapshot"), " ")
	for _, want := range []string{"--prompt-cache-bytes 512000000", "--prompt-cache-size 1"} {
		if !strings.Contains(args, want) {
			t.Errorf("missing %q — uncapped, the cache grows until it fills RAM\ngot: %s", want, args)
		}
	}
	// Local only. This is not a network service.
	if !strings.Contains(args, "--host 127.0.0.1") {
		t.Errorf("the server must bind to localhost only, got: %s", args)
	}
}

// The server is run through serve.py, which needs the repo id to advertise
// and accept it; the flags after it must be exactly ServerArgs.
func TestServerCommandRunsTheWrapper(t *testing.T) {
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	cfg := DefaultConfig(c, m)
	cmd := ServerCommand(cfg, m, "/snapshot")
	if cmd[0] != serveScriptPath() || cmd[1] != "--served-model-id" || cmd[2] != m.Repo {
		t.Fatalf("wrapper not first: %v", cmd[:3])
	}
	if cmd[3] != "--drain-seconds" || cmd[4] != strconv.Itoa(int(DrainWindow(cfg.MaxTokens).Seconds())) {
		t.Errorf("the drain window must be passed, not left to a second constant: %v", cmd[3:5])
	}
	if strings.Join(cmd[5:], " ") != strings.Join(ServerArgs(cfg, m, "/snapshot"), " ") {
		t.Error("the flags after nightsmith's own must be ServerArgs, unchanged")
	}
}

// The embedded wrapper must refuse to run on any mlx-lm but the pinned one:
// unpatched, the server looks healthy and drops bad requests on the floor.
func TestEmbeddedWrapperGuardsThePin(t *testing.T) {
	if !strings.Contains(string(serveScript), `PINNED_MLX_LM = "`+mlxLMVersion+`"`) {
		t.Errorf("serve.py must pin mlx-lm %s, the version runtime.lock installs", mlxLMVersion)
	}
}

// `stop` must still find a server that the previous version started, and must
// never claim a stranger's process.
func TestIsOurServerKnowsBothLaunchForms(t *testing.T) {
	t.Setenv("HOME", "/Users/x")
	py := pythonBin()
	cases := map[string]bool{
		py + " " + serveScriptPath() + " --served-model-id r --model /m": true,
		py + " -m mlx_lm server --model /m":                              true, // v0.1.1
		"/usr/bin/python3 -m mlx_lm server --model /m":                   false,
		py + " -m http.server 8080":                                      false,
	}
	for cmdline, want := range cases {
		if got := isOurServer(cmdline); got != want {
			t.Errorf("isOurServer(%q) = %v, want %v", cmdline, got, want)
		}
	}
}

// Without these a client that omits them gets mlx_lm's defaults (512 tokens),
// whatever the config file says.
func TestAnswerDefaultsReachTheServer(t *testing.T) {
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	cfg := DefaultConfig(c, m)
	cfg.MaxTokens, cfg.Temperature = 1234, 0.7
	args := strings.Join(ServerArgs(cfg, m, "/snapshot"), " ")
	for _, want := range []string{"--max-tokens 1234", "--temp 0.7"} {
		if !strings.Contains(args, want) {
			t.Errorf("missing %q in %s", want, args)
		}
	}
}

func TestModelBinIsAHardLinkToTheRuntimePython(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	// The venv layout: bin/python3 is a symlink to uv's Python elsewhere.
	real := filepath.Join(home, "real-python")
	mustWrite(t, real, 10)
	os.MkdirAll(filepath.Dir(pythonBin()), 0o755)
	if err := os.Symlink(real, pythonBin()); err != nil {
		t.Fatal(err)
	}

	if got := ensureModelBin(); got != modelBin() {
		t.Fatalf("got %s, want the link", got)
	}
	a, _ := os.Lstat(modelBin())
	b, _ := os.Stat(real)
	if !os.SameFile(a, b) {
		t.Error("nightsmith-model must be a hard link, not a copy or a symlink")
	}

	// A new runtime Python: the old link is replaced, not trusted.
	os.Remove(real)
	mustWrite(t, real, 20)
	ensureModelBin()
	a, _ = os.Lstat(modelBin())
	b, _ = os.Stat(real)
	if !os.SameFile(a, b) {
		t.Error("a stale link must be relinked")
	}

	// No Python at all: fall back rather than fail the start.
	os.Remove(real)
	if got := ensureModelBin(); got != pythonBin() {
		t.Errorf("got %s, want the pythonBin fallback", got)
	}
}

// The drain window used to be a flat 30 s while max_tokens allowed 125 s of
// writing, so `stop` threw away any answer longer than about 360 tokens.
func TestDrainWindowCoversAWholeAnswer(t *testing.T) {
	c, _ := LoadCatalog()
	full := float64(c.Defaults.MaxTokens) / measuredDecodeTokS
	got := DrainWindow(c.Defaults.MaxTokens)
	if got.Seconds() < full {
		t.Errorf("drain is %s, but a %d-token answer takes %.0fs to write",
			got, c.Defaults.MaxTokens, full)
	}
	if got <= 30*time.Second {
		t.Errorf("drain is %s — the measured default answer does not fit in it", got)
	}
	// A short answer still gets a floor, and a huge max_tokens does not turn
	// a stop into a hang.
	if DrainWindow(1) != minDrainWindow {
		t.Errorf("a tiny max_tokens must still get the floor, got %s", DrainWindow(1))
	}
	if DrainWindow(0) != minDrainWindow {
		t.Errorf("an unset max_tokens must get the floor, got %s", DrainWindow(0))
	}
	if DrainWindow(1_000_000) != maxDrainWindow {
		t.Errorf("a stop must still feel like a stop, got %s", DrainWindow(1_000_000))
	}
}

// serve.py holds the measurements; these constants only repeat them so the
// tool can explain itself. If one moves without the other, the tool starts
// quoting a limit the server does not have.
func TestServerLimitsMatchTheWrapper(t *testing.T) {
	for _, c := range []struct {
		name string
		want int
	}{
		{"MAX_PROMPT_TOKENS", maxPromptTokens},
		{"MAX_CONCURRENT", maxConcurrentRequests},
	} {
		re := regexp.MustCompile(c.name + ` = ([0-9_]+)`)
		m := re.FindStringSubmatch(string(serveScript))
		if m == nil {
			t.Fatalf("serve.py no longer defines %s", c.name)
		}
		got, err := strconv.Atoi(strings.ReplaceAll(m[1], "_", ""))
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Errorf("serve.py has %s = %d, this file says %d", c.name, got, c.want)
		}
	}
	// status reads this marker out of the log to explain a server that exited
	// on its own. Two spellings would mean it never finds one.
	if !strings.Contains(string(serveScript), `FATAL_MARKER = "`+fatalMarker+`"`) {
		t.Errorf("serve.py must write %q, or status cannot explain a crash", fatalMarker)
	}
}

// A server that ran out of memory says so and exits. The next start must not
// delete the only evidence of why.
func TestStartKeepsTheLastLog(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	os.MkdirAll(stateDir(), 0o755)
	if err := os.WriteFile(serverLog(), []byte("the last words\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rotateLog()
	if b, err := os.ReadFile(previousLog()); err != nil || string(b) != "the last words\n" {
		t.Fatalf("the previous log was not kept: %v %q", err, b)
	}
	// Nothing to keep: no stray empty file left behind.
	os.Remove(previousLog())
	os.WriteFile(serverLog(), nil, 0o644)
	rotateLog()
	if _, err := os.Stat(previousLog()); err == nil {
		t.Error("an empty log must not be rotated")
	}
}

func TestStoppedOnItsOwnExplainsACrashAndNothingElse(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	os.MkdirAll(stateDir(), 0o755)

	// Stopped normally: the PID file is gone, and nothing is claimed.
	os.WriteFile(serverLog(), []byte(fatalMarker+" the generation thread died\n"), 0o644)
	if got := StoppedOnItsOwn(); got != "" {
		t.Errorf("a clean stop must explain nothing, got %q", got)
	}

	// Died: the PID file is still there and the log says why.
	writePID(4242)
	if got := StoppedOnItsOwn(); got != "the generation thread died" {
		t.Errorf("got %q, want the reason from the log", got)
	}

	// After a restart the story is stale: the rotated log belongs to an
	// earlier run, and telling it again would blame this server for a crash
	// that was not its.
	rotateLog()
	os.WriteFile(serverLog(), []byte("ordinary startup chatter\n"), 0o644)
	if got := StoppedOnItsOwn(); got != "" {
		t.Errorf("got %q from a previous run's log, want silence", got)
	}

	// A server that simply was not running says nothing at all.
	os.Remove(serverLog())
	os.Remove(previousLog())
	if got := StoppedOnItsOwn(); got != "" {
		t.Errorf("got %q, want silence", got)
	}
}
