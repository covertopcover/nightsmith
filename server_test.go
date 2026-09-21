package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	if strings.Join(cmd[3:], " ") != strings.Join(ServerArgs(cfg, m, "/snapshot"), " ") {
		t.Error("the flags after the wrapper must be ServerArgs, unchanged")
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
