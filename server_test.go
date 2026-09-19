package main

import (
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
