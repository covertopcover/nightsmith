package main

import (
	"strings"
	"testing"
)

// A config file without its explanations is one the user cannot act on:
// "Being technical is fine. Being unexplained is not."
func TestWrittenConfigExplainsItself(t *testing.T) {
	c, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	out := renderConfig(DefaultConfig(c, m))

	for _, want := range []string{
		"measured:  512 -> 9.9 GB peak",  // the number that justifies the cap
		"1,900 vs 131",                   // what the cache buys
		"60/60 identical",                // why temperature 0
		"3,000 tokens without answering", // why thinking is off
		"never call Hugging Face again",  // what offline means
	} {
		if !strings.Contains(out, want) {
			t.Errorf("config is missing the reason %q — it would be numbers without consequences", want)
		}
	}
}

// Round-tripping matters because the file is the user's to edit; a setting they
// change must survive being read back.
func TestConfigRoundTrips(t *testing.T) {
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	orig := DefaultConfig(c, m)
	orig.PromptCacheMB = 1500
	orig.Thinking = true

	var got Config
	if _, err := tomlDecode(renderConfig(orig), &got); err != nil {
		t.Fatal(err)
	}
	if got.PromptCacheMB != 1500 || !got.Thinking || got.Model != orig.Model {
		t.Errorf("round trip lost settings: %+v", got)
	}
}

// The defaults are the measured configuration. If these drift, every number the
// tool quotes stops being about the configuration it actually runs.
func TestDefaultsAreTheMeasuredConfiguration(t *testing.T) {
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	d := DefaultConfig(c, m)
	if d.PromptCacheMB != 512 || d.PromptCacheSlots != 1 {
		t.Errorf("cache defaults %d/%d are not the benchmarked 512/1", d.PromptCacheMB, d.PromptCacheSlots)
	}
	if d.Temperature != 0 {
		t.Errorf("temperature %v breaks the reproducibility background work depends on", d.Temperature)
	}
	if d.Thinking {
		t.Error("thinking must default off — it is spent out of max_tokens")
	}
	if !d.Offline {
		t.Error("offline must default on, or a background task can stall on the network")
	}
}

// A misspelled setting is ignored by any TOML decoder. It must at least be
// named, or editing the file silently does nothing.
func TestUnknownKeysAreReported(t *testing.T) {
	c, unknown, err := parseConfig([]byte("prompt_cache_mb = 1024\nprompt_cahce_mb = 9\nport = 8081\n"))
	if err != nil {
		t.Fatal(err)
	}
	if c.PromptCacheMB != 1024 || c.Port != 8081 {
		t.Errorf("known keys must still decode: %+v", c)
	}
	if len(unknown) != 1 || unknown[0] != "prompt_cahce_mb" {
		t.Errorf("unknown = %v, want [prompt_cahce_mb]", unknown)
	}
}

func TestChangedSettingsNamesWhatMoved(t *testing.T) {
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	was := DefaultConfig(c, m)
	if ch := changedSettings(was, was); len(ch) != 0 {
		t.Errorf("identical configs reported as changed: %v", ch)
	}
	now := was
	now.PromptCacheMB = 1024
	now.Thinking = true
	ch := changedSettings(was, now)
	if len(ch) != 2 || ch[0] != "prompt_cache_mb (512 → 1024)" || ch[1] != "thinking (false → true)" {
		t.Errorf("got %v", ch)
	}
}

// The snapshot is written with renderConfig and read back with parseConfig;
// if the two disagreed, a running server would always look out of date.
func TestRenderedConfigRoundTrips(t *testing.T) {
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	want := DefaultConfig(c, m)
	got, unknown, err := parseConfig([]byte(renderConfig(want)))
	if err != nil || len(unknown) != 0 {
		t.Fatalf("err=%v unknown=%v", err, unknown)
	}
	if ch := changedSettings(want, got); len(ch) != 0 {
		t.Errorf("round trip changed %v", ch)
	}
}

// The same text is written twice: once as the user's settings, once as the
// record of what the running server was started with. It used to open by
// naming config.toml in both, so server.toml claimed to be another file.
func TestEachConfigFileNamesItself(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	cfg := DefaultConfig(c, m)

	if first := headerOf(renderConfig(cfg)); !strings.Contains(first, "config.toml") {
		t.Errorf("config.toml opens with %q", first)
	}
	first := headerOf(renderRunningConfig(cfg))
	if !strings.Contains(first, "server.toml") {
		t.Errorf("server.toml opens with %q", first)
	}
	if strings.Contains(first, "config.toml") {
		t.Errorf("server.toml still claims to be config.toml: %q", first)
	}
}

func headerOf(s string) string {
	line, _, _ := strings.Cut(s, "\n")
	return line
}
