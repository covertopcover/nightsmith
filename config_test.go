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
