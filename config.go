package main

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	"github.com/BurntSushi/toml"
)

// The configuration file: real names, real values, real
// consequences — not friendly labels over hidden numbers. A light/balanced/full
// dial sounds kind and is worse, because it hides which number moved.
//
// The file stays flat on purpose. The three-tier taxonomy behind these settings
// (server flags / standard request params / per-model template args) is real,
// and absorbing it is the tool's job, not the user's.

type Config struct {
	Model            string  `toml:"model"`
	PromptCacheMB    int     `toml:"prompt_cache_mb"`
	PromptCacheSlots int     `toml:"prompt_cache_slots"`
	MaxTokens        int     `toml:"max_tokens"`
	Temperature      float64 `toml:"temperature"`
	Thinking         bool    `toml:"thinking"`
	ModelsDir        string  `toml:"models_dir"`
	Port             int     `toml:"port"`
	Offline          bool    `toml:"offline"`
}

func DefaultConfig(c *Catalog, m Model) Config {
	return Config{
		Model:            m.Repo,
		PromptCacheMB:    c.Defaults.PromptCacheMB,
		PromptCacheSlots: c.Defaults.PromptCacheSlots,
		MaxTokens:        c.Defaults.MaxTokens,
		Temperature:      c.Defaults.Temperature,
		Thinking:         false,
		ModelsDir:        modelsDir(),
		Port:             8080,
		Offline:          true,
	}
}

func LoadConfig() (Config, error) {
	c, _, err := readConfigFile(configPath())
	return c, err
}

func readConfigFile(path string) (Config, []string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, nil, err
	}
	c, unknown, err := parseConfig(b)
	if err != nil {
		return c, nil, fmt.Errorf("%s is not readable as TOML: %w", path, err)
	}
	return c, unknown, nil
}

// parseConfig also returns the keys that are not settings. The file is edited
// by hand, and a TOML decoder ignores what it does not know, so a misspelled
// setting would otherwise do nothing and say nothing.
func parseConfig(b []byte) (Config, []string, error) {
	var c Config
	md, err := toml.Decode(string(b), &c)
	if err != nil {
		return c, nil, err
	}
	var unknown []string
	for _, k := range md.Undecoded() {
		unknown = append(unknown, k.String())
	}
	sort.Strings(unknown)
	return c, unknown, nil
}

// runningConfigPath holds the settings the running server was started with,
// written by StartServer. A timestamp comparison would not do: `start`
// rewrites config.toml itself when it has to move port.
func runningConfigPath() string { return filepath.Join(stateDir(), "server.toml") }

// changedSettings names each setting whose value differs, as
// "prompt_cache_mb (512 → 1024)".
func changedSettings(running, now Config) []string {
	var out []string
	a, b := reflect.ValueOf(running), reflect.ValueOf(now)
	for i := 0; i < a.NumField(); i++ {
		x, y := a.Field(i).Interface(), b.Field(i).Interface()
		if x != y {
			name := a.Type().Field(i).Tag.Get("toml")
			out = append(out, fmt.Sprintf("%s (%v → %v)", name, x, y))
		}
	}
	return out
}

// ConfigWarnings is what a user editing the file needs to be told: keys that
// are not settings, and settings the running server has not picked up.
func ConfigWarnings(running bool) []string {
	var w []string
	now, unknown, err := readConfigFile(configPath())
	if err != nil {
		return nil // the caller has already loaded it; this is advice only
	}
	for _, k := range unknown {
		w = append(w, fmt.Sprintf("%s is not a setting — it is being ignored.", k))
	}
	if running {
		if was, _, err := readConfigFile(runningConfigPath()); err == nil {
			if ch := changedSettings(was, now); len(ch) > 0 {
				w = append(w, "Changed since the server started: "+strings.Join(ch, ", ")+
					".\n     'nightsmith stop' then 'nightsmith start' to apply.")
			}
		}
	}
	return w
}

func ConfigExists() bool {
	_, err := os.Stat(configPath())
	return err == nil
}

// WriteConfig writes the file with its explanations intact. The comments are
// not decoration — the rule is that every setting states what it is, what the
// default is, and what measurably happens if you change it. A config file
// written without them is a config file the user cannot act on.
func WriteConfig(c Config) error {
	if err := os.MkdirAll(filepath.Dir(configPath()), 0o755); err != nil {
		return err
	}
	return os.WriteFile(configPath(), []byte(renderConfig(c)), 0o644)
}

func renderConfig(c Config) string {
	return fmt.Sprintf(`# ~/.nightsmith/config.toml
# These are passed straight to the model server. Defaults are
# measured on a 16 GB M4; "measured" notes say what was observed.

# ── Model ────────────────────────────────────────────────────
# The biggest lever by far. Most of the total footprint is this
# one file. Everything else here moves hundreds of megabytes;
# this moves gigabytes.
model = %q

# ── Prompt cache ─────────────────────────────────────────────
# Keeps prompts the model has already read, so a repeated
# opening doesn't get re-read. A hit is ~14x faster to first
# token: 1,900 vs 131 tokens/sec.
#
# It is capped on purpose. Uncapped, it grows until it fills
# RAM — that is what made the old Ollama setup swap 8.3 GB.
prompt_cache_mb = %d
  # measured:  512 -> 9.9 GB peak
  #           1500 -> 11.3 GB peak, identical speed and output
  #        no cap -> 12.3 GB peak

prompt_cache_slots = %d
  # How many different prompts to remember.
  # 1 fits tasks that are alike and run one after another:
  # each task reuses the previous task's opening.
  # Raise it if you interleave different kinds of task.

# ── Answers ──────────────────────────────────────────────────
max_tokens = %d
  # Hard stop for a single answer. Real answers run under 600.
  # The cap exists because Gemma's thinking mode once looped
  # 3,000 tokens without answering — which stalls a queue.

temperature = %v
  # 0 = the same question gives the same answer every time
  # (verified: 60/60 identical). Raise toward 1 for variety,
  # and lose reproducibility.

thinking = %v
  # Let the model reason at length before answering. Off by
  # default, and on this workload that default is doing real
  # work: thinking multiplies output tokens roughly 3-10x, it
  # is spent out of max_tokens above, and reasoning text in a
  # reply that was supposed to be JSON will not parse.
  #
  # How this is sent differs per model, and nightsmith handles
  # that for you — see 'nightsmith model list'. Some models
  # cannot turn it off at all.

# ── Location ─────────────────────────────────────────────────
models_dir = %q
port = %d

offline = %v
  # After the download, never call Hugging Face again. Stops a
  # background task stalling on the network, and stops weights
  # changing under you between runs.
`, c.Model, c.PromptCacheMB, c.PromptCacheSlots, c.MaxTokens,
		c.Temperature, c.Thinking, c.ModelsDir, c.Port, c.Offline)
}
