package main

import (
	_ "embed"
	"fmt"
	"sort"

	"github.com/BurntSushi/toml"
)

// models.toml is the per-model capability table, embedded so the binary carries
// it. A per-model capability table is the product: Nightsmith
// absorbs this so the user never learns it."
//
//go:embed models.toml
var modelsTOML []byte

type Catalog struct {
	Meta struct {
		Schema          int    `toml:"schema"`
		SizesVerifiedOn string `toml:"sizes_verified_on"`
	} `toml:"meta"`
	Defaults struct {
		PromptCacheMB      int      `toml:"prompt_cache_mb"`
		PromptCacheSlots   int      `toml:"prompt_cache_slots"`
		MaxTokens          int      `toml:"max_tokens"`
		Temperature        float64  `toml:"temperature"`
		WeightsRAMFraction float64  `toml:"weights_ram_fraction"`
		NeverConfigurable  []string `toml:"never_configurable"`
	} `toml:"defaults"`
	Models      []Model      `toml:"model"`
	FamilyNotes []FamilyNote `toml:"family_note"`
}

type Model struct {
	Repo            string  `toml:"repo"`
	Revision        string  `toml:"revision"` // Hub commit; "" only for unchecked models
	Architecture    string  `toml:"architecture"`
	ParamsM         float64 `toml:"params_m"`
	WeightsBytes    int64   `toml:"weights_bytes"`
	TokenizerBytes  int64   `toml:"tokenizer_bytes"`
	MinRAMGB        int     `toml:"min_ram_gb"`
	DefaultForRAMGB int     `toml:"default_for_ram_gb"`
	Status          string  `toml:"status"`
	Note            string  `toml:"note"`

	Measured     *Measured         `toml:"measured"`
	Thinking     Thinking          `toml:"thinking"`
	BrokenFlags  map[string]string `toml:"broken_flags"`
	Capabilities map[string]string `toml:"capabilities"`
}

// Measured holds numbers observed on real hardware. Its presence is what makes
// a claim quotable; nothing here may be filled in from arithmetic.
type Measured struct {
	Machine         string  `toml:"machine"`
	Source          string  `toml:"source"`
	PeakMemoryGB    float64 `toml:"peak_memory_gb"`
	IdleMemoryGB    float64 `toml:"idle_memory_gb"`
	DecodeTokS      float64 `toml:"decode_tok_s"`
	PrefillTokSCold float64 `toml:"prefill_tok_s_cold"`
	PrefillTokSWarm float64 `toml:"prefill_tok_s_warm"`
	SoakMinutes     int     `toml:"soak_minutes"`
	SoakRequests    int     `toml:"soak_requests"`
	Swap            bool    `toml:"swap"`
}

// Thinking records how one model's chat template spells "reason before
// answering". There is no standard: the key name, whether it can be turned off
// at all, and what omitting it means all vary per model.
type Thinking struct {
	Key            string `toml:"key"`
	Transport      string `toml:"transport"`
	CanDisable     bool   `toml:"can_disable"`
	OmittingMeans  string `toml:"omitting_means"`
	SendExplicitly bool   `toml:"send_explicitly"`
	Status         string `toml:"status"`
	Note           string `toml:"note"`
}

type FamilyNote struct {
	Family      string `toml:"family"`
	Thinking    string `toml:"thinking"`
	Consequence string `toml:"consequence"`
}

func LoadCatalog() (*Catalog, error) {
	var c Catalog
	if err := toml.Unmarshal(modelsTOML, &c); err != nil {
		return nil, fmt.Errorf("model table is unreadable: %w", err)
	}
	if len(c.Models) == 0 {
		return nil, fmt.Errorf("model table is empty")
	}
	return &c, nil
}

// TotalBytes is what actually gets downloaded: weights plus the tokenizer,
// which is 32 MB for Gemma 4 and not nothing.
func (m Model) TotalBytes() int64 { return m.WeightsBytes + m.TokenizerBytes }

// Measured reports whether this row has numbers behind it or is arithmetic.
// Untested rungs are labelled as such, deliberately.
func (m Model) IsMeasured() bool { return m.Status == "measured" && m.Measured != nil }

func (c *Catalog) Find(repo string) (Model, bool) {
	for _, m := range c.Models {
		if m.Repo == repo {
			return m, true
		}
	}
	return Model{}, false
}

// PickFor chooses the model for a machine with this much RAM: the largest
// default whose rung this machine meets.
//
// The rule errs low on purpose. An untested rung failing small means a weaker
// model than the machine could handle; failing big means a Metal OOM the user
// cannot diagnose.
func (c *Catalog) PickFor(ramGB int) (Model, bool) {
	best := Model{}
	found := false
	for _, m := range c.Models {
		if m.DefaultForRAMGB == 0 || m.DefaultForRAMGB > ramGB {
			continue
		}
		if !found || m.DefaultForRAMGB > best.DefaultForRAMGB {
			best, found = m, true
		}
	}
	return best, found
}

// Ordered returns the table smallest first, which is the order `model list`
// prints: what fits, then what doesn't, always with sizes. The sizes are the
// point — mlx-community/Qwen3.8-Flash-Next-4bit is 111 GB and its name says
// -4bit exactly like the 2.7 GB one does.
func (c *Catalog) Ordered() []Model {
	out := append([]Model(nil), c.Models...)
	sort.Slice(out, func(i, j int) bool { return out[i].TotalBytes() < out[j].TotalBytes() })
	return out
}
