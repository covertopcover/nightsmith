package main

import "fmt"

// The memory arithmetic. "The tool does the arithmetic,
// not the user." This is what replaces a light/balanced/full dial — the user
// sees the actual constraint and can reason about it.

const (
	// The GPU working-set limit is a share of unified memory, not all of it.
	// This is the number that decides whether a model runs at all, and it is
	// the one every "run LLMs locally!" guide omits.
	//
	// Only a fallback: once the runtime is installed, Metal is asked directly
	// (RuntimeCeilingBytes). The fraction is what Metal reported on a base
	// M4 / 16 GB under macOS 26.6 and 27.0: 12,713,115,648 of 17,179,869,184
	// bytes = 0.74. It has not been read on any other RAM size.
	//
	// Units matter here, and were once wrong. "16 GB" of RAM is 16 GiB; model
	// sizes are decimal GB. The benchmark's "11.8 GB ceiling" was that same
	// Metal figure in GiB — 12.71 decimal GB. Every GB in this file is 1e9
	// bytes, and the ceiling is computed from bytes, never from the label.
	gpuCeilingFraction = 0.74

	// Everything resident that is not weights and not prompt cache: the
	// runtime, the KV cache for a single request, framework overhead.
	// Derived from the measured 16 GB case — 10.22 peak − 6.74 weights − 0.51
	// cache = 2.97, all decimal GB — so it is one machine's number, not a law.
	// (It was 2.6 when derived from an earlier 9.9 read off `top`.)
	overheadGB = 3.0

	bytesPerGB = 1e9
)

type Estimate struct {
	WeightsGB  float64
	CacheGB    float64
	OverheadGB float64
	PeakGB     float64
	CeilingGB  float64
	Fits       bool
	HeadroomGB float64
	Measured   bool // true when this exact configuration has been run
}

// GPUCeilingGB is the ceiling to plan against. detectedGB is what the machine
// reported; pass 0 when that could not be read, and the rule of thumb is used
// instead. The second return says which one you got, because a configuration
// approved against an estimated ceiling deserves to be described that way.
func GPUCeilingGB(ramBytes int64, detectedGB float64) (float64, bool) {
	if detectedGB > 0 {
		return detectedGB, true
	}
	return float64(ramBytes) * gpuCeilingFraction / bytesPerGB, false
}

// ramBytes turns the spec-sheet figure back into bytes: "16 GB" is 16 GiB.
func ramBytes(ramGB int) int64 { return int64(ramGB) << 30 }

// EstimatePeak computes what a configuration will cost against what the machine
// can give it. Everything here is arithmetic the user could do and shouldn't
// have to.
func EstimatePeak(m Model, promptCacheMB int, ramGB int, ceilingGB float64) Estimate {
	ceiling, _ := GPUCeilingGB(ramBytes(ramGB), ceilingGB)
	e := Estimate{
		WeightsGB:  float64(m.WeightsBytes) / bytesPerGB,
		CacheGB:    float64(promptCacheMB) / 1000,
		OverheadGB: overheadGB,
		CeilingGB:  ceiling,
	}
	e.PeakGB = e.WeightsGB + e.CacheGB + e.OverheadGB
	e.HeadroomGB = e.CeilingGB - e.PeakGB
	e.Fits = e.HeadroomGB > 0

	// Only claim "measured" when the row was measured AND this is the
	// configuration it was measured in. A row measured at 512 MB of cache says
	// nothing about the same model at 1500.
	if m.IsMeasured() && promptCacheMB == 512 {
		e.Measured = true
		e.PeakGB = m.Measured.PeakMemoryGB
		e.HeadroomGB = e.CeilingGB - e.PeakGB
		e.Fits = e.HeadroomGB > 0
	}
	return e
}

// WeightsFit applies the separate, blunter rule used when choosing a model:
// weights alone must stay under 55% of RAM, leaving room for cache and macOS.
// It is deliberately more conservative than EstimatePeak.
func WeightsFit(m Model, ramGB int, fraction float64) bool {
	return float64(m.WeightsBytes)/bytesPerGB <= float64(ramGB)*fraction
}

// RefuseReason explains, in the user's terms, why a configuration cannot run —
// or returns "" when it can. The measured OOM is cited because an abstract
// ceiling invites arguing with it and an observed failure does not.
func RefuseReason(m Model, e Estimate, ramGB int) string {
	if e.Fits {
		return ""
	}
	r := fmt.Sprintf("%s needs ~%.1f GB of weights.\n"+
		"    This Mac's GPU ceiling is %.1f GB.",
		shortRepo(m.Repo), e.WeightsGB, e.CeilingGB)
	if ramGB == 16 {
		r += "\n    Measured: 10.4 GB of weights already triggers\n" +
			"    \"[METAL] Insufficient Memory\" on this machine."
	}
	return r
}

// ThinkingConflict catches the settings pair that bites, because its failure is
// an empty answer rather than an error: reasoning is counted against
// max_tokens, so on a hard task it can consume the whole budget and return
// nothing.
func ThinkingConflict(thinking bool, maxTokens int) string {
	if !thinking || maxTokens >= 4000 {
		return ""
	}
	return fmt.Sprintf(`thinking = true  with  max_tokens = %d

     Reasoning is counted against max_tokens. On a hard task it
     can consume the whole budget and return nothing at all.
     Suggested: max_tokens = 4000.`, maxTokens)
}
