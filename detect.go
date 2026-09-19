package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// Looking at this Mac. Two hardware facts decide the
// answer, and "run LLMs locally!" guides consistently skip both.
//
//	RAM decides whether it runs      — against the GPU wired limit, not total RAM
//	Bandwidth decides how fast       — and it varies 4.5x across chips all called M4
//
// So a MacBook Air M4/16 GB and a Mac Studio M4 Max/16 GB must not be given the
// same answer. Detect the chip tier, not just the memory.

type Machine struct {
	Chip              string  // "Apple M4 Pro"
	Tier              string  // "base" | "Pro" | "Max" | "Ultra"
	BandwidthGB       float64 // memory bandwidth, GB/s
	GPUCores          int
	RAMGB             int // as the spec sheet says it: 16, which is 16 GiB
	RAMBytes          int64
	CeilingGB         float64 // GPU working-set limit, decimal GB like every other size here
	CeilingIsMeasured bool
	MacOS             string
	DiskFreeGB        float64
	AppleSilicon      bool
	UnderRosetta      bool
}

// ChipTier reads the tier and its bandwidth out of a CPU brand string. This is
// the whole of the second axis, and it is pure string work so it can be tested
// without a Mac.
//
// Bandwidth figures are the published ones per tier. They are what turns
// "12.6 tok/s" into an honest per-machine promise instead of a number borrowed
// from someone else's hardware.
func ChipTier(brand string) (tier string, bandwidth float64, ok bool) {
	b := strings.TrimSpace(brand)
	if !strings.HasPrefix(b, "Apple M") {
		return "", 0, false
	}
	switch {
	case strings.HasSuffix(b, "Ultra"):
		return "Ultra", 800, true
	case strings.HasSuffix(b, "Max"):
		return "Max", 450, true
	case strings.HasSuffix(b, "Pro"):
		return "Pro", 273, true
	default:
		return "base", 120, true
	}
}

// SpeedFactor is how much faster this machine should be than the one the
// benchmark ran on, which was a base M4 at ~120 GB/s. Bandwidth, not RAM, sets
// decode speed — the same model is ~4x faster on a Max.
func SpeedFactor(bandwidth float64) float64 {
	if bandwidth <= 0 {
		return 1
	}
	return bandwidth / 120
}

// ParseDiskFreeKB reads `df -Pk <path>` output. POSIX -P guarantees one record
// per filesystem, which is the only reason this is safe to parse.
func ParseDiskFreeKB(out string) (float64, error) {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) < 2 {
		return 0, fmt.Errorf("df produced no rows")
	}
	f := strings.Fields(lines[len(lines)-1])
	if len(f) < 4 {
		return 0, fmt.Errorf("df row has %d fields, expected at least 4", len(f))
	}
	kb, err := strconv.ParseFloat(f[3], 64)
	if err != nil {
		return 0, fmt.Errorf("df free-space column %q is not a number", f[3])
	}
	return kb, nil
}

// ParseGPUCores reads "gpu-core-count" = 10 out of ioreg's listing of the GPU.
func ParseGPUCores(ioreg string) int {
	for _, line := range strings.Split(ioreg, "\n") {
		if !strings.Contains(line, `"gpu-core-count"`) {
			continue
		}
		_, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return n
		}
	}
	return 0
}

func sysctlString(key string) string {
	out, err := exec.Command("sysctl", "-n", key).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func sysctlInt(key string) int64 {
	v := sysctlString(key)
	if v == "" {
		return 0
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return 0
	}
	return n
}

// Detect inspects this Mac. Every field it cannot read is left at its zero
// value rather than guessed, because the screen that prints these is the one
// that has to earn the right to ask a question.
func Detect() (Machine, error) {
	var m Machine

	m.Chip = sysctlString("machdep.cpu.brand_string")
	tier, bw, ok := ChipTier(m.Chip)
	m.Tier, m.BandwidthGB, m.AppleSilicon = tier, bw, ok

	// uname lies under Rosetta, so this is how an Apple Silicon Mac in an
	// x86-64 terminal is told apart from a genuine Intel one. Refusing the
	// first would be a confident wrong refusal, which is worse than an
	// unclear one because the user believes it.
	m.UnderRosetta = sysctlInt("sysctl.proc_translated") == 1

	if mem := sysctlInt("hw.memsize"); mem > 0 {
		m.RAMBytes = mem
		m.RAMGB = int(float64(mem)/(1<<30) + 0.5) // report as the Mac's spec sheet does
	}
	// There is no sysctl for GPU cores (hw.perflevel0.gpucorecount does not
	// exist, checked on an M4 under macOS 26 and 27). The GPU driver's
	// registry entry has it.
	if out, err := exec.Command("ioreg", "-rc", "AGXAccelerator", "-d1").Output(); err == nil {
		m.GPUCores = ParseGPUCores(string(out))
	}

	// The ceiling is asked of Metal itself once the runtime is installed — see
	// RuntimeCeilingBytes. Before that (Screen 1, on a fresh Mac) it is the
	// fraction measured on the one machine that has reported it, flagged as an
	// estimate. A raised iogpu.wired_limit_mb is not consulted: Metal's figure
	// already reflects it, and before the runtime exists it is rare enough not
	// to plan on.
	var detected float64
	if b, ok := RuntimeCeilingBytes(); ok {
		detected = float64(b) / bytesPerGB
	}
	m.CeilingGB, m.CeilingIsMeasured = GPUCeilingGB(m.RAMBytes, detected)

	if out, err := exec.Command("sw_vers", "-productVersion").Output(); err == nil {
		m.MacOS = strings.TrimSpace(string(out))
	}

	if out, err := exec.Command("df", "-Pk", homeDir()).Output(); err == nil {
		if kb, err := ParseDiskFreeKB(string(out)); err == nil {
			m.DiskFreeGB = kb * 1024 / bytesPerGB
		}
	}

	return m, nil
}

// Refusal returns the sentence to print and stop on, or "" to continue.
// Refuse clearly rather than fall back. A CPU fallback
// would be too slow to function as a product while still generating the support
// load of one, and a paid fallback contradicts the free premise.
func (m Machine) Refusal() string {
	if m.UnderRosetta {
		return `This terminal is running under Rosetta, so it reports an Intel CPU.
  This Mac is Apple Silicon and nightsmith will work here.

  Right-click your terminal app in Applications → Get Info → untick
  "Open using Rosetta", reopen it, and run this again.`
	}
	if !m.AppleSilicon {
		return `This Mac has an Intel processor. MLX needs Apple Silicon
  (M1 or newer), so nightsmith can't run here.

  It'll work on any Mac from 2020 onward with an M-series chip.`
	}
	return ""
}
