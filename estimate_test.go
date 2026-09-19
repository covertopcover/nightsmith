package main

import (
	"math"
	"testing"
)

func closeTo(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// The measured 16 GB case is the anchor for every number in this file. If this
// test fails, the arithmetic has drifted from what was measured.
func TestEstimateMatchesTheMeasuredMachine(t *testing.T) {
	c, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := c.Find("mlx-community/gemma-4-12B-it-4bit")
	if !ok {
		t.Fatal("the one measured model is missing from the table")
	}
	// 12.71 GB (decimal) is what Metal reported on a base M4 / 16 GB —
	// 12,713,115,648 bytes. The benchmark's "11.8" was the same figure in GiB.
	e := EstimatePeak(m, 512, 16, 12.713)

	if !closeTo(e.CeilingGB, 12.713, 0.01) {
		t.Errorf("GPU ceiling = %.2f, want the machine's reported 12.71", e.CeilingGB)
	}
	if !closeTo(e.PeakGB, 10.2, 0.05) {
		t.Errorf("peak = %.2f, want 10.2 (measured)", e.PeakGB)
	}
	if !closeTo(e.HeadroomGB, 2.5, 0.1) {
		t.Errorf("headroom = %.2f, want ~2.5", e.HeadroomGB)
	}
	if !e.Fits || !e.Measured {
		t.Errorf("the measured configuration should fit and report as measured")
	}
}

// Arithmetic and measurement must agree on the case where both exist,
// otherwise the estimate is not trustworthy on the rungs that have no
// measurement at all.
func TestArithmeticAgreesWithMeasurementOnTheKnownCase(t *testing.T) {
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	weightsGB := float64(m.WeightsBytes) / bytesPerGB // 6.74 GB
	sum := weightsGB + 0.512 + overheadGB
	if !closeTo(sum, m.Measured.PeakMemoryGB, 0.1) {
		t.Errorf("weights+cache+overhead = %.2f, measured peak is %.2f", sum, m.Measured.PeakMemoryGB)
	}
}

func TestRefusesWhatCannotRun(t *testing.T) {
	c, _ := LoadCatalog()
	big, _ := c.Find("mlx-community/Qwen3.8-27B-4bit")
	e := EstimatePeak(big, 512, 16, 12.713)
	if e.Fits {
		t.Fatal("16.1 GB of weights must not fit on a 16 GB Mac")
	}
	reason := RefuseReason(big, e, 16)
	if reason == "" {
		t.Fatal("a refusal must explain itself")
	}
	// The measured OOM is what makes the refusal credible rather than arguable.
	if !contains(reason, "Insufficient Memory") {
		t.Errorf("a 16 GB refusal should cite the observed Metal OOM, got:\n%s", reason)
	}
}

func TestPicksErrLowNotHigh(t *testing.T) {
	c, _ := LoadCatalog()
	for _, tc := range []struct {
		ram  int
		want string
	}{
		{8, "mlx-community/MiniCPM5-2B-8bit"},
		{16, "mlx-community/gemma-4-12B-it-4bit"},
		{24, "mlx-community/Qwen3-14B-4bit"},
		{32, "mlx-community/Qwen3.8-27B-4bit"},
		{64, "mlx-community/Qwen3.6-35B-A3B-4bit"},
	} {
		got, ok := c.PickFor(tc.ram)
		if !ok || got.Repo != tc.want {
			t.Errorf("%d GB picked %q, want %q", tc.ram, got.Repo, tc.want)
		}
		// Whatever is picked must clear the 55% weights rule it was picked by.
		if !WeightsFit(got, tc.ram, 0.55) {
			t.Errorf("%d GB picked %s, which breaks the 55%% weights rule", tc.ram, got.Repo)
		}
	}
}

// The 111 GB model must never be selected for any machine the tool supports.
// It exists in the table to be shown crossed out.
func TestTheHugeModelIsNeverADefault(t *testing.T) {
	c, _ := LoadCatalog()
	for _, ram := range []int{8, 16, 24, 32, 48, 64, 128, 256, 512} {
		if got, ok := c.PickFor(ram); ok && got.Repo == "mlx-community/Qwen3.8-Flash-Next-4bit" {
			t.Errorf("%d GB picked the 111 GB model", ram)
		}
	}
}

func TestThinkingAndMaxTokensConflict(t *testing.T) {
	if ThinkingConflict(true, 1500) == "" {
		t.Error("thinking=true with max_tokens=1500 must warn — it returns nothing at all")
	}
	if ThinkingConflict(false, 1500) != "" {
		t.Error("thinking=false is the default and must not warn")
	}
	if ThinkingConflict(true, 4000) != "" {
		t.Error("the suggested max_tokens must not itself warn")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// The fallback must be computed from bytes, in decimal GB like every model
// size. It was once 16 × 0.75 = "12.0 GB" — GiB arithmetic printed next to
// decimal model sizes. Metal's real figure on a 16 GB M4 is 12,713,115,648
// bytes: 12.71 GB, or the 11.84 "GB" (GiB) the benchmark quoted.
func TestCeilingFallbackIsInDecimalGBFromBytes(t *testing.T) {
	fallback, measured := GPUCeilingGB(16<<30, 0)
	if measured {
		t.Fatal("no detected value was passed, so this cannot claim to be measured")
	}
	if !closeTo(fallback, 12.713, 0.01) {
		t.Fatalf("fallback for 16 GiB = %.3f GB, want Metal's measured 12.713", fallback)
	}
	detected, measured := GPUCeilingGB(16<<30, 12.5)
	if !measured || !closeTo(detected, 12.5, 0.001) {
		t.Errorf("a detected ceiling must be used as given, got %.2f", detected)
	}
}
