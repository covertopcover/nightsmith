package main

import "testing"

// These four chips all say "M4" and differ 4.5x in bandwidth. A
// RAM-only heuristic gives them the same model and wildly different lives.
func TestChipTierSeparatesMachinesThatLookAlike(t *testing.T) {
	cases := []struct {
		brand string
		tier  string
		bw    float64
		ok    bool
	}{
		{"Apple M4", "base", 120, true},
		{"Apple M4 Pro", "Pro", 273, true},
		{"Apple M4 Max", "Max", 450, true},
		{"Apple M2 Ultra", "Ultra", 800, true},
		{"Apple M1", "base", 120, true},
		{"  Apple M3 Max  ", "Max", 450, true},
		{"Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz", "", 0, false},
		{"", "", 0, false},
	}
	for _, c := range cases {
		tier, bw, ok := ChipTier(c.brand)
		if tier != c.tier || bw != c.bw || ok != c.ok {
			t.Errorf("ChipTier(%q) = (%q, %v, %v), want (%q, %v, %v)",
				c.brand, tier, bw, ok, c.tier, c.bw, c.ok)
		}
	}
}

func TestSpeedFactorIsRelativeToTheBenchmarkedMachine(t *testing.T) {
	if f := SpeedFactor(120); f != 1 {
		t.Errorf("the benchmarked machine must be the 1.0 baseline, got %v", f)
	}
	// The same model is ~4x faster on a Max.
	if f := SpeedFactor(450); f < 3.5 || f > 4.5 {
		t.Errorf("a Max should be ~4x the base M4, got %.2fx", f)
	}
	if f := SpeedFactor(0); f != 1 {
		t.Errorf("unknown bandwidth must not invent a multiplier, got %v", f)
	}
}

func TestParseDiskFree(t *testing.T) {
	out := `Filesystem 1024-blocks      Used Available Capacity Mounted on
/dev/disk3s5   482797652 123456789  35651584      78% /System/Volumes/Data
`
	kb, err := ParseDiskFreeKB(out)
	if err != nil {
		t.Fatal(err)
	}
	if gb := kb * 1024 / bytesPerGB; gb < 36 || gb > 37 {
		t.Errorf("free space = %.1f GB, want ~36.5", gb)
	}
	if _, err := ParseDiskFreeKB("Filesystem 1024-blocks\n"); err == nil {
		t.Error("a df listing with no rows must be an error, not zero bytes free")
	}
}

// A wrong refusal is worse than an unclear one, because the user believes it.
func TestRosettaIsNotRefusedAsIntel(t *testing.T) {
	rosetta := Machine{Chip: "Intel(R) Core(TM) i7", AppleSilicon: false, UnderRosetta: true}
	if r := rosetta.Refusal(); !contains(r, "Rosetta") {
		t.Errorf("an Apple Silicon Mac under Rosetta must be told how to fix it, got:\n%s", r)
	}
	intel := Machine{Chip: "Intel(R) Core(TM) i7", AppleSilicon: false}
	if r := intel.Refusal(); !contains(r, "Apple Silicon") || contains(r, "Rosetta") {
		t.Errorf("a genuine Intel Mac gets the clean refusal, got:\n%s", r)
	}
	good := Machine{Chip: "Apple M4", AppleSilicon: true}
	if r := good.Refusal(); r != "" {
		t.Errorf("a supported Mac must not be refused, got:\n%s", r)
	}
}

// Read off `ioreg -rc AGXAccelerator -d1` on a base M4. There is no sysctl for
// this; the one the code once read does not exist.
func TestParseGPUCores(t *testing.T) {
	out := `+-o AGXAcceleratorG16G  <class AGXAcceleratorG16G, id 0x100000356>
    {
      "model" = "Apple M4"
      "gpu-core-count" = 10
      "AGXParameterBufferMaxSize" = 1006632960
    }`
	if n := ParseGPUCores(out); n != 10 {
		t.Errorf("ParseGPUCores = %d, want 10", n)
	}
	if n := ParseGPUCores(""); n != 0 {
		t.Errorf("no ioreg output must mean unknown (0), got %d", n)
	}
}

func TestParseSwapUsed(t *testing.T) {
	for in, want := range map[string]float64{
		"total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)":         0,
		"total = 2048.00M  used = 1060.25M  free = 987.75M  (encrypted)": 1060.25,
		"total = 3.00G  used = 1.50G  free = 1.50G  (encrypted)":         1536,
		"": 0,
	} {
		if got := ParseSwapUsedMB(in); !closeTo(got, want, 0.01) {
			t.Errorf("ParseSwapUsedMB(%q) = %v, want %v", in, got, want)
		}
	}
}
