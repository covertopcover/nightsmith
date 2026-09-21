package main

import (
	"os"
	"strings"
	"testing"
)

func TestColourIsOnlyAddedWhenAsked(t *testing.T) {
	line := "  ✓  ok · ⚠ careful · ✗ no"
	if colorize(line, false) != line {
		t.Error("colour off must leave the text byte-for-byte alone")
	}
	on := colorize(line, true)
	for _, want := range []string{"\033[32m✓", "\033[33m⚠", "\033[31m✗"} {
		if !strings.Contains(on, want) {
			t.Errorf("missing %q in %q", want, on)
		}
	}
}

// A pipe or a log file must never get escape codes, and NO_COLOR wins even
// on a terminal.
func TestNoColourForPipesOrNO_COLOR(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()
	if colorOn(w) {
		t.Error("a pipe is not a terminal")
	}
	t.Setenv("NO_COLOR", "1")
	if tty, err := os.OpenFile("/dev/tty", os.O_WRONLY, 0); err == nil {
		defer tty.Close()
		if colorOn(tty) {
			t.Error("NO_COLOR must turn colour off on a terminal too")
		}
	}
}

// The runtime lines must line up with "Started the model" and the rest.
func TestStepLinesShareAColumn(t *testing.T) {
	a := stepLine("Fetched uv", "1s")
	b := stepLine("Installed MLX (mlx-lm 0.32.0)", "8s")
	if strings.Index(a, "1s") != strings.Index(b, "8s") {
		t.Errorf("columns differ:\n%q\n%q", a, b)
	}
	if want := "  ✓  Runtime ready                    12s\n"; stepLine("Runtime ready", "12s") != want {
		t.Errorf("drifted from the existing layout: %q", stepLine("Runtime ready", "12s"))
	}
}
