package main

import (
	"strings"
	"testing"
)

// The failure this exists to prevent, seen on a real install: setup finishes,
// prints "nightsmith start", and the shell answers "command not found",
// because the PATH line the installer added reaches only future shells.
//
// Nobody is told to source anything. Either a fresh login shell follows — in
// which case the short name is correct — or none does, and the full path is
// printed, which needs no PATH at all.
func TestTheCommandsPrintedAlwaysWork(t *testing.T) {
	t.Setenv("HOME", "/Users/x")

	if got := CommandName(""); got != "nightsmith" {
		t.Errorf("PATH already had it: want the short name, got %q", got)
	}
	if got := CommandName("fresh-shell"); got != "nightsmith" {
		t.Errorf("a login shell follows, so the short name works: got %q", got)
	}
	got := CommandName("full-path")
	if !strings.HasPrefix(got, "~/") || !strings.HasSuffix(got, "/nightsmith") {
		t.Errorf("no shell follows: want a tilde path that needs no PATH, got %q", got)
	}
}
