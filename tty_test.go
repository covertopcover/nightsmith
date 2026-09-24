package main

import (
	"os"
	"testing"
)

// The distinction the bare command's meaning rests on.
//
// /dev/null is a character device, so the cheap check that is right for
// colour is wrong for this: a cron job or a launchd service running
// `nightsmith </dev/null` would look like a terminal, open a conversation,
// read EOF and exit silently — when what it asked for was status and an exit
// code. isConsole asks the kernel for terminal settings instead.
func TestDevNullIsNotAConsole(t *testing.T) {
	f, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if !isTerminal(f) {
		t.Skip("this platform does not report /dev/null as a character device, so there is nothing to distinguish")
	}
	if isConsole(f) {
		t.Error("/dev/null passed the console check — a cron job would land in a conversation")
	}
}

// A pipe is neither, and must never be mistaken for a terminal.
func TestAPipeIsNotAConsole(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	if isConsole(r) {
		t.Error("a pipe passed the console check")
	}
}
