package main

import (
	"fmt"
	"os"
	"strings"
)

// Colour for ✓ ⚠ ✗, the same three colours install.sh prints, so the tool
// reads as a continuation of the installer rather than a different program.
//
// The glyphs stay literal in every format string, so the code still reads
// like the screens; colour is applied on the way out. Off when NO_COLOR is
// set (no-color.org), for TERM=dumb, and whenever the output is not a
// terminal — escape codes in a log file or a pipe are noise.

var stdoutColor = colorOn(os.Stdout)

func colorOn(f *os.File) bool {
	if os.Getenv("NO_COLOR") != "" || os.Getenv("TERM") == "dumb" {
		return false
	}
	return isTerminal(f)
}

// isTerminal is a character-device check, which is all that is needed here
// and needs no cgo.
func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

var colorizer = strings.NewReplacer(
	"✓", "\033[32m✓\033[0m",
	"⚠", "\033[33m⚠\033[0m",
	"✗", "\033[31m✗\033[0m",
)

func colorize(s string, on bool) string {
	if !on {
		return s
	}
	return colorizer.Replace(s)
}

// printf is fmt.Printf with the glyphs coloured when stdout is a terminal.
func printf(format string, args ...any) {
	fmt.Print(colorize(fmt.Sprintf(format, args...), stdoutColor))
}

// stepColumn is where the detail column starts on a ✓ line, so that
// "Runtime ready", "Started the model" and the rest line up.
const stepColumn = 33

// stepLine renders "  ✓  <label padded>  <detail>".
func stepLine(label, detail string) string {
	return fmt.Sprintf("  ✓  %-*s%s\n", stepColumn, label, detail)
}

// step prints a line saying what is happening now, and returns the function
// that replaces it with the finished line. On a terminal the "…" line is
// overwritten in place; elsewhere only the finished line is printed, so a log
// holds no half-lines.
func step(doing string) func(label, detail string) {
	if isTerminal(os.Stdout) {
		printf("     %s…", doing)
	}
	return func(label, detail string) {
		if isTerminal(os.Stdout) {
			printf("\r%s\r", strings.Repeat(" ", len(doing)+8))
		}
		printf("%s", stepLine(label, detail))
	}
}
