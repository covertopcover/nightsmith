//go:build !windows

package main

import (
	"os"
	"syscall"
	"unsafe"
)

// isConsole is a real isatty, and the difference from isTerminal matters.
//
// isTerminal (ui.go) asks whether a file is a character device, which is all
// colour needs. But /dev/null is a character device too — so a cron job or a
// launchd service running `nightsmith </dev/null` would pass that test, open
// a conversation, read EOF and exit silently, when what it asked for was
// status and an exit code.
//
// Whether the bare command talks or reports is the one decision in this tool
// that must not get that wrong, so it asks the kernel for terminal settings
// instead: that succeeds on a tty and on nothing else.
func isConsole(f *os.File) bool {
	var termios syscall.Termios
	_, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, f.Fd(),
		uintptr(syscall.TIOCGETA), uintptr(unsafe.Pointer(&termios)), 0, 0, 0)
	return errno == 0
}
