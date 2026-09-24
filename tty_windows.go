//go:build windows

package main

import "os"

// Nightsmith is Apple Silicon only; this exists so the package still builds
// under GOOS=windows, the way procattr_windows.go does.
func isConsole(f *os.File) bool { return false }
