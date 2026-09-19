//go:build windows

package main

import "syscall"

// See procattr_unix.go. Present so the tree keeps compiling for Windows, which
// is not the same as Windows being supported: there is no MLX there, and
// nothing in this tool would have a runtime to talk to.
func detachedProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{CreationFlags: 0x00000200} // CREATE_NEW_PROCESS_GROUP
}
