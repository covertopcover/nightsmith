//go:build unix

package main

import "syscall"

// Detach the model server from this terminal, so closing the window does not
// take the server with it. `nightsmith stop` is how it ends.
//
// This file exists to keep the one genuinely platform-specific call in the
// codebase isolated. Everything else — the command surface, the model table,
// the arithmetic, the probe — is portable as written. What would not port is
// the product, not the code: MLX is Apple silicon, and the GPU-ceiling
// arithmetic is a unified-memory fact with no meaning against discrete VRAM.
func detachedProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setsid: true}
}
