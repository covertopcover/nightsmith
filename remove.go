package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// `nightsmith remove` (alias `uninstall`).
//
// A tool that is easy to remove is easier to install, and this one puts ~7 GB
// on a stranger's disk. Four rules the screen enforces:
//
//  1. List before deleting, and total it.
//  2. Default to no — [y/N], the inverse of setup's [Y/n].
//  3. Never delete what is shared. ~/.cache/huggingface may hold models Ollama
//     or LM Studio depend on. Setup treats finding one there as a gift; a gift
//     is not ours to throw away.
//  4. Reclaim in the same units the install used, or neither number is believed.
//
// Every install decision that avoided root is what makes this a promise the tool can
// keep: two directories and one binary, no password, nothing left behind.

type RemovalItem struct {
	Path  string
	Bytes int64
	What  string
}

type Removal struct {
	Items      []RemovalItem
	TotalBytes int64
	Shared     []RemovalItem // listed, explicitly not removed
	PathLine   string        // the shell rc line we deliberately leave alone
}

// PlanRemoval reads what is actually on disk rather than printing a table of
// expected sizes. The total has to be true or the closing line is marketing.
func PlanRemoval() Removal {
	var r Removal
	for _, c := range []struct{ path, what string }{
		{modelsDir(), "the model"},
		{runtimeDir(), "MLX, the model runtime"},
		{pythonDir(), "a private copy of Python"},
		{uvDir(), "uv, which installed that Python"},
		{uvCacheDir(), ""}, // only present if an install was interrupted
		{configPath(), "your settings"},
		{chatsDir(), "your conversations"},
		{pidPath(), ""},
		{filepath.Join(stateDir(), "server.log"), ""},
		{serveScriptPath(), ""},
		{runningConfigPath(), ""},
		{binPath(), "the command itself"},
	} {
		n, err := dirSize(c.path)
		if err != nil {
			continue // not there; nothing to report and nothing to remove
		}
		r.Items = append(r.Items, RemovalItem{Path: c.path, Bytes: n, What: c.what})
		r.TotalBytes += n
	}
	// Wherever the shared cache is — HF_HUB_CACHE and HF_HOME move it, and
	// setup reads it from there — named, sized, and left alone.
	shared := sharedCacheRoot()
	if n, err := dirSize(shared); err == nil {
		r.Shared = append(r.Shared, RemovalItem{Path: shared, Bytes: n})
	}
	return r
}

func dirSize(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	if !info.IsDir() {
		return info.Size(), nil
	}
	var total int64
	err = filepath.WalkDir(path, func(_ string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // an unreadable corner must not abort the total
		}
		if !d.IsDir() {
			if fi, err := d.Info(); err == nil {
				total += fi.Size()
			}
		}
		return nil
	})
	return total, err
}

func (r Removal) Describe() string {
	var b strings.Builder
	if len(r.Items) == 0 {
		return "  Nothing to remove — nightsmith isn't set up on this Mac.\n"
	}
	b.WriteString("\n  This removes:\n\n")
	for _, it := range r.Items {
		if it.What == "" {
			continue // internal bookkeeping, not worth a line in the user's list
		}
		b.WriteString(fmt.Sprintf("    %-26s %8s   %s\n",
			shortPath(it.Path), humanBytes(it.Bytes), it.What))
	}
	for _, s := range r.Shared {
		b.WriteString(fmt.Sprintf("\n  It does not touch %s (%s) — Ollama and\n",
			shortPath(s.Path), humanBytes(s.Bytes)))
		b.WriteString("  LM Studio share it, and it is not ours to delete.\n")
	}
	b.WriteString("\n  The PATH line in your shell profile is left alone; removing\n")
	b.WriteString("  it would edit a file you may have changed since.\n")
	return b.String()
}

// Execute deletes. It stops the server first: removing the weights out from
// under a running process leaves a model server serving nothing.
func (r Removal) Execute() (int64, error) {
	if _, err := StopServer(); err != nil {
		return 0, fmt.Errorf("couldn't stop the model server first: %w", err)
	}
	var freed int64
	for _, it := range r.Items {
		if err := os.RemoveAll(it.Path); err != nil {
			return freed, fmt.Errorf("couldn't remove %s: %w", it.Path, err)
		}
		freed += it.Bytes
	}
	// Only if it is now empty. Leaving a stray directory behind contradicts the
	// closing line.
	if entries, err := os.ReadDir(stateDir()); err == nil && len(entries) == 0 {
		os.Remove(stateDir())
	}
	return freed, nil
}

func shortPath(p string) string {
	if h := homeDir(); h != "." && strings.HasPrefix(p, h) {
		return "~" + p[len(h):]
	}
	return p
}
