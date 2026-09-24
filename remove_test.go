package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Rule 3, and the one with the worst consequence if broken: deleting a cache
// two other tools depend on.
func TestRemovalNeverTouchesTheSharedCache(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	mustWrite(t, filepath.Join(home, ".nightsmith", "models", "weights.safetensors"), 4096)
	mustWrite(t, filepath.Join(home, ".cache", "huggingface", "blobs", "shared"), 8192)

	r := PlanRemoval()
	for _, it := range r.Items {
		if strings.Contains(it.Path, ".cache/huggingface") {
			t.Fatalf("the shared cache is queued for deletion: %s", it.Path)
		}
	}
	if len(r.Shared) != 1 {
		t.Fatalf("the shared cache should be listed but spared, got %d entries", len(r.Shared))
	}
	if !strings.Contains(r.Describe(), "does not touch") {
		t.Error("the screen must say out loud that the shared cache is spared")
	}

	if _, err := r.Execute(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(home, ".cache", "huggingface", "blobs", "shared")); err != nil {
		t.Fatal("the shared cache was deleted — Ollama and LM Studio depend on it")
	}
}

// Rule 1 and 4: the total is read off disk, and it is what actually comes back.
func TestRemovalTotalIsTrue(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	mustWrite(t, filepath.Join(home, ".nightsmith", "models", "a"), 3_000_000)
	mustWrite(t, filepath.Join(home, ".nightsmith", "runtime", "b"), 1_000_000)
	mustWrite(t, filepath.Join(home, ".nightsmith", "config.toml"), 500)

	r := PlanRemoval()
	if r.TotalBytes != 4_000_500 {
		t.Errorf("total = %d, want 4,000,500 — summed from disk, not from a table", r.TotalBytes)
	}
	freed, err := r.Execute()
	if err != nil {
		t.Fatal(err)
	}
	if freed != r.TotalBytes {
		t.Errorf("reported %d freed but planned %d; the closing line would be a lie", freed, r.TotalBytes)
	}
	if _, err := os.Stat(filepath.Join(home, ".nightsmith")); !os.IsNotExist(err) {
		t.Error("~/.nightsmith should be gone — 'back to how it was' has to be literal")
	}
}

func TestRemovalOnACleanMacSaysSo(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	r := PlanRemoval()
	if r.TotalBytes != 0 {
		t.Errorf("nothing is installed, so nothing should be queued, got %d bytes", r.TotalBytes)
	}
	if !strings.Contains(r.Describe(), "isn't set up") {
		t.Errorf("removing nothing should say so plainly, got:\n%s", r.Describe())
	}
}

func mustWrite(t *testing.T, path string, size int) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
}

// serve.py is written on every start; if remove left it, ~/.nightsmith would
// survive a remove that said it was gone.
func TestRemovalIncludesTheServerWrapper(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	mustWrite(t, serveScriptPath(), 100)
	for _, it := range PlanRemoval().Items {
		if it.Path == serveScriptPath() {
			return
		}
	}
	t.Error("serve.py is not in the removal plan")
}

func TestRemovalIncludesTheRunningConfigSnapshot(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	mustWrite(t, runningConfigPath(), 100)
	for _, it := range PlanRemoval().Items {
		if it.Path == runningConfigPath() {
			return
		}
	}
	t.Error("server.toml is not in the removal plan")
}

// `remove` promises the Mac is back to how it was, and a stray file makes the
// closing line false. It has been wrong once already: v0.1.7 added a rotated
// server.log.1 and did not add it here, so removal left the file — and, because
// the directory was no longer empty, the directory too.
//
// Every file nightsmith writes under ~/.nightsmith belongs in this list AND in
// PlanRemoval. If you add one to the product, add it here: the test fails
// loudly, which is the only reason it would ever be noticed.
func TestRemoveLeavesNothingBehind(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	for _, dir := range []string{modelsDir(), runtimeDir(), pythonDir(), uvDir(),
		uvCacheDir(), chatsDir()} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		mustWrite(t, filepath.Join(dir, "something"), 1024)
	}
	for _, file := range []string{configPath(), pidPath(), serverLog(), previousLog(),
		serveScriptPath(), runningConfigPath()} {
		mustWrite(t, file, 64)
	}
	os.MkdirAll(filepath.Dir(binPath()), 0o755)
	mustWrite(t, binPath(), 4096)

	if _, err := PlanRemoval().Execute(); err != nil {
		t.Fatal(err)
	}

	if entries, err := os.ReadDir(stateDir()); err == nil {
		var left []string
		for _, e := range entries {
			left = append(left, e.Name())
		}
		t.Errorf("~/.nightsmith survived, holding %v — add each to PlanRemoval", left)
	} else if !os.IsNotExist(err) {
		t.Fatal(err)
	}
	if _, err := os.Stat(binPath()); err == nil {
		t.Error("the command itself was left behind")
	}
}
