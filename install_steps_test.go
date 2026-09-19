package main

import (
	"os"
	"path/filepath"
	"testing"
)

// fakeSnapshot lays out a Hugging Face cache the way huggingface_hub does:
// snapshot files are symlinks into blobs/.
func fakeSnapshot(t *testing.T, hub string, m Model, shardSizes ...int) string {
	t.Helper()
	repo := repoCacheDir(hub, m.Repo)
	snap := filepath.Join(repo, "snapshots", m.Revision)
	blobs := filepath.Join(repo, "blobs")
	for _, d := range []string{snap, blobs} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	os.WriteFile(filepath.Join(snap, "config.json"), []byte("{}"), 0o644)
	for i, n := range shardSizes {
		blob := filepath.Join(blobs, "shard"+string(rune('a'+i)))
		os.WriteFile(blob, make([]byte, n), 0o644)
		os.Symlink(blob, filepath.Join(snap, "model-"+string(rune('a'+i))+".safetensors"))
	}
	return snap
}

func TestFindSnapshotAcceptsOnlyAComplete(t *testing.T) {
	m := Model{Repo: "mlx-community/x-4bit", Revision: "abc123", WeightsBytes: 300}

	hub := t.TempDir()
	want := fakeSnapshot(t, hub, m, 100, 200)
	if got, ok := FindSnapshot(hub, m); !ok || got != want {
		t.Fatalf("complete snapshot not found: %q %v", got, ok)
	}

	// One shard missing: sizes no longer add up. This is the folder that
	// looks like a model and is not one.
	short := t.TempDir()
	fakeSnapshot(t, short, m, 100)
	if _, ok := FindSnapshot(short, m); ok {
		t.Error("a snapshot missing a shard must not count as downloaded")
	}

	// A shard whose blob is gone — the link dangles.
	dangling := t.TempDir()
	snap := fakeSnapshot(t, dangling, m, 100, 200)
	target, _ := os.Readlink(filepath.Join(snap, "model-b.safetensors"))
	os.Remove(target)
	if _, ok := FindSnapshot(dangling, m); ok {
		t.Error("a dangling shard link must not count as downloaded")
	}

	// A different revision is a different model, however complete.
	other := m
	other.Revision = "def456"
	if _, ok := FindSnapshot(hub, other); ok {
		t.Error("a snapshot at another revision must not satisfy the pin")
	}
}

func TestSharedCacheIsSearchedFirst(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", filepath.Join(t.TempDir(), "shared"))
	c := Config{ModelsDir: t.TempDir()}
	m := Model{Repo: "mlx-community/x-4bit", Revision: "abc123", WeightsBytes: 100}

	fakeSnapshot(t, ownHubDir(c), m, 100)
	if _, shared, ok := LocateModel(m, c); !ok || shared {
		t.Fatalf("own copy: ok=%v shared=%v", ok, shared)
	}
	fakeSnapshot(t, os.Getenv("HF_HUB_CACHE"), m, 100)
	if _, shared, ok := LocateModel(m, c); !ok || !shared {
		t.Errorf("a complete copy in the shared cache must be preferred: ok=%v shared=%v", ok, shared)
	}
}
