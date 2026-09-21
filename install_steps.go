package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Getting the model onto the Mac:
//
//   - Look in ~/.cache/huggingface first. Ollama and LM Studio share it, and
//     "Found it already on your Mac. Nothing to download." is the best
//     sentence this onboarding can produce. It is read, never written or
//     deleted — it is not ours.
//   - Otherwise download into ~/.nightsmith/models, so `remove` can take back
//     exactly what setup put there.
//   - Resume a broken transfer, and say so.
//   - Pin the revision, not just the repo.
//
// Both places use the Hugging Face cache layout, so one lookup serves both.
// The server is then given the snapshot's path, not the repo id: mlx_lm loads
// a local path as-is and never asks the network which revision is current.

// hubDirs lists the caches to search, shared first. The shared one honours the
// same variables the Hugging Face tools themselves do, so a user who moved it
// is found where they put it.
func hubDirs(c Config) []string {
	shared := filepath.Join(hfCacheDir(), "hub")
	if v := os.Getenv("HF_HUB_CACHE"); v != "" {
		shared = v
	} else if v := os.Getenv("HF_HOME"); v != "" {
		shared = filepath.Join(v, "hub")
	}
	return []string{shared, ownHubDir(c)}
}

// sharedCacheRoot is the whole shared directory, for `remove` to name and
// spare: ~/.cache/huggingface by default, or wherever the variables moved it.
func sharedCacheRoot() string {
	if v := os.Getenv("HF_HUB_CACHE"); v != "" {
		return v
	}
	if v := os.Getenv("HF_HOME"); v != "" {
		return v
	}
	return hfCacheDir()
}

func ownHubDir(c Config) string { return filepath.Join(c.ModelsDir, "hub") }

func repoCacheDir(hub, repo string) string {
	return filepath.Join(hub, "models--"+strings.ReplaceAll(repo, "/", "--"))
}

// FindSnapshot returns the path of a complete copy of m in one hub cache.
//
// "Complete" is checked, not assumed: the safetensors in the snapshot must add
// up to exactly the size the table records. A folder with a missing shard
// looks like a model and fails on load — or, worse, the server starts and
// serves nothing, which is the failure this product exists to catch.
func FindSnapshot(hub string, m Model) (string, bool) {
	dir := repoCacheDir(hub, m.Repo)
	rev := m.Revision
	if rev == "" {
		// An unchecked model has no pinned revision; take what was downloaded.
		b, err := os.ReadFile(filepath.Join(dir, "refs", "main"))
		if err != nil {
			return "", false
		}
		rev = strings.TrimSpace(string(b))
	}
	snap := filepath.Join(dir, "snapshots", rev)
	if _, err := os.Stat(filepath.Join(snap, "config.json")); err != nil {
		return "", false
	}
	var weights int64
	shards := 0
	entries, err := os.ReadDir(snap)
	if err != nil {
		return "", false
	}
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".safetensors") {
			continue
		}
		fi, err := os.Stat(filepath.Join(snap, e.Name())) // follows the blob symlink
		if err != nil {
			return "", false // a dangling link is a missing shard
		}
		weights += fi.Size()
		shards++
	}
	if shards == 0 {
		return "", false
	}
	if m.WeightsBytes > 0 && weights != m.WeightsBytes {
		return "", false
	}
	return snap, true
}

// LocateModel finds a complete copy anywhere we are allowed to read.
func LocateModel(m Model, c Config) (path string, shared bool, ok bool) {
	dirs := hubDirs(c)
	for i, hub := range dirs {
		if p, ok := FindSnapshot(hub, m); ok {
			return p, i == 0 && hub != ownHubDir(c), true
		}
	}
	return "", false, false
}

// partialBytes is how much of the model is already here: finished blobs plus
// partial ones. Blobs may be symlinks into a hub-wide store (huggingface_hub
// 1.x), so each is stat'ed through its link.
func partialBytes(c Config, m Model) int64 {
	var n int64
	entries, _ := os.ReadDir(filepath.Join(repoCacheDir(ownHubDir(c), m.Repo), "blobs"))
	for _, e := range entries {
		if fi, err := os.Stat(filepath.Join(repoCacheDir(ownHubDir(c), m.Repo), "blobs", e.Name())); err == nil && !fi.IsDir() {
			n += fi.Size()
		}
	}
	return n
}

// removeOrphanPartials deletes .incomplete files once the model is complete.
// A finished download never needs them, and an abandoned one can be gigabytes.
// Only our own cache — the shared one is never written.
func removeOrphanPartials(c Config, m Model) {
	dir := filepath.Join(repoCacheDir(ownHubDir(c), m.Repo), "blobs")
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".incomplete") {
			os.Remove(filepath.Join(dir, e.Name()))
		}
	}
}

// ensureModel makes sure a complete, pinned copy of m is on this Mac and
// returns its path.
func ensureModel(m Model, c Config) (string, error) {
	if p, shared, ok := LocateModel(m, c); ok {
		removeOrphanPartials(c, m) // left by an older, interrupted attempt
		if shared {
			printf("  ✓  Found %s already on your Mac. Nothing to download.\n", shortRepo(m.Repo))
		} else {
			printf("  ✓  %s is already downloaded\n", shortRepo(m.Repo))
		}
		return p, nil
	}

	if have := partialBytes(c, m); have > 0 && m.TotalBytes() > 0 {
		printf("     Picking up where it stopped — %s of %s already here.\n",
			humanBytes(have), humanBytes(m.TotalBytes()))
	} else {
		printf("     Downloading %s (%s)…\n", shortRepo(m.Repo), humanBytes(m.TotalBytes()))
	}

	start := time.Now()
	rev := m.Revision
	if rev == "" {
		rev = "main"
	}
	// huggingface_hub is already in the runtime (mlx-lm depends on it), and
	// over plain HTTP it resumes its .incomplete files on its own — see
	// HF_HUB_DISABLE_XET below. The patterns are mlx_lm's own, so exactly the
	// files the server will read are fetched and nothing else.
	cmd := exec.Command(pythonBin(), "-c", `import sys
from huggingface_hub import snapshot_download
from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS
snapshot_download(sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3],
                  allow_patterns=DEFAULT_ALLOW_PATTERNS)`,
		m.Repo, rev, ownHubDir(c))
	cmd.Env = append(uvEnv(),
		"HF_HOME="+c.ModelsDir, // keeps tokens, caches, everything, out of ~/.cache
		"HF_HUB_DISABLE_TELEMETRY=1",
		// Its own bar counts files, not bytes: "Fetching 9 files 78%" sits
		// unchanged for minutes while 6.8 GB arrives. Ours counts bytes.
		"HF_HUB_DISABLE_PROGRESS_BARS=1",
		// Xet transfers do not resume. Measured 2026-09-19: a download killed
		// at 3.0 GB restarted from zero and left the 3.2 GB of partial files
		// behind as orphans. Plain HTTP resumes its .incomplete files with a
		// Range request — the same files kept growing across a kill.
		"HF_HUB_DISABLE_XET=1",
	)
	var out strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Start(); err != nil {
		return "", err
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	tick := time.NewTicker(time.Second)
	var err error
wait:
	for {
		select {
		case err = <-done:
			break wait
		case <-tick.C:
			printf("\r     %s  %s of %s ", progressBar(partialBytes(c, m), m.TotalBytes(), 24),
				humanBytes(partialBytes(c, m)), humanBytes(m.TotalBytes()))
		}
	}
	tick.Stop()
	printf("\r%s\r", strings.Repeat(" ", 60))
	if err != nil {
		return "", fmt.Errorf("the download stopped: %v\n%s\n\n"+
			"    What arrived is kept. Run 'nightsmith' again and it picks up\n"+
			"    where it stopped.", err, indentTail(out.String(), 6))
	}

	removeOrphanPartials(c, m)
	p, _, ok := LocateModel(m, c)
	if !ok {
		return "", fmt.Errorf("the download finished, but %s isn't complete on disk — its\n"+
			"    size doesn't match what the model table records. Run 'nightsmith'\n"+
			"    again to fetch what's missing.", shortRepo(m.Repo))
	}
	printf("  ✓  Downloaded %s   %s   %s\n", shortRepo(m.Repo),
		humanBytes(m.TotalBytes()), humanDuration(time.Since(start).Seconds()))
	return p, nil
}

func progressBar(have, total int64, width int) string {
	if total <= 0 {
		return ""
	}
	n := int(float64(width) * float64(have) / float64(total))
	if n > width {
		n = width
	}
	return strings.Repeat("▓", n) + strings.Repeat("░", width-n)
}
