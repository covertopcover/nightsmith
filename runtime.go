package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// The runtime: uv, a private Python, and mlx-lm pinned to a git commit.
// The chain collapses to one static binary that vendors
// its own Python, which installs pinned wheels. Nothing compiles, so no Xcode;
// no Homebrew; no password. Nothing outside ~/.nightsmith is touched.
//
// Layout, all under ~/.nightsmith:
//
//	uv/        the uv binary. NOT inside runtime/: uv refuses to create a
//	           venv in a directory that already has files in it
//	python/    the Python uv downloads (UV_PYTHON_INSTALL_DIR)
//	runtime/   the venv the server runs from — pythonBin() points here
//
// Verified on a base M4 / 16 GB under macOS 26.6 and 27.0, 2026-09-19.

const (
	uvVersion = "0.12.17"
	// From the release's own .sha256 file. A mismatch stops the install: this
	// binary is about to be executed.
	uvSHA256      = "85f00cbdc6dd3e97eba4c31b4d014375a9fdfe8f570023b84e5102fc3456896b"
	pythonVersion = "3.12.14"
	// Must match the mlx-lm line in runtime.lock. It is what the version check
	// below asserts, because a PyPI 0.31.3 that slipped in would look healthy.
	mlxLMVersion = "0.32.0"
)

//go:embed runtime.lock
var runtimeLock []byte

func uvBin() string        { return filepath.Join(uvDir(), "uv") }
func runtimeStamp() string { return filepath.Join(runtimeDir(), ".nightsmith-runtime") }

// wantStamp identifies exactly what a finished install contains. If any pin
// changes, the stamp changes, and the runtime is rebuilt rather than trusted.
func wantStamp() string {
	h := sha256.Sum256(append([]byte(uvVersion+"\n"+pythonVersion+"\n"), runtimeLock...))
	return hex.EncodeToString(h[:])
}

// runtimeReady says whether a complete, current runtime is already here. It is
// what makes a second `nightsmith` instant: the stamp is written last, so a
// half-finished install never has one.
func runtimeReady() bool {
	b, err := os.ReadFile(runtimeStamp())
	if err != nil || strings.TrimSpace(string(b)) != wantStamp() {
		return false
	}
	_, err = os.Stat(pythonBin())
	return err == nil
}

// uvEnv is the environment every uv call runs in. It points uv at our
// directories and strips anything from the user's shell that could reach in —
// an active venv, a PYTHONPATH, their own uv config.
func uvEnv() []string {
	var env []string
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(k, "UV_") || strings.HasPrefix(k, "PYTHON") ||
			k == "VIRTUAL_ENV" || k == "CONDA_PREFIX" || k == "PIP_INDEX_URL" {
			continue
		}
		env = append(env, kv)
	}
	return append(env,
		"UV_PYTHON_INSTALL_DIR="+pythonDir(),
		"UV_CACHE_DIR="+uvCacheDir(),
		"UV_NO_CONFIG=1",
	)
}

// ensureRuntime installs the runtime, or confirms it is already there.
func ensureRuntime() error {
	if runtimeReady() {
		fmt.Printf("  ✓  Runtime ready                    already installed\n")
		return nil
	}
	start := time.Now()
	fmt.Printf("     Installing the runtime (Python and MLX, ~450 MB)…\n")

	if err := os.MkdirAll(stateDir(), 0o755); err != nil {
		return err
	}
	if err := installUV(); err != nil {
		return err
	}
	// --managed-python: never the system's Python, never Homebrew's. (Not also
	// UV_PYTHON_PREFERENCE — uv rejects the two together.)
	// --clear: a previous attempt may have left a partial venv behind.
	if out, err := runUV("venv", "--clear", "--quiet", "--managed-python",
		"--python", pythonVersion, runtimeDir()); err != nil {
		return runtimeFailed("couldn't install Python", out, err)
	}

	lock := filepath.Join(stateDir(), "runtime.lock")
	if err := os.WriteFile(lock, runtimeLock, 0o644); err != nil {
		return err
	}
	defer os.Remove(lock)
	if out, err := runUV("pip", "install", "--quiet", "--python", pythonBin(),
		"-r", lock); err != nil {
		return runtimeFailed("couldn't install MLX", out, err)
	}

	// Prove the pin took, rather than trusting that it did. This imports the
	// real thing on the real GPU; a wrong mlx-lm would pass every step above.
	got, err := runtimePython(`import mlx_lm, mlx.core as mx
assert mx.default_device() == mx.gpu, "MLX has no GPU"
print(mlx_lm.__version__)`)
	if err != nil {
		return runtimeFailed("installed MLX, but it doesn't run", got, err)
	}
	if strings.TrimSpace(got) != mlxLMVersion {
		return fmt.Errorf("installed mlx-lm %s, expected %s — the pin did not hold",
			strings.TrimSpace(got), mlxLMVersion)
	}

	// The download cache is ~350 MB of wheels that are already installed.
	os.RemoveAll(uvCacheDir())

	if err := os.WriteFile(runtimeStamp(), []byte(wantStamp()+"\n"), 0o644); err != nil {
		return err
	}
	fmt.Printf("  ✓  Runtime ready                    %s\n", humanDuration(time.Since(start).Seconds()))
	return nil
}

func runtimeFailed(what, out string, err error) error {
	return fmt.Errorf("%s: %v\n\n%s\n\n    Nothing outside ~/.nightsmith was touched. Running 'nightsmith'\n"+
		"    again picks up from here.", what, err, indentTail(out, 12))
}

// indentTail keeps the last n lines of a tool's output, indented. Enough to
// report the problem, not a wall of pip.
func indentTail(out string, n int) string {
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	for i := range lines {
		lines[i] = "    " + lines[i]
	}
	return strings.Join(lines, "\n")
}

func runUV(args ...string) (string, error) {
	cmd := exec.Command(uvBin(), args...)
	cmd.Env = uvEnv()
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// runtimePython runs a snippet in the installed runtime and returns stdout.
func runtimePython(code string) (string, error) {
	cmd := exec.Command(pythonBin(), "-c", code)
	cmd.Env = uvEnv()
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return stderr.String(), err
	}
	return stdout.String(), nil
}

// RuntimeCeilingBytes asks Metal for this Mac's GPU working-set limit —
// recommendedMaxWorkingSetSize, which is the number a model must fit under.
// It needs no cgo: MLX exposes it, and MLX is already installed once setup has
// run. Returns false before then, and the caller falls back to an estimate.
func RuntimeCeilingBytes() (int64, bool) {
	if !runtimeReady() {
		return 0, false
	}
	out, err := runtimePython(`import mlx.core as mx
print(mx.device_info()["max_recommended_working_set_size"])`)
	if err != nil {
		return 0, false
	}
	n, err := strconv.ParseInt(strings.TrimSpace(out), 10, 64)
	return n, err == nil && n > 0
}

// installUV fetches the pinned uv release, checks it against the pinned
// checksum, and unpacks only the uv binary.
func installUV() error {
	if b, err := exec.Command(uvBin(), "--version").Output(); err == nil &&
		strings.Contains(string(b), uvVersion) {
		return nil
	}
	url := fmt.Sprintf("https://github.com/astral-sh/uv/releases/download/%s/uv-aarch64-apple-darwin.tar.gz", uvVersion)
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("couldn't download uv — check your connection and run 'nightsmith' again: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("couldn't download uv: %s returned %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("the uv download was cut off — run 'nightsmith' again: %w", err)
	}
	if sum := sha256.Sum256(body); hex.EncodeToString(sum[:]) != uvSHA256 {
		return fmt.Errorf("the uv download doesn't match its pinned checksum, so it was not run")
	}

	gz, err := gzip.NewReader(bytes.NewReader(body))
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return fmt.Errorf("the uv archive has no uv binary in it")
		}
		if err != nil {
			return err
		}
		if filepath.Base(h.Name) != "uv" || h.Typeflag != tar.TypeReg {
			continue
		}
		if err := os.MkdirAll(uvDir(), 0o755); err != nil {
			return err
		}
		tmp := uvBin() + ".partial"
		f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
		if err != nil {
			return err
		}
		if _, err := io.Copy(f, tr); err != nil {
			f.Close()
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
		return os.Rename(tmp, uvBin())
	}
}
