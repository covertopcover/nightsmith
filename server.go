package main

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// mlx_lm.server's lifecycle. It is the runtime because it
// is the one measured on the target machine — 9.9 GB peak, 12.6 tok/s, a
// 92-minute soak flat, deterministic. vllm-mlx's extra features all serve
// use cases that are out of scope.
//
// Start/stop is in scope because mlx_lm.server has no keep-alive and needs
// several flags to launch correctly; leaving a non-developer holding that
// command line is not a finished setup. Supervision and scheduling are NOT in
// scope — that is an open question, and it is not answered here.

// ServerArgs builds the command line. Kept separate from running it so the
// flags can be tested without a Mac, which matters because a wrong flag here
// fails silently: mlx_lm.server accepts every known-bad flag happily and breaks
// only when real work arrives.
//
// modelPath is the local snapshot from ensureModel, not the repo id: mlx_lm
// loads a path as-is, so the pinned revision is exactly what runs.
func ServerArgs(c Config, m Model, modelPath string) []string {
	args := []string{
		"--model", modelPath,
		"--port", strconv.Itoa(c.Port),
		"--host", "127.0.0.1", // local only; this is not a network service

		// The cap that keeps peak memory at 9.9 GB instead of 12.3. Plain
		// bytes: the benchmark's "512e6" is rejected by the pinned mlx-lm,
		// whose size parser now takes only digits with an M/MB/G/GB suffix.
		"--prompt-cache-bytes", strconv.Itoa(c.PromptCacheMB * 1_000_000),
		"--prompt-cache-size", strconv.Itoa(c.PromptCacheSlots),

		// The defaults for a client that sends neither. Without them a
		// client got mlx_lm's own (512 tokens) whatever the config said, and
		// both settings changed only nightsmith's own probe.
		"--max-tokens", strconv.Itoa(c.MaxTokens),
		"--temp", strconv.FormatFloat(c.Temperature, 'f', -1, 64),
	}

	// The thinking setting as the server's default, so a client that sends
	// nothing gets the configured behaviour — not mlx_lm's, which is to think
	// whenever the model can (see models.toml, Gemma 4).
	if kw := ThinkingKwargs(m, c.Thinking); kw != nil {
		if b, err := json.Marshal(kw); err == nil {
			args = append(args, "--chat-template-args", string(b))
		}
	}

	// Deliberately absent, for every model: --kv-bits and --draft-model.
	// models.toml records why per architecture. For Gemma 4 the rotating KV
	// cache has no quantized implementation and all 30 requests failed; the
	// official draft model will not load and a substitute corrupted every
	// answer, then OOMed. The server accepts both flags without complaint.
	for flag := range m.BrokenFlags {
		_ = flag // never emitted; present in the table so `model list` can explain
	}

	return args
}

// serve.py wraps the pinned mlx_lm server so its HTTP edges tell clients the
// truth: a 400 instead of a dropped connection, a /health that says when the
// model is still loading, the repo id instead of a path in /v1/models, and an
// unknown model id refused before the server unloads the real one to look for
// it. Generation itself is untouched. Written out on every start, so a new
// binary always runs its own copy.
//
//go:embed serve.py
var serveScript []byte

func serveScriptPath() string { return filepath.Join(stateDir(), "serve.py") }

// ServerCommand is the full argv after the interpreter.
func ServerCommand(c Config, m Model, modelPath string) []string {
	return append([]string{serveScriptPath(), "--served-model-id", m.Repo},
		ServerArgs(c, m, modelPath)...)
}

// modelBin is the name the server runs under. macOS names a process after
// its executable file, so without this Activity Monitor shows "python3.12" —
// and argv[0] cannot change that. It is a hard link to the runtime's Python:
// same inode, so the same code signature and no extra disk, and Python still
// finds the venv through pyvenv.cfg one directory up.
func modelBin() string { return filepath.Join(runtimeDir(), "bin", "nightsmith-model") }

// ensureModelBin returns modelBin, (re)linking it if needed, or pythonBin if
// the link cannot be made. A wrong process name is not worth failing a start.
func ensureModelBin() string {
	target, err := filepath.EvalSymlinks(pythonBin())
	if err != nil {
		return pythonBin()
	}
	ti, err := os.Stat(target)
	if err != nil {
		return pythonBin()
	}
	if li, err := os.Lstat(modelBin()); err == nil {
		if os.SameFile(li, ti) {
			return modelBin()
		}
		os.Remove(modelBin()) // left by an older runtime
	}
	if err := os.Link(target, modelBin()); err != nil {
		return pythonBin()
	}
	return modelBin()
}

// pythonBin is the interpreter inside the vendored runtime. Never the system
// Python, never Homebrew's: the whole point is that nothing outside
// ~/.nightsmith is touched or depended on.
func pythonBin() string { return filepath.Join(runtimeDir(), "bin", "python3") }

func writePID(pid int) error {
	return os.WriteFile(pidPath(), []byte(strconv.Itoa(pid)), 0o644)
}

func readPID() (int, bool) {
	b, err := os.ReadFile(pidPath())
	if err != nil {
		return 0, false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil || pid <= 0 {
		return 0, false
	}
	return pid, true
}

// ProcessAlive reports whether the recorded PID is still there. Signal 0 checks
// for existence without touching the process.
//
// This answers "is something running", which is NOT the same question as "can
// it serve a request" — see probe.go. A post-OOM zombie is alive by this test
// and useless by the only one that matters.
func ProcessAlive(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(syscall.Signal(0)) == nil
}

// PortFree reports whether nothing is listening on this port. 8080 is the most
// common development port there is; the rule is to pick another
// silently rather than fail. It also matters for honesty: a probe sent to
// someone else's server on 8080 could print a checkmark for a model that
// never started.
func PortFree(port int) bool {
	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false
	}
	l.Close()
	return true
}

// FreePort returns preferred if it is free, otherwise the next free one.
func FreePort(preferred int) (int, error) {
	for p := preferred; p < preferred+100; p++ {
		if PortFree(p) {
			return p, nil
		}
	}
	return 0, fmt.Errorf("ports %d–%d are all in use", preferred, preferred+99)
}

// StartServer launches the server and waits until it is listening. Listening
// is not the same as working — only Probe can say that — but probing a port
// nothing has opened yet fails instantly, and that failure is a lie too.
//
// If the configured port has been taken by another program since setup, a free
// one is chosen and c.Port updated; the caller saves it.
func StartServer(c *Config, m Model) (int, error) {
	if pid, ok := ServerPID(); ok {
		return pid, nil // idempotent: re-running a command to check it worked is
		// what non-developers do, and it must not start a second server
	}

	modelPath, _, ok := LocateModel(m, *c)
	if !ok {
		return 0, fmt.Errorf("%s isn't on this Mac any more — run 'nightsmith' to fetch it again",
			shortRepo(m.Repo))
	}
	if !PortFree(c.Port) {
		p, err := FreePort(c.Port + 1)
		if err != nil {
			return 0, err
		}
		c.Port = p
	}

	if err := os.MkdirAll(stateDir(), 0o755); err != nil {
		return 0, err
	}
	if err := os.WriteFile(serveScriptPath(), serveScript, 0o644); err != nil {
		return 0, err
	}
	if err := os.WriteFile(runningConfigPath(), []byte(renderConfig(*c)), 0o644); err != nil {
		return 0, err
	}
	cmd := exec.Command(ensureModelBin(), ServerCommand(*c, m, modelPath)...)
	cmd.Env = append(uvEnv(), "HF_HOME="+c.ModelsDir, "HF_HUB_DISABLE_TELEMETRY=1")
	if c.Offline {
		// Stops a background task stalling on the network, and stops weights
		// changing under you between runs.
		cmd.Env = append(cmd.Env, "HF_HUB_OFFLINE=1")
	}
	logFile, err := os.OpenFile(serverLog(),
		os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return 0, err
	}
	cmd.Stdout, cmd.Stderr = logFile, logFile
	cmd.SysProcAttr = detachedProcAttr() // survives this terminal closing

	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("couldn't start the model server: %w", err)
	}
	logFile.Close() // the child has its own handle
	pid := cmd.Process.Pid
	if err := writePID(pid); err != nil {
		return pid, err
	}
	// Reap it if it dies while we are still here, so ProcessAlive sees the
	// death instead of a zombie.
	exited := make(chan struct{})
	go func() { cmd.Wait(); close(exited) }()

	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		select {
		case <-exited:
			os.Remove(pidPath())
			return 0, fmt.Errorf("the model server stopped as it started. Its last words:\n\n%s",
				indentTail(readLog(), 12))
		default:
		}
		if conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", c.Port), time.Second); err == nil {
			conn.Close()
			return pid, nil
		}
		time.Sleep(250 * time.Millisecond)
	}
	return pid, fmt.Errorf("the model server started but never opened port %d", c.Port)
}

func serverLog() string { return filepath.Join(stateDir(), "server.log") }

func readLog() string {
	b, _ := os.ReadFile(serverLog())
	return string(b)
}

// ServerPID returns the running server's PID — only if the recorded process is
// alive AND is our server. A PID file outlives a reboot, and PIDs are reused:
// trusting the number alone could report a stranger's process as the model,
// or send it a SIGTERM.
func ServerPID() (int, bool) {
	pid, ok := readPID()
	if !ok || !ProcessAlive(pid) {
		return 0, false
	}
	out, err := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "command=").Output()
	if err != nil {
		return 0, false
	}
	return pid, isOurServer(string(out))
}

// isOurServer recognises both launch forms: serve.py, and the bare
// `-m mlx_lm server` of v0.1.1 and earlier — so `stop` after an upgrade still
// finds a server the previous version started.
func isOurServer(cmdline string) bool {
	if !strings.Contains(cmdline, runtimeDir()) {
		return false
	}
	return strings.Contains(cmdline, serveScriptPath()+" --served-model-id") ||
		strings.Contains(cmdline, "mlx_lm server")
}

// On SIGTERM, serve.py stops taking new completions and lets the ones in
// progress finish, for up to drainSeconds (its DRAIN_SECONDS — keep the two
// equal). StopServer waits stopWait before SIGKILL: the drain plus the time
// it takes to unmap 7 GB of weights.
const (
	drainSeconds = 30
	stopWait     = (drainSeconds + 15) * time.Second
)

// InFlight asks the server's /health how many requests it is working on.
// Zero on any failure: it only decides whether to say "finishing…".
func InFlight(port int) int {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	var body struct {
		InFlight int `json:"in_flight"`
	}
	if json.NewDecoder(resp.Body).Decode(&body) != nil {
		return 0
	}
	return body.InFlight
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// StopServer stops the server and waits until it has actually gone. Returning
// early is not harmless: `model use` starts the next model straight after, and
// two models resident on 16 GB is the Metal OOM the benchmark hit.
func StopServer() (bool, error) {
	pid, ok := ServerPID()
	if !ok {
		os.Remove(pidPath()) // stale, or never ours
		return false, nil
	}
	defer os.Remove(pidPath())
	defer os.Remove(runningConfigPath())
	p, err := os.FindProcess(pid)
	if err != nil {
		return false, err
	}
	if err := p.Signal(syscall.SIGTERM); err != nil {
		return false, err
	}
	for deadline := time.Now().Add(stopWait); time.Now().Before(deadline); {
		if !ProcessAlive(pid) {
			return true, nil
		}
		time.Sleep(250 * time.Millisecond)
	}
	p.Signal(syscall.SIGKILL)
	for i := 0; i < 20 && ProcessAlive(pid); i++ {
		time.Sleep(250 * time.Millisecond)
	}
	return true, nil
}
