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

// What the server will take at once, mirrored here so the tool can explain a
// refusal without asking the server, and so `-p` can refuse an oversized
// prompt before sending it anywhere.
//
// serve.py holds the reasoning and the measurements, and is the source of
// truth: server_test.go reads these values back out of it and fails if the
// two drift apart.
const (
	maxPromptTokens       = 40_000
	maxConcurrentRequests = 3

	// Where this side stops sending without asking. Higher than the ceiling
	// because this side estimates from characters and the server counts with
	// the model's tokenizer: half as much again is past anything the two could
	// disagree about, and everything below it is the server's call to make.
	certainlyTooLarge = maxPromptTokens * 3 / 2
)

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
	return append([]string{
		serveScriptPath(),
		"--served-model-id", m.Repo,
		// How long serve.py lets answers in progress finish after a stop.
		// Passed rather than hard-coded at both ends, because the right
		// number depends on max_tokens and there is no way to keep two
		// constants equal by asking people to remember.
		"--drain-seconds", strconv.Itoa(int(DrainWindow(c.MaxTokens).Seconds())),
	}, ServerArgs(c, m, modelPath)...)
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
	if err := os.WriteFile(runningConfigPath(), []byte(renderRunningConfig(*c)), 0o644); err != nil {
		return 0, err
	}
	cmd := exec.Command(ensureModelBin(), ServerCommand(*c, m, modelPath)...)
	cmd.Env = append(uvEnv(), "HF_HOME="+c.ModelsDir, "HF_HUB_DISABLE_TELEMETRY=1")
	if c.Offline {
		// Stops a background task stalling on the network, and stops weights
		// changing under you between runs.
		cmd.Env = append(cmd.Env, "HF_HUB_OFFLINE=1")
	}
	rotateLog()
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

func serverLog() string   { return filepath.Join(stateDir(), "server.log") }
func previousLog() string { return serverLog() + ".1" }

func readLog() string {
	b, _ := os.ReadFile(serverLog())
	return string(b)
}

// rotateLog keeps the last run's log instead of overwriting it. A server that
// ran out of memory says why and then exits, and the next start is usually
// the very next thing that happens — so truncating here would delete the only
// evidence of the failure the user is trying to explain.
func rotateLog() {
	if fi, err := os.Stat(serverLog()); err != nil || fi.Size() == 0 {
		return
	}
	os.Remove(previousLog())
	os.Rename(serverLog(), previousLog())
}

// fatalMarker is what serve.py writes when the generation thread has died.
// It must match FATAL_MARKER there.
const fatalMarker = "nightsmith-fatal:"

// StoppedOnItsOwn explains a server that is no longer running but was never
// stopped — the PID file is still there, and the log says the model ran out
// of memory. Empty when the server was stopped normally, or said nothing.
//
// This exists because the fix for the post-OOM zombie was to let the process
// exit. Without it the failure would go from a wrong answer (404 for ever) to
// no answer at all: "set up, but not running", with the reason on the floor.
// Only the current log is read, never the rotated one. server.log is written
// by the process the PID file names, because a start rotates before it opens
// one — so the rotated log belongs to some earlier run, and a crash recorded
// in it would be told as though it had just happened.
func StoppedOnItsOwn() string {
	if _, recorded := readPID(); !recorded {
		return "" // a clean stop removes the PID file
	}
	b, err := os.ReadFile(serverLog())
	if err != nil {
		return ""
	}
	return lastLineWith(string(b), fatalMarker)
}

func lastLineWith(log, marker string) string {
	found := ""
	for _, line := range strings.Split(log, "\n") {
		if i := strings.Index(line, marker); i >= 0 {
			found = strings.TrimSpace(line[i+len(marker):])
		}
	}
	return found
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
// progress finish. How long that takes is not a constant: it is a whole
// answer's worth of writing.
//
// It used to be a flat 30 s against max_tokens = 1500 at 12.6 tokens/s —
// 125 s of work given 30 s to finish, so `nightsmith stop` silently threw
// away any answer longer than about 360 tokens. The window is derived from
// the two numbers that decide it instead, and passed to serve.py so there is
// one of it rather than two that have to be kept equal.
const (
	measuredDecodeTokS = 12.6 // base M4 / 16 GB, flat over a 92-minute soak
	drainMargin        = 1.25 // slower machines, and the last token's overhead
	minDrainWindow     = 30 * time.Second
	maxDrainWindow     = 5 * time.Minute // a stop must still feel like a stop
	unmapWeights       = 15 * time.Second
)

// DrainWindow is how long an answer in progress is given to finish.
func DrainWindow(maxTokens int) time.Duration {
	if maxTokens <= 0 {
		return minDrainWindow
	}
	d := time.Duration(float64(maxTokens) / measuredDecodeTokS * drainMargin * float64(time.Second))
	switch {
	case d < minDrainWindow:
		return minDrainWindow
	case d > maxDrainWindow:
		return maxDrainWindow
	}
	return d.Round(time.Second)
}

// RunningDrainWindow is the window the server that is up was started with —
// read from the settings it recorded at launch, not from the file the user
// may have edited since.
func RunningDrainWindow() time.Duration {
	if c, _, err := readConfigFile(runningConfigPath()); err == nil {
		return DrainWindow(c.MaxTokens)
	}
	if c, err := LoadConfig(); err == nil {
		return DrainWindow(c.MaxTokens)
	}
	return minDrainWindow
}

// stopWait is the drain plus the time it takes to unmap 7 GB of weights.
func stopWait() time.Duration { return RunningDrainWindow() + unmapWeights }

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
	wait := stopWait()
	for deadline := time.Now().Add(wait); time.Now().Before(deadline); {
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
