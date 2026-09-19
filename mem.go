package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// Memory, measured rather than estimated. Screen 2's fifth line — "Measured it
// here: … · 9.9 GB peak · no swap" — is what turns Screen 1's promise into
// something checkable, so the number has to be the machine's own.
//
// The source is the kernel's lifetime peak physical footprint for the server
// process (proc_pid_rusage → ri_lifetime_max_phys_footprint). It is exact, not
// sampled, so a spike between samples cannot hide; and "phys footprint" is the
// same measure Activity Monitor and `top -stats mem` show, which is how the
// benchmark measured. It includes the Metal buffers holding the weights.
//
// It is read through the runtime's Python (ctypes, libproc) because Go cannot
// call libproc without cgo. The runtime is always installed by the time there
// is a server to measure.

const footprintPy = `import ctypes, sys
lib = ctypes.CDLL("/usr/lib/libproc.dylib")
buf = (ctypes.c_uint64 * 64)()
# RUSAGE_INFO_V4 = 4. After the 16-byte uuid (two uint64 slots): index 7 is
# ri_phys_footprint, index 28 is ri_lifetime_max_phys_footprint.
if lib.proc_pid_rusage(int(sys.argv[1]), 4, ctypes.byref(buf)) != 0:
    sys.exit(1)
print(buf[2 + 7], buf[2 + 28])`

// Footprint returns the process's current and peak physical footprint, bytes.
func Footprint(pid int) (now, peak int64, err error) {
	out, err := runtimePythonArgs(footprintPy, strconv.Itoa(pid))
	if err != nil {
		return 0, 0, fmt.Errorf("couldn't read memory use of pid %d", pid)
	}
	f := strings.Fields(out)
	if len(f) != 2 {
		return 0, 0, fmt.Errorf("unexpected footprint output %q", out)
	}
	now, _ = strconv.ParseInt(f[0], 10, 64)
	peak, _ = strconv.ParseInt(f[1], 10, 64)
	return now, peak, nil
}

// SwapUsedMB reads `sysctl vm.swapusage`: "total = 0.00M  used = 0.00M …".
func SwapUsedMB() float64 {
	return ParseSwapUsedMB(sysctlString("vm.swapusage"))
}

func ParseSwapUsedMB(s string) float64 {
	_, rest, ok := strings.Cut(s, "used = ")
	if !ok {
		return 0
	}
	v := strings.Fields(rest)[0]
	mult := 1.0
	switch {
	case strings.HasSuffix(v, "G"):
		mult = 1024
	case strings.HasSuffix(v, "K"):
		mult = 1.0 / 1024
	}
	n, _ := strconv.ParseFloat(strings.TrimRight(v, "MGK"), 64)
	return n * mult
}

// MemWatch brackets a piece of work: swap before, then the process's peak and
// swap after.
type MemWatch struct {
	pid   int
	swap0 float64
}

func WatchMemory(pid int) MemWatch { return MemWatch{pid: pid, swap0: SwapUsedMB()} }

// Stop returns the human line, e.g. "7.4 GB peak · no swap".
func (w MemWatch) Stop() string {
	_, peak, err := Footprint(w.pid)
	grew := SwapUsedMB() - w.swap0
	swap := "no swap"
	if grew > 50 {
		swap = fmt.Sprintf("swap grew %.0f MB", grew)
	}
	if err != nil || peak == 0 {
		return swap
	}
	return fmt.Sprintf("%s peak · %s", humanBytes(peak), swap)
}

// DiskNeeded is what setup will put on disk that is not there yet: the
// runtime, unless it is installed, and the model, unless a complete copy is
// already somewhere setup may read it.
func DiskNeeded(m Model, c Config) int64 {
	var n int64
	if !runtimeReady() {
		// Measured 2026-09-19: runtime 336 MB + Python 70 MB + uv 36 MB, plus
		// uv's 353 MB download cache while installing (deleted afterwards).
		n += 800_000_000
	}
	if _, _, ok := LocateModel(m, c); !ok {
		n += m.TotalBytes()
	}
	return n
}

func runtimePythonArgs(code string, args ...string) (string, error) {
	cmd := exec.Command(pythonBin(), append([]string{"-c", code}, args...)...)
	cmd.Env = uvEnv()
	out, err := cmd.Output()
	return string(out), err
}
