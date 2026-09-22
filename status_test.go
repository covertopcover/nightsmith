package main

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestStatusReportStates(t *testing.T) {
	cfg := Config{Model: "mlx-community/gemma-4-12B-it-4bit", Port: 8081}

	cases := []struct {
		name     string
		pid      int
		probeErr error
		state    string
		code     error
	}{
		{"not running", 0, nil, "not_running", statusNotRunning},
		{"up but silent", 42, errors.New("timed out"), "not_answering", statusNotAnswering},
		{"answered", 42, nil, "ready", nil},
	}
	for _, c := range cases {
		r := newStatusReport(cfg, c.pid, c.probeErr, 7_300_000_000)
		if r.State != c.state {
			t.Errorf("%s: state %q, want %q", c.name, r.State, c.state)
		}
		if r.exit() != c.code {
			t.Errorf("%s: exit %v, want %v", c.name, r.exit(), c.code)
		}
		if r.URL != "http://127.0.0.1:8081/v1" || r.Port != 8081 || r.Model != cfg.Model {
			t.Errorf("%s: endpoint fields wrong: %+v", c.name, r)
		}
	}
}

// Memory is only reported for a server that answered: a number next to a
// server that could not answer would read as health.
func TestStatusReportJSONOmitsWhatWasNotMeasured(t *testing.T) {
	cfg := Config{Model: "m", Port: 8080}
	b, _ := json.Marshal(newStatusReport(cfg, 42, errors.New("no answer"), 0))
	var got map[string]any
	json.Unmarshal(b, &got)
	if _, ok := got["memory_bytes"]; ok {
		t.Errorf("memory_bytes present for a server that did not answer: %s", b)
	}
	if got["error"] != "no answer" || got["pid"] != float64(42) {
		t.Errorf("unexpected: %s", b)
	}
}
