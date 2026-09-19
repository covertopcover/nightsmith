package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The trap this table exists to absorb, found on real hardware: Gemma 4's
// template says omitting enable_thinking means off, but mlx_lm fills in
// thinking=on when the key is absent. So "off" is always sent explicitly, for
// every model that can turn it off.
func TestThinkingOffIsAlwaysSentExplicitly(t *testing.T) {
	c, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	gemma, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	mini, _ := c.Find("mlx-community/MiniCPM5-2B-8bit")

	for _, m := range []Model{gemma, mini} {
		for _, want := range []bool{false, true} {
			if kw := ThinkingKwargs(m, want); kw == nil || kw[m.Thinking.Key] != want {
				t.Errorf("%s thinking=%v must send %s=%v explicitly, got %v",
					shortRepo(m.Repo), want, m.Thinking.Key, want, kw)
			}
		}
	}
	// And the server gets it as its default, for clients that send nothing.
	args := strings.Join(ServerArgs(DefaultConfig(c, gemma), gemma, "/snapshot"), " ")
	if !strings.Contains(args, `--chat-template-args {"enable_thinking":false}`) {
		t.Errorf("server must default thinking off, got: %s", args)
	}
}

// Thinking that the server moved into its own field is still thinking.
func TestReasoningFieldCountsAsThinking(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{
				"content": "The harbour is quiet.", "reasoning": "* Topic: a harbour"}}},
			"usage": map[string]int{"completion_tokens": 40},
		})
	}))
	defer srv.Close()
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	res, err := Probe(portOf(t, srv.URL), Config{MaxTokens: 100}, m, "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.ThinkingOff {
		t.Error("a reply with a reasoning field must not be reported as thinking off")
	}
}

func TestAnswerSpentOnThinkingIsNamed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{
				"content": nil, "reasoning": "* Topic: ..."}}},
		})
	}))
	defer srv.Close()
	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	_, err := Probe(portOf(t, srv.URL), Config{MaxTokens: 100}, m, "hi")
	if err == nil || !strings.Contains(err.Error(), "thinking") {
		t.Errorf("an answer consumed by reasoning must say so, got %v", err)
	}
}

// An unverified model must be flagged, not assumed to behave like Gemma 4.
func TestUnverifiedThinkingIsDisclosed(t *testing.T) {
	c, _ := LoadCatalog()
	q, _ := c.Find("mlx-community/Qwen3-14B-4bit")
	ok, why := ThinkingHonest(q, false)
	if !ok || why == "" {
		t.Errorf("an estimate-status model must say so, got ok=%v why=%q", ok, why)
	}
	// And it sends the key explicitly, because assuming omission means off is
	// exactly the mistake MiniCPM5 proves is possible.
	if kw := ThinkingKwargs(q, false); kw == nil {
		t.Error("an unverified model must send the setting explicitly, not assume")
	}
}

func TestShowsThinkingCatchesBothMarkerStyles(t *testing.T) {
	for _, leaked := range []string{
		"<think>\nlet me work through this\n</think>\n\nThe answer is 4.",
		"<|think|> reasoning <|/think|> Done.",
	} {
		if !ShowsThinking(leaked) {
			t.Errorf("reasoning leaked into the answer undetected: %q", leaked)
		}
	}
	if ShowsThinking("Hello. I'm running on your Mac, not in the cloud.") {
		t.Error("a clean answer must not be reported as thinking")
	}
}

// The failure that a health check calls healthy. /v1/models returned 200 while
// the server could serve nothing, in three different ways.
func TestEmptyAnswerIsAFailureNotASuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": "   "}}},
		})
	}))
	defer srv.Close()

	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	_, err := Probe(portOf(t, srv.URL), Config{Model: m.Repo, MaxTokens: 100}, m, "hello")
	if err == nil {
		t.Fatal("an empty answer must be an error — this is the post-OOM zombie")
	}
	if !strings.Contains(err.Error(), "empty") {
		t.Errorf("the error should name the problem, got %q", err)
	}
}

func TestRealAnswerIsMeasured(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		// Thinking off is sent explicitly — omitted, mlx_lm turns it on.
		kw, _ := req["chat_template_kwargs"].(map[string]any)
		if kw["enable_thinking"] != false {
			t.Errorf("Gemma 4 must send enable_thinking=false explicitly: %v", req)
		}
		// No model field: the server was started from a local path, and a
		// repo id here would make it try to load a second model.
		if _, present := req["model"]; present {
			t.Errorf("the probe must not name a model: %v", req)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{
				"content": "Hello. I'm running on your Mac, not in the cloud."}}},
			"usage": map[string]int{"completion_tokens": 12},
		})
	}))
	defer srv.Close()

	c, _ := LoadCatalog()
	m, _ := c.Find("mlx-community/gemma-4-12B-it-4bit")
	got, err := Probe(portOf(t, srv.URL), Config{Model: m.Repo, MaxTokens: 100}, m, "say hello")
	if err != nil {
		t.Fatal(err)
	}
	if got.Answer == "" || got.TokensOut != 12 {
		t.Errorf("probe lost the answer or the count: %+v", got)
	}
	if !got.ThinkingKnown || !got.ThinkingOff {
		t.Error("a clean answer from a model that can disable thinking should report thinking off")
	}
}

func portOf(t *testing.T, url string) int {
	t.Helper()
	i := strings.LastIndex(url, ":")
	var p int
	if _, err := fmtSscan(url[i+1:], &p); err != nil {
		t.Fatalf("could not read port from %q", url)
	}
	return p
}
