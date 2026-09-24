package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// The most important code in the product.
//
// Observed while benchmarking mlx_lm.server: /v1/models returned 200 while the
// server was unable to serve a single request — in three different ways
// (unsupported model, unsupported draft model, post-OOM zombie).
//
//	A checkmark may only appear after a real completion came back.
//
// So there is no health-check function in this file, and there must never be
// one. The server's /health (serve.py) exists for clients, as a hint; nothing
// here reads it. Anything less than a real answer is a health check that lies, and that
// is the whole difference between a tool that works and a tool that reports
// that it works.

type ProbeResult struct {
	Answer        string
	TokensOut     int
	Elapsed       time.Duration
	TokPerSec     float64
	ThinkingOff   bool // the setting was asked for AND the output shows it took
	ThinkingKnown bool
}

type chatRequest struct {
	Model              string         `json:"model,omitempty"`
	Messages           []chatMessage  `json:"messages"`
	Stream             bool           `json:"stream,omitempty"`
	StreamOptions      *streamOptions `json:"stream_options,omitempty"`
	MaxTokens          int            `json:"max_tokens"`
	Temperature        float64        `json:"temperature"`
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

// streamOptions is only ever sent as {"include_usage": true}. The pinned
// server reads include_usage with a bare dict index, so a stream_options
// without that key raises after the headers are out — which reaches a client
// as a stream that simply stops. A pointer, so it is omitted entirely rather
// than sent empty.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning"` // mlx_lm splits thinking out here
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// ThinkingKwargs renders the per-model thinking setting. This is the table
// earning its place: the key name is not standard, and whether omitting it
// means "off" differs per model. Gemma 4 requires the key to be both defined
// and truthy, so omitting it is already off. MiniCPM5 only emits a thinking
// branch when the key is defined at all, so "off" must be sent explicitly.
func ThinkingKwargs(m Model, want bool) map[string]any {
	if m.Thinking.Key == "" {
		return nil
	}
	if !m.Thinking.CanDisable && !want {
		// DeepSeek-R1 and its distills. Refuse to pretend: the caller is told
		// via ThinkingHonest, and nothing is sent that would imply control we
		// do not have.
		return nil
	}
	// Always explicit. Omitting was once "correct" for Gemma 4 on the strength
	// of its template, but mlx_lm fills in thinking=on for any model that can
	// think when the key is absent — see models.toml. Omission is never off.
	return map[string]any{m.Thinking.Key: want}
}

// ThinkingHonest says whether the requested setting can actually be honoured.
func ThinkingHonest(m Model, want bool) (ok bool, why string) {
	if !want {
		if !m.Thinking.CanDisable {
			return false, fmt.Sprintf("%s cannot turn thinking off — its architecture always reasons.",
				shortRepo(m.Repo))
		}
	}
	if m.Thinking.Status == "estimate" {
		return true, fmt.Sprintf("%s: the thinking setting has not been verified against this model's template. The probe below checks it took effect.",
			shortRepo(m.Repo))
	}
	return true, ""
}

// answerVerdict is the single place that decides whether a completion counts.
// Both clients come through here — the one-shot Probe below and the streaming
// one in stream.go — so the post-OOM zombie and the answer spent entirely on
// reasoning cannot be reported two different ways. A second implementation of
// this judgement is how one of them would quietly start calling a failure a
// success.
func answerVerdict(content, reasoning string) error {
	if strings.TrimSpace(content) != "" {
		return nil
	}
	if strings.TrimSpace(reasoning) != "" {
		// The whole budget went on reasoning. Say that, because "empty" would
		// send the user looking for an out-of-memory problem that isn't there.
		return fmt.Errorf("the model spent its whole answer thinking and never replied")
	}
	// This is the post-OOM zombie, and a health check would have called it
	// healthy. An empty answer is a failure.
	return fmt.Errorf("the model server replied, but the answer was empty")
}

// Probe asks the model a real question and reads the real answer. Everything
// the tool claims afterwards rests on this returning.
func Probe(port int, c Config, m Model, question string) (ProbeResult, error) {
	var r ProbeResult

	// No model field: the server then answers with the model it was started
	// with. The repo id is accepted too (serve.py maps it), but leaving it out
	// cannot be misspelled.
	body, err := json.Marshal(chatRequest{
		Messages:           []chatMessage{{Role: "user", Content: question}},
		MaxTokens:          c.MaxTokens,
		Temperature:        c.Temperature,
		ChatTemplateKwargs: ThinkingKwargs(m, c.Thinking),
	})
	if err != nil {
		return r, err
	}

	start := time.Now()
	// Generous, because a cold start on a 16 GB Mac loads 6.8 GB of weights
	// first. The failure this guards against is the one that matters: a server
	// that answered /v1/models with 200 and then never completes a request.
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port),
		"application/json", bytes.NewReader(body))
	if err != nil {
		return r, fmt.Errorf("the model server didn't answer: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return r, err
	}
	if resp.StatusCode != http.StatusOK {
		return r, fmt.Errorf("the model server returned %d: %s",
			resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		return r, fmt.Errorf("the model server's reply wasn't valid JSON: %w", err)
	}
	var content, reasoning string
	if len(cr.Choices) > 0 {
		content, reasoning = cr.Choices[0].Message.Content, cr.Choices[0].Message.Reasoning
	}
	if err := answerVerdict(content, reasoning); err != nil {
		return r, err
	}

	r.Answer = strings.TrimSpace(cr.Choices[0].Message.Content)
	r.TokensOut = cr.Usage.CompletionTokens
	r.Elapsed = time.Since(start)
	if r.Elapsed.Seconds() > 0 && r.TokensOut > 0 {
		r.TokPerSec = float64(r.TokensOut) / r.Elapsed.Seconds()
	}

	// Settings that can lie are verified by reading the output, not by
	// trusting the flag. A model that ignores the thinking setting does so
	// silently, and the user should learn that at setup rather than weeks later
	// when a queue is mysteriously slow.
	if m.Thinking.Key != "" && m.Thinking.CanDisable {
		r.ThinkingKnown = true
		// Both places: mlx_lm moves reasoning out of the answer into its own
		// field, so markers in the text alone once printed "Thinking stayed
		// off" for a model that had thought for 1,500 tokens.
		r.ThinkingOff = !ShowsThinking(r.Answer) &&
			strings.TrimSpace(cr.Choices[0].Message.Reasoning) == ""
	}
	return r, nil
}

// ShowsThinking looks for reasoning that leaked into the answer. Both marker
// styles in the table appear here: Gemma 4's <|think|> and the <think> tags
// used by Qwen-lineage and MiniCPM templates.
func ShowsThinking(answer string) bool {
	for _, marker := range []string{"<think>", "</think>", "<|think|>", "<|/think|>"} {
		if strings.Contains(answer, marker) {
			return true
		}
	}
	return false
}
