package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// The streaming client. It is separate from probe.go because that file's
// header is a rule about what may never live in it, and a second HTTP client
// sitting beside Probe would blur it. What the two must share is the
// judgement, not the transport: both call answerVerdict, so an empty answer
// and an answer spent entirely on reasoning mean the same thing here as they
// do there.
//
// The failure this file exists to catch:
//
//	A stream that stops early is a well-formed HTTP response.
//
// The pinned server writes its error only if no headers have gone out yet.
// Once a stream has started, a crash mid-generation sends nothing at all —
// the connection just ends, cleanly, on a 200. A reader that trusts EOF
// reports half an answer as a whole one. So EOF is not an ending here:
// `data: [DONE]`, or a chunk carrying a finish_reason, is the only proof a
// stream finished.

const (
	// No total timeout: a conversation can legitimately run longer than any
	// number picked here, and Probe's five minutes would cut one off. The two
	// bounds below replace it.
	streamHeaderTimeout = 2 * time.Minute // covers a ~45 s cold load with room
	streamIdleTimeout   = 60 * time.Second

	// Below this, a speed figure measures latency rather than writing speed.
	// Setup learned this the expensive way: a two-token answer divided by a
	// cold start once printed "5 words a second" for a model that writes 9.
	minTokensForSpeed = 40
)

// errStreamTruncated is the ending that isn't one. Callers map it to the
// "running but not answering" exit code, because that is what it is.
var errStreamTruncated = errors.New("the model server stopped mid-answer")

type Turn struct {
	Messages    []chatMessage
	MaxTokens   int
	Temperature float64
	Kwargs      map[string]any // ThinkingKwargs(m, c.Thinking)
}

type StreamResult struct {
	Answer    string
	Reasoning string

	// TokensOut is exact — it comes from the server's own usage chunk, not
	// from counting chunks. A chunk is not reliably a token.
	TokensOut    int
	PromptTokens int
	// CachedTokens is how much of the prompt the server's prefix cache
	// reused. It is the only measured answer to "does a long conversation
	// stay cheap", and it comes free with the usage chunk.
	CachedTokens int
	UsageKnown   bool

	FirstToken   time.Duration // prefill: how long before anything came back
	Elapsed      time.Duration
	FinishReason string // "stop" | "length" | "tool_calls"
	Interrupted  bool
}

// TokPerSec measures decode alone: the tokens after the first, over the time
// after the first arrived. Including prefill in the divisor is how you report
// a 9 word/s model as a 5 word/s one.
func (r StreamResult) TokPerSec() float64 {
	decode := r.Elapsed - r.FirstToken
	if !r.UsageKnown || r.TokensOut < 2 || decode <= 0 {
		return 0
	}
	return float64(r.TokensOut-1) / decode.Seconds()
}

// SpeedWorthPrinting keeps an unrepresentative number off the screen rather
// than printing it with a caveat.
func (r StreamResult) SpeedWorthPrinting() bool {
	return r.UsageKnown && r.TokensOut >= minTokensForSpeed && r.TokPerSec() > 0
}

// chatChunk is one SSE frame. Every field is optional: the usage frame
// carries an empty choices array, and an ordinary frame carries no usage — so
// nothing here may be indexed without checking first.
type chatChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

// StreamChat sends one turn and calls onDelta as text arrives. It makes the
// same promises Probe does, and adds one: it returns an error rather than a
// short answer when the stream ends without saying it finished.
//
// A cancelled ctx is not an error. The user stopped it on purpose, and what
// arrived before they did is returned with Interrupted set.
func StreamChat(ctx context.Context, port int, c Config, m Model, t Turn,
	onDelta, onReasoning func(string)) (StreamResult, error) {
	var r StreamResult

	body, err := json.Marshal(chatRequest{
		Messages: t.Messages,
		Stream:   true,
		// Always with include_usage, never bare: the pinned server reads the
		// key with a plain index, so a stream_options without it raises after
		// the headers are out — arriving here as a stream that just stops.
		StreamOptions:      &streamOptions{IncludeUsage: true},
		MaxTokens:          t.MaxTokens,
		Temperature:        t.Temperature,
		ChatTemplateKwargs: t.Kwargs,
	})
	if err != nil {
		return r, err
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port),
		bytes.NewReader(body))
	if err != nil {
		return r, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	client := &http.Client{
		Transport: &http.Transport{ResponseHeaderTimeout: streamHeaderTimeout},
	}

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			r.Interrupted, r.Elapsed = true, time.Since(start)
			return r, nil
		}
		return r, fmt.Errorf("the model server didn't answer: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Every refusal happens before a byte of stream is written, so this is
		// a normal JSON body — including the 503 the server sends while it is
		// draining, whose message is already the right thing to show.
		raw, _ := io.ReadAll(resp.Body)
		return r, fmt.Errorf("the model server returned %d: %s",
			resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	// A server that stops writing without closing would otherwise hang here
	// forever. Nothing in the protocol distinguishes "still thinking" from
	// "gone", so silence past the deadline is treated as gone.
	var lastByte atomic.Int64
	lastByte.Store(time.Now().UnixNano())
	var idleGaveUp atomic.Bool
	watchdog := time.NewTicker(time.Second)
	defer watchdog.Stop()
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-watchdog.C:
				if time.Since(time.Unix(0, lastByte.Load())) > streamIdleTimeout {
					idleGaveUp.Store(true)
					cancel()
					return
				}
			}
		}
	}()

	var content, reasoning strings.Builder
	var sawDone, sawFinish bool

	scanner := bufio.NewScanner(resp.Body)
	// The default 64 KB line cap would fail on one oversized frame, and the
	// failure would look exactly like a truncated stream.
	scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)

	for scanner.Scan() {
		lastByte.Store(time.Now().UnixNano())
		line := strings.TrimRight(scanner.Text(), "\r")
		if line == "" || strings.HasPrefix(line, ":") {
			continue // frame separator, or a heartbeat comment
		}
		payload, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue // an SSE field we don't use (event:, id:, retry:)
		}
		payload = strings.TrimSpace(payload)
		if payload == "[DONE]" {
			sawDone = true
			break
		}

		var chunk chatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return r, fmt.Errorf("the model server sent a chunk that wasn't valid JSON: %w", err)
		}

		if u := chunk.Usage; u != nil {
			r.PromptTokens, r.TokensOut, r.UsageKnown = u.PromptTokens, u.CompletionTokens, true
			if d := u.PromptTokensDetails; d != nil {
				r.CachedTokens = d.CachedTokens
			}
		}
		for _, choice := range chunk.Choices {
			if d := choice.Delta.Content; d != "" {
				if r.FirstToken == 0 {
					r.FirstToken = time.Since(start)
				}
				content.WriteString(d)
				if onDelta != nil {
					onDelta(d)
				}
			}
			if d := choice.Delta.Reasoning; d != "" {
				if r.FirstToken == 0 {
					r.FirstToken = time.Since(start)
				}
				reasoning.WriteString(d)
				if onReasoning != nil {
					onReasoning(d)
				}
			}
			if choice.FinishReason != nil {
				sawFinish = true
				r.FinishReason = *choice.FinishReason
			}
		}
	}

	r.Elapsed = time.Since(start)
	r.Answer = strings.TrimSpace(content.String())
	r.Reasoning = strings.TrimSpace(reasoning.String())

	// Ctrl-C. Not an error, and what arrived is worth keeping: it is on the
	// user's screen either way.
	if ctx.Err() != nil && !idleGaveUp.Load() {
		r.Interrupted = true
		return r, nil
	}

	if err := scanner.Err(); err != nil && !idleGaveUp.Load() {
		return r, fmt.Errorf("%w: %v", errStreamTruncated, err)
	}
	if idleGaveUp.Load() {
		return r, fmt.Errorf("%w: nothing arrived for %s", errStreamTruncated, streamIdleTimeout)
	}
	if !sawDone && !sawFinish {
		// A clean EOF on a 200, with no ending. This is the case the whole
		// file is built around: the server died and could not say so.
		return r, errStreamTruncated
	}

	return r, answerVerdict(r.Answer, r.Reasoning)
}
