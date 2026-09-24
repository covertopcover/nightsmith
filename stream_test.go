package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// sseServer writes raw SSE frames, so these tests exercise the wire format
// the pinned server actually produces rather than a Go client's idea of it.
// write is given a flusher because a stream nobody flushes is just a slow
// body, and the incremental-callback test would pass on one.
func sseServer(t *testing.T, write func(w http.ResponseWriter, flush func())) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		flush := func() {
			if f != nil {
				f.Flush()
			}
		}
		flush()
		write(w, flush)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func delta(text string) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"delta": map[string]string{"content": text}, "finish_reason": nil}},
	})
	return "data: " + string(b) + "\n\n"
}

func finalFrame(reason string) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"delta": map[string]any{}, "finish_reason": reason}},
	})
	return "data: " + string(b) + "\n\n"
}

func usageFrame(prompt, completion, cached int) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{},
		"usage": map[string]any{
			"prompt_tokens": prompt, "completion_tokens": completion,
			"prompt_tokens_details": map[string]int{"cached_tokens": cached},
		},
	})
	return "data: " + string(b) + "\n\n"
}

func stream(t *testing.T, srv *httptest.Server, onDelta func(string)) (StreamResult, error) {
	t.Helper()
	return StreamChat(context.Background(), portOf(t, srv.URL), Config{}, Model{},
		Turn{Messages: []chatMessage{{Role: "user", Content: "hello"}}}, onDelta, nil)
}

// The most important test in the file. The pinned server writes its error
// only while no headers have gone out; once a stream has started, a crash
// sends nothing and the connection simply ends on a 200. Trusting EOF here
// reports half an answer as a whole one.
func TestStreamCutShortIsAFailure(t *testing.T) {
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		fmt.Fprint(w, delta("The harbour is "))
		fmt.Fprint(w, delta("quiet"))
		flush() // …and then the server dies: no finish_reason, no [DONE]
	})

	res, err := stream(t, srv, nil)
	if !errors.Is(err, errStreamTruncated) {
		t.Fatalf("a stream that ends without [DONE] or a finish_reason must fail, got err=%v", err)
	}
	// What did arrive is still reported: it is on the user's screen either way.
	if res.Answer != "The harbour is quiet" {
		t.Errorf("the partial answer must survive the failure, got %q", res.Answer)
	}
}

// A finish_reason is an ending even if the connection drops before [DONE].
func TestFinishReasonWithoutDoneIsAnEnding(t *testing.T) {
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		fmt.Fprint(w, delta("Ready."))
		fmt.Fprint(w, finalFrame("stop"))
		flush()
	})
	if _, err := stream(t, srv, nil); err != nil {
		t.Fatalf("a finish_reason ends a stream, got %v", err)
	}
}

// Probe and StreamChat must not drift apart on what counts as an answer. Two
// implementations of this judgement is how one of them starts calling a
// failure a success.
func TestVerdictIsIdenticalAcrossBothClients(t *testing.T) {
	cases := []struct{ name, content, reasoning string }{
		{"empty", "", ""},
		{"thinking only", "", "* weighing it up"},
		{"whitespace only", "   \n ", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			oneShot := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				json.NewEncoder(w).Encode(map[string]any{
					"choices": []map[string]any{{"message": map[string]string{
						"content": c.content, "reasoning": c.reasoning}}},
				})
			}))
			defer oneShot.Close()
			_, probeErr := Probe(portOf(t, oneShot.URL), Config{}, Model{}, "hello")

			streamed := sseServer(t, func(w http.ResponseWriter, flush func()) {
				b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{
					"delta":         map[string]string{"content": c.content, "reasoning": c.reasoning},
					"finish_reason": "stop"}}})
				fmt.Fprint(w, "data: "+string(b)+"\n\n", "data: [DONE]\n\n")
				flush()
			})
			_, streamErr := stream(t, streamed, nil)

			if probeErr == nil || streamErr == nil {
				t.Fatalf("both must fail: probe=%v stream=%v", probeErr, streamErr)
			}
			if probeErr.Error() != streamErr.Error() {
				t.Errorf("the two clients disagree:\n  probe:  %v\n  stream: %v", probeErr, streamErr)
			}
		})
	}
}

// A test that only checked the assembled string would pass on a client that
// never streamed at all.
func TestTextArrivesIncrementally(t *testing.T) {
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		for _, word := range []string{"A ", "spring ", "tide "} {
			fmt.Fprint(w, delta(word))
			flush()
		}
		fmt.Fprint(w, finalFrame("stop"), "data: [DONE]\n\n")
		flush()
	})

	var calls []string
	res, err := stream(t, srv, func(s string) { calls = append(calls, s) })
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 3 {
		t.Errorf("want one callback per chunk, got %d: %q", len(calls), calls)
	}
	if res.Answer != "A spring tide" {
		t.Errorf("got %q", res.Answer)
	}
}

// The usage frame carries an empty choices array, so anything that indexes
// choices[0] panics on it. It is also the only source of an exact token count
// and of cached_tokens.
func TestUsageFrameIsReadAndDoesNotPanic(t *testing.T) {
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		fmt.Fprint(w, delta("Ready."), finalFrame("stop"), usageFrame(1200, 312, 1100), "data: [DONE]\n\n")
		flush()
	})
	res, err := stream(t, srv, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !res.UsageKnown || res.TokensOut != 312 || res.PromptTokens != 1200 {
		t.Errorf("usage not read: %+v", res)
	}
	if res.CachedTokens != 1100 {
		t.Errorf("cached_tokens is the only measured answer to whether a long conversation stays cheap, got %d", res.CachedTokens)
	}
}

// Hitting the token ceiling is a complete stream with a short answer, not a
// broken one. Confusing the two would send a user hunting for a crash.
func TestLengthIsAnEndingNotAFailure(t *testing.T) {
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		fmt.Fprint(w, delta("It begins"), finalFrame("length"), "data: [DONE]\n\n")
		flush()
	})
	res, err := stream(t, srv, nil)
	if err != nil {
		t.Fatalf("finish_reason=length is not an error, got %v", err)
	}
	if res.FinishReason != "length" {
		t.Errorf("the caller needs the reason to say so, got %q", res.FinishReason)
	}
}

// Ctrl-C is not a failure. The user did it on purpose, and what arrived stays.
func TestCancellationKeepsWhatArrived(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		fmt.Fprint(w, delta("Half an "))
		flush()
		time.Sleep(2 * time.Second) // long enough that the cancel lands first
		fmt.Fprint(w, "data: [DONE]\n\n")
	})

	go func() {
		time.Sleep(150 * time.Millisecond)
		cancel()
	}()
	res, err := StreamChat(ctx, portOf(t, srv.URL), Config{}, Model{},
		Turn{Messages: []chatMessage{{Role: "user", Content: "hi"}}}, nil, nil)

	if err != nil {
		t.Fatalf("an interrupt is not an error, got %v", err)
	}
	if !res.Interrupted {
		t.Error("the caller must be able to tell an interrupt from an ending")
	}
	if res.Answer != "Half an" {
		t.Errorf("what arrived before the interrupt must survive, got %q", res.Answer)
	}
}

// Every refusal happens before a byte of stream, so it arrives as an ordinary
// JSON body — including the 503 sent while the server is draining, whose
// message is already the right thing to show a user.
func TestRefusalBeforeHeadersIsPassedThrough(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "10")
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, `{"error":{"message":"the server is stopping; retry once it is started again"}}`)
	}))
	defer srv.Close()

	_, err := stream(t, srv, nil)
	if err == nil || !strings.Contains(err.Error(), "the server is stopping") {
		t.Fatalf("the server's own message must reach the user, got %v", err)
	}
}

// One long answer in a single frame would otherwise trip the scanner's
// default 64 KB line cap, and the failure would look exactly like a truncated
// stream — a bug that only ever appears in production.
func TestOversizedFrameIsNotMistakenForTruncation(t *testing.T) {
	long := strings.Repeat("a very long answer indeed. ", 4000) // ~108 KB
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		fmt.Fprint(w, delta(long), finalFrame("stop"), "data: [DONE]\n\n")
		flush()
	})
	res, err := stream(t, srv, nil)
	if err != nil {
		t.Fatalf("a frame over 64 KB must parse, got %v", err)
	}
	if len(res.Answer) < len(long)-1 {
		t.Errorf("answer truncated: got %d bytes, want %d", len(res.Answer), len(long))
	}
}

// Wire quirks that are legal SSE and would otherwise be read as content or as
// a missing ending.
func TestWireQuirks(t *testing.T) {
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		fmt.Fprint(w, ": keep-alive\n\n")           // a comment, not data
		fmt.Fprint(w, "event: message\n")           // a field we don't use
		fmt.Fprint(w, strings.TrimSuffix(delta("Tides "), "\n\n")+"\r\n\r\n") // CRLF
		b, _ := json.Marshal(map[string]any{"choices": []map[string]any{
			{"delta": map[string]string{"content": "turn."}, "finish_reason": nil}}})
		fmt.Fprint(w, "data:"+string(b)+"\n\n") // no space after the colon
		fmt.Fprint(w, finalFrame("stop"), "data: [DONE]\n\n")
		flush()
	})
	res, err := stream(t, srv, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Answer != "Tides turn." {
		t.Errorf("got %q, want %q", res.Answer, "Tides turn.")
	}
}

// Prefill is not decode. Dividing the whole elapsed time by the token count
// is how setup once reported a 9 word/s model as a 5 word/s one.
func TestSpeedExcludesTimeToFirstToken(t *testing.T) {
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		time.Sleep(300 * time.Millisecond) // prefill
		for i := 0; i < 50; i++ {
			fmt.Fprint(w, delta("word "))
			flush()
		}
		fmt.Fprint(w, finalFrame("stop"), usageFrame(10, 50, 0), "data: [DONE]\n\n")
		flush()
	})
	res, err := stream(t, srv, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.FirstToken < 250*time.Millisecond {
		t.Errorf("prefill should have been measured, got %s", res.FirstToken)
	}
	whole := float64(res.TokensOut) / res.Elapsed.Seconds()
	if res.TokPerSec() <= whole {
		t.Errorf("decode rate %.1f must beat the whole-elapsed rate %.1f", res.TokPerSec(), whole)
	}
	if !res.SpeedWorthPrinting() {
		t.Error("50 tokens is enough to report a speed")
	}
}

// A two-word answer divided by a cold start measures latency, not writing
// speed, and printing it as words-a-second is a claim that wasn't measured.
func TestShortAnswerReportsNoSpeed(t *testing.T) {
	srv := sseServer(t, func(w http.ResponseWriter, flush func()) {
		fmt.Fprint(w, delta("Ready."), finalFrame("stop"), usageFrame(10, 2, 0), "data: [DONE]\n\n")
		flush()
	})
	res, err := stream(t, srv, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.SpeedWorthPrinting() {
		t.Error("a 2-token answer must not produce a words-a-second claim")
	}
}
