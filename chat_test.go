package main

import (
	"bufio"
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
)

// The combination is the useful case, not an edge one:
//
//	cat notes.txt | nightsmith -p "summarise this"
//
// puts the instruction first and the material after it, which is the order a
// model needs them in.
func TestQuestionComesFromArgvStdinOrBoth(t *testing.T) {
	cases := []struct {
		name  string
		args  []string
		piped string
		want  string
	}{
		{"argv alone", []string{"what's", "a", "spring", "tide?"}, "",
			"what's a spring tide?"},
		{"no question, so stdin is the question", nil, "summarise this please",
			"summarise this please"},
		{"a lone dash joins the two", []string{"summarise this", "-"}, "the harbour was quiet",
			"summarise this\n\nthe harbour was quiet"},
		{"a dash alone", []string{"-"}, "from a file", "from a file"},
		{"nothing at all", nil, "", ""},
		{"an empty pipe leaves argv alone", []string{"hello", "-"}, "", "hello"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := buildQuestion(c.args, strings.NewReader(c.piped))
			if err != nil {
				t.Fatal(err)
			}
			if got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

// The bug this pins down: reading stdin because it merely isn't a terminal
// hangs forever wherever stdin is a pipe nobody closes — a CI runner, a cron
// job, an agent's shell. A question in argv must never touch stdin at all.
func TestAQuestionInArgvNeverReadsStdin(t *testing.T) {
	got, err := buildQuestion([]string{"hello"}, readerThatFailsIfUsed{t})
	if err != nil || got != "hello" {
		t.Fatalf("got %q, %v", got, err)
	}
}

type readerThatFailsIfUsed struct{ t *testing.T }

func (r readerThatFailsIfUsed) Read([]byte) (int, error) {
	r.t.Fatal("stdin was read when argv already carried the question")
	return 0, nil
}

// The footer may only state what was measured. A two-token answer divided by
// a cold start is a latency figure wearing a throughput label.
func TestFooterOmitsSpeedItCannotJustify(t *testing.T) {
	short := StreamResult{TokensOut: 2, UsageKnown: true, Elapsed: 2e9, FirstToken: 1e9}
	if strings.Contains(turnFooter(short, Config{}), "word") {
		t.Errorf("a 2-token answer must not carry a speed: %q", turnFooter(short, Config{}))
	}

	long := StreamResult{TokensOut: 300, UsageKnown: true, Elapsed: 26e9, FirstToken: 1e9}
	if !strings.Contains(turnFooter(long, Config{}), "words a second") {
		t.Errorf("a 300-token answer should report a speed: %q", turnFooter(long, Config{}))
	}
}

// Hitting the ceiling is a complete answer that stopped early, and the user
// can do something about it — so say which setting, and where it lives.
func TestFooterExplainsTheTokenCeiling(t *testing.T) {
	res := StreamResult{TokensOut: 1500, UsageKnown: true, Elapsed: 120e9,
		FirstToken: 1e9, FinishReason: "length"}
	got := turnFooter(res, Config{MaxTokens: 1500})
	if !strings.Contains(got, "max_tokens") || !strings.Contains(got, "1500") {
		t.Errorf("the ceiling and its setting must both be named: %q", got)
	}
}

// ── The conversation ────────────────────────────────────────────────────────

// replay drives the whole REPL headless: no server, no model, no terminal.
func replay(t *testing.T, typed string, ask asker) string {
	t.Helper()
	var out strings.Builder
	if err := runChat(bufio.NewReader(strings.NewReader(typed)), &out, "a-model", ask, nil, nil, WindowPolicy{}); err != nil {
		t.Fatalf("runChat: %v", err)
	}
	return out.String()
}

// echoing answers with the turn count, so a test can see what the model was
// actually sent rather than only what came back.
func echoing(seen *[][]chatMessage) asker {
	return func(ctx context.Context, msgs []chatMessage, onDelta func(string)) (StreamResult, error) {
		*seen = append(*seen, append([]chatMessage(nil), msgs...))
		answer := fmt.Sprintf("answer %d", len(*seen))
		onDelta(answer)
		return StreamResult{Answer: answer, UsageKnown: true, TokensOut: 60,
			Elapsed: 5e9, FirstToken: 1e9, FinishReason: "stop"}, nil
	}
}

// The whole point of a conversation: the second question carries the first
// question and its answer, or it is not a conversation at all.
func TestEachTurnCarriesTheOnesBeforeIt(t *testing.T) {
	var seen [][]chatMessage
	replay(t, "what's a spring tide?\nand a neap tide?\n", echoing(&seen))

	if len(seen) != 2 {
		t.Fatalf("want 2 turns, got %d", len(seen))
	}
	if len(seen[1]) != 3 {
		t.Fatalf("the second turn must send user, assistant, user — got %d: %+v", len(seen[1]), seen[1])
	}
	want := []chatMessage{
		{Role: "user", Content: "what's a spring tide?"},
		{Role: "assistant", Content: "answer 1"},
		{Role: "user", Content: "and a neap tide?"},
	}
	for i, w := range want {
		if seen[1][i] != w {
			t.Errorf("message %d: got %+v, want %+v", i, seen[1][i], w)
		}
	}
}

func TestNewStartsAFreshConversation(t *testing.T) {
	var seen [][]chatMessage
	replay(t, "first\n/new\nsecond\n", echoing(&seen))

	if len(seen) != 2 {
		t.Fatalf("want 2 turns, got %d", len(seen))
	}
	if len(seen[1]) != 1 {
		t.Errorf("/new must drop what came before, got %+v", seen[1])
	}
}

// A typo'd command silently becoming a question is worse than being told.
func TestAnUnknownCommandIsNotSentToTheModel(t *testing.T) {
	var seen [][]chatMessage
	out := replay(t, "/halp\n", echoing(&seen))

	if len(seen) != 0 {
		t.Errorf("an unknown command must not reach the model, got %+v", seen)
	}
	if !strings.Contains(out, "/halp isn't a command") {
		t.Errorf("the user must be told, got:\n%s", out)
	}
}

// …but a leading space is the escape hatch, so a line that really does start
// with a slash can still be sent.
func TestALeadingSpaceSendsASlashLine(t *testing.T) {
	var seen [][]chatMessage
	replay(t, " /etc/hosts — what is it?\n", echoing(&seen))

	if len(seen) != 1 || seen[0][0].Content != "/etc/hosts — what is it?" {
		t.Fatalf("got %+v", seen)
	}
}

func TestBackslashJoinsLines(t *testing.T) {
	var seen [][]chatMessage
	replay(t, "first line \\\nsecond line\n", echoing(&seen))

	if len(seen) != 1 {
		t.Fatalf("a continued line is one message, got %d", len(seen))
	}
	if seen[0][0].Content != "first line \nsecond line" {
		t.Errorf("got %q", seen[0][0].Content)
	}
}

// A cooked-mode terminal delivers a pasted block as separate lines, which
// would otherwise fire one turn per line.
func TestPasteIsOneMessage(t *testing.T) {
	var seen [][]chatMessage
	replay(t, "/paste\nline one\nline two\nline three\n.\n", echoing(&seen))

	if len(seen) != 1 {
		t.Fatalf("a paste is one message, got %d turns", len(seen))
	}
	if seen[0][0].Content != "line one\nline two\nline three" {
		t.Errorf("got %q", seen[0][0].Content)
	}
}

// Ctrl-D at the prompt leaves, and does not look like a crash.
func TestEOFLeavesCleanly(t *testing.T) {
	var seen [][]chatMessage
	replay(t, "a question\n", echoing(&seen)) // no trailing input after it
}

// An interrupted answer is kept: it is on the user's screen either way, and
// dropping it would break the prefix the next turn extends.
func TestInterruptedAnswerStaysInTheConversation(t *testing.T) {
	var seen [][]chatMessage
	turn := 0
	ask := func(ctx context.Context, msgs []chatMessage, onDelta func(string)) (StreamResult, error) {
		seen = append(seen, append([]chatMessage(nil), msgs...))
		turn++
		if turn == 1 {
			onDelta("half an ans")
			return StreamResult{Answer: "half an ans", Interrupted: true}, nil
		}
		return StreamResult{Answer: "ok", UsageKnown: true, TokensOut: 60, Elapsed: 5e9, FirstToken: 1e9}, nil
	}
	out := replay(t, "one\ntwo\n", ask)

	if !strings.Contains(out, "Stopped") {
		t.Errorf("the user must be told it stopped, got:\n%s", out)
	}
	if len(seen[1]) != 3 || seen[1][1].Content != "half an ans" {
		t.Errorf("the partial answer must stay in the conversation, got %+v", seen[1])
	}
}

// A turn that failed did not happen, so the question must not be left behind
// to be re-sent as context on the next one.
func TestAFailedTurnLeavesNothingBehind(t *testing.T) {
	var seen [][]chatMessage
	turn := 0
	ask := func(ctx context.Context, msgs []chatMessage, onDelta func(string)) (StreamResult, error) {
		seen = append(seen, append([]chatMessage(nil), msgs...))
		turn++
		if turn == 1 {
			return StreamResult{}, errStreamTruncated
		}
		return StreamResult{Answer: "ok", UsageKnown: true, TokensOut: 60, Elapsed: 5e9, FirstToken: 1e9}, nil
	}
	replay(t, "one\ntwo\n", ask)

	if len(seen[1]) != 1 {
		t.Errorf("a failed turn must not linger as context, got %+v", seen[1])
	}
}

// The margin is applied to text that arrives a few characters at a time and
// knows nothing about lines.
func TestStreamedTextGetsTheHouseMargin(t *testing.T) {
	var out strings.Builder
	i := &indenter{w: &out, prefix: "  "}
	for _, chunk := range []string{"first", " line\nsec", "ond line"} {
		i.WriteString(chunk)
	}
	i.finish()

	if out.String() != "  first line\n  second line\n" {
		t.Errorf("got %q", out.String())
	}
}

// The conversation reaches the disk as it happens, so quitting is not losing.
func TestTheConversationIsWrittenDownAsItHappens(t *testing.T) {
	isolatedHome(t)

	s, err := NewSession("a-model")
	if err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	rec := &sessionRecorder{model: "a-model", out: &out, cur: s}
	var seen [][]chatMessage
	if err := runChat(bufio.NewReader(strings.NewReader("first\nsecond\n")), &out,
		"a-model", echoing(&seen), rec, nil, WindowPolicy{}); err != nil {
		t.Fatal(err)
	}

	back, err := readSession(s.Path)
	if err != nil {
		t.Fatal(err)
	}
	conv := back.Conversation()
	if len(conv) != 4 {
		t.Fatalf("want two exchanges on disk, got %d: %+v", len(conv), conv)
	}
	if conv[0].Content != "first" || conv[1].Content != "answer 1" || conv[2].Content != "second" {
		t.Errorf("wrong order on disk: %+v", conv)
	}
}

// A turn that failed did not happen. Half an exchange on disk would come back
// tomorrow as context for a question that was never answered.
func TestAFailedTurnIsNotWrittenDown(t *testing.T) {
	isolatedHome(t)

	s, _ := NewSession("a-model")
	var out strings.Builder
	rec := &sessionRecorder{model: "a-model", out: &out, cur: s}
	ask := func(ctx context.Context, msgs []chatMessage, onDelta func(string)) (StreamResult, error) {
		return StreamResult{}, errStreamTruncated
	}
	if err := runChat(bufio.NewReader(strings.NewReader("a question\n")), &out, "a-model", ask, rec, nil, WindowPolicy{}); err != nil {
		t.Fatal(err)
	}

	back, _ := readSession(s.Path)
	if len(back.Messages) != 0 {
		t.Errorf("nothing should have been recorded, got %+v", back.Messages)
	}
}

// Resuming drops you into a conversation you have to recognise, so the last
// exchange is printed before the prompt.
func TestResumingShowsWhatYouWereTalkingAbout(t *testing.T) {
	var seen [][]chatMessage
	var out strings.Builder
	history := []chatMessage{
		{Role: "user", Content: "what's a spring tide?"},
		{Role: "assistant", Content: "The bigger-than-usual one."},
	}
	if err := runChat(bufio.NewReader(strings.NewReader("and a neap tide?\n")), &out,
		"a-model", echoing(&seen), nil, history, WindowPolicy{}); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(out.String(), "Picking up where you left off") {
		t.Errorf("a resumed conversation should say so:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "what's a spring tide?") {
		t.Errorf("the recap should show what was said:\n%s", out.String())
	}
	if len(seen[0]) != 3 {
		t.Errorf("the resumed history must be sent with the new question, got %+v", seen[0])
	}
}

// Recording must never be the reason a conversation ends.
func TestADiskThatRefusesDoesNotEndTheConversation(t *testing.T) {
	isolatedHome(t)

	var out strings.Builder
	broken := &sessionRecorder{model: "a-model", out: &out,
		cur: &Session{ID: "x", Path: filepath.Join(t.TempDir(), "no-such-dir", "x.jsonl")}}
	var seen [][]chatMessage
	if err := runChat(bufio.NewReader(strings.NewReader("one\ntwo\n")), &out,
		"a-model", echoing(&seen), broken, nil, WindowPolicy{}); err != nil {
		t.Fatalf("a disk problem must not end the conversation: %v", err)
	}
	if len(seen) != 2 {
		t.Errorf("both turns should still have been answered, got %d", len(seen))
	}
	if !strings.Contains(out.String(), "won't be here tomorrow") {
		t.Errorf("the user should be told once:\n%s", out.String())
	}
	if strings.Count(out.String(), "won't be here tomorrow") != 1 {
		t.Error("…and only once")
	}
}
