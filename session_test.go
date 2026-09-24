package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func isolatedHome(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
}

func TestASessionSurvivesBeingWrittenAndRead(t *testing.T) {
	isolatedHome(t)

	s, err := NewSession("mlx-community/gemma-4-12B-it-4bit")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Append(SessionMessage{Role: "user", Content: "what's a spring tide?"}); err != nil {
		t.Fatal(err)
	}
	if err := s.Append(SessionMessage{Role: "assistant", Content: "The bigger one.", Tokens: 42}); err != nil {
		t.Fatal(err)
	}

	back, err := readSession(s.Path)
	if err != nil {
		t.Fatal(err)
	}
	if back.ID != s.ID || back.Model != s.Model {
		t.Errorf("header lost: %+v", back)
	}
	conv := back.Conversation()
	if len(conv) != 2 || conv[0].Content != "what's a spring tide?" || conv[1].Role != "assistant" {
		t.Errorf("conversation came back wrong: %+v", conv)
	}
}

// A transcript is written one line at a time so that a crash costs at most
// the turn in progress. A half-written final line must not take the rest of
// the conversation with it.
func TestAHalfWrittenLastLineDoesNotLoseTheRest(t *testing.T) {
	isolatedHome(t)

	s, _ := NewSession("a-model")
	s.Append(SessionMessage{Role: "user", Content: "one"})
	s.Append(SessionMessage{Role: "assistant", Content: "two"})

	f, err := os.OpenFile(s.Path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(`{"v":1,"kind":"message","role":"user","cont`) // killed mid-write
	f.Close()

	back, err := readSession(s.Path)
	if err != nil {
		t.Fatal(err)
	}
	if len(back.Messages) != 2 {
		t.Errorf("want the 2 complete messages, got %d", len(back.Messages))
	}
}

// The promise that makes tool calls cheap to add later: a transcript written
// by a version that knows more than this one must still open. Nothing here
// emits a tool call — this fixture is what stops that from becoming a
// breaking change when something does.
func TestARecordFromTheFutureIsSkippedNotFatal(t *testing.T) {
	isolatedHome(t)

	if err := os.MkdirAll(chatsDir(), 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(chatsDir(), "2026-09-23-120000-beef.jsonl")
	fixture := strings.Join([]string{
		`{"v":1,"kind":"session","id":"beef","created":"2026-09-23T12:00:00Z","model":"a-model"}`,
		`{"v":1,"kind":"message","role":"user","content":"what's the weather?"}`,
		`{"v":2,"kind":"message","role":"assistant","content":"","tool_calls":[{"id":"c1","name":"weather"}]}`,
		`{"v":2,"kind":"message","role":"tool","tool_call_id":"c1","content":"14C, raining"}`,
		`{"v":2,"kind":"ledger","note":"something this version has never heard of"}`,
		`{"v":1,"kind":"message","role":"assistant","content":"Raining, 14 degrees."}`,
		"",
	}, "\n")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	back, err := readSession(path)
	if err != nil {
		t.Fatalf("a transcript from a newer version must still open: %v", err)
	}
	if back.ID != "beef" {
		t.Errorf("header lost: %+v", back)
	}
	// The user turn, the empty tool-calling assistant turn, and the final
	// answer are roles this version knows; the tool result and the unknown
	// kind are not.
	for _, m := range back.Messages {
		if m.Role == "tool" || m.Kind == "ledger" {
			t.Errorf("a record this version does not understand was kept: %+v", m)
		}
	}
	if len(back.Messages) != 3 {
		t.Errorf("want the 3 understood messages, got %d: %+v", len(back.Messages), back.Messages)
	}
}

// Ids resolve like short shas: any unique prefix, and an ambiguous one says
// so rather than picking.
func TestIDsResolveByUniquePrefix(t *testing.T) {
	isolatedHome(t)
	if err := os.MkdirAll(chatsDir(), 0o700); err != nil {
		t.Fatal(err)
	}
	for i, id := range []string{"aa01", "aa02", "bb03"} {
		path := filepath.Join(chatsDir(), "2026-09-2"+string(rune('1'+i))+"-120000-"+id+".jsonl")
		body := `{"v":1,"kind":"session","id":"` + id + `","model":"m"}` + "\n" +
			`{"v":1,"kind":"message","role":"user","content":"hello"}` + "\n"
		os.WriteFile(path, []byte(body), 0o600)
	}

	if s, err := OpenSession("bb"); err != nil || s.ID != "bb03" {
		t.Errorf("a unique prefix should resolve: %v", err)
	}
	if s, err := OpenSession("aa01"); err != nil || s.ID != "aa01" {
		t.Errorf("a full id should resolve: %v", err)
	}
	_, err := OpenSession("aa")
	if err == nil || !strings.Contains(err.Error(), "aa01") || !strings.Contains(err.Error(), "aa02") {
		t.Errorf("an ambiguous prefix must name the candidates, got %v", err)
	}
	if _, err := OpenSession("zz"); err == nil {
		t.Error("a missing id must be an error, not the latest conversation")
	}
}

func TestTheNewestConversationIsTheOneContinued(t *testing.T) {
	isolatedHome(t)

	old, _ := NewSession("a-model")
	old.Append(SessionMessage{Role: "user", Content: "the old one"})
	time.Sleep(1100 * time.Millisecond) // the filename's timestamp is to the second
	recent, _ := NewSession("a-model")
	recent.Append(SessionMessage{Role: "user", Content: "the new one"})

	got, err := LatestSession()
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != recent.ID {
		t.Errorf("got %s, want the newest (%s)", got.ID, recent.ID)
	}
}

// The title is the first thing the user said, taken at read time. Generating
// one would cost a whole completion at 12 tok/s to produce a label.
func TestListingShowsWhatWasAskedAndSkipsEmptyOnes(t *testing.T) {
	isolatedHome(t)

	used, _ := NewSession("a-model")
	used.Append(SessionMessage{Role: "user", Content: "what's a spring tide?\nand a neap one?"})
	used.Append(SessionMessage{Role: "assistant", Content: "…"})
	NewSession("a-model") // opened and never used

	list, err := ListSessions()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 1 {
		t.Fatalf("an empty conversation is not worth offering, got %d", len(list))
	}
	if list[0].Title != "what's a spring tide?" {
		t.Errorf("the title should be the first line of the first question, got %q", list[0].Title)
	}
	if list[0].Turns != 1 {
		t.Errorf("got %d turns", list[0].Turns)
	}
}

// A transcript of what someone said in private is not settings.
func TestTranscriptsAreNotWorldReadable(t *testing.T) {
	isolatedHome(t)

	s, err := NewSession("a-model")
	if err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(s.Path)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode().Perm()&0o077 != 0 {
		t.Errorf("transcript is %v — it should be readable only by its owner", fi.Mode().Perm())
	}
	di, err := os.Stat(chatsDir())
	if err != nil {
		t.Fatal(err)
	}
	if di.Mode().Perm()&0o077 != 0 {
		t.Errorf("chats dir is %v", di.Mode().Perm())
	}
}

// Anything not in PlanRemoval's table is left behind on the user's disk, and
// conversations are the last thing that should quietly survive a `remove`.
func TestRemoveTakesTheConversationsToo(t *testing.T) {
	isolatedHome(t)

	s, _ := NewSession("a-model")
	s.Append(SessionMessage{Role: "user", Content: "hello"})

	var found bool
	for _, item := range PlanRemoval().Items {
		if item.Path == chatsDir() {
			found = true
			if item.What == "" {
				t.Error("conversations should be named in the listing, not removed silently")
			}
		}
	}
	if !found {
		t.Error("remove would leave the conversations behind")
	}
}
