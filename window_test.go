package main

import (
	"strings"
	"testing"
)

func exchange(n int, size int) []chatMessage {
	var msgs []chatMessage
	for i := 0; i < n; i++ {
		msgs = append(msgs,
			chatMessage{Role: "user", Content: strings.Repeat("q", size)},
			chatMessage{Role: "assistant", Content: strings.Repeat("a", size)})
	}
	return msgs
}

func TestAConversationThatFitsIsSentWhole(t *testing.T) {
	msgs := exchange(3, 40)
	got, r := BuildWindow(msgs, WindowPolicy{BudgetTokens: 8000})

	if len(got) != len(msgs) {
		t.Errorf("nothing should have been dropped, got %d of %d", len(got), len(msgs))
	}
	if r.DroppedExchanges != 0 || r.NearLimit {
		t.Errorf("no notice is due: %+v", r)
	}
	if r.Notice() != "" {
		t.Errorf("nothing to say, said %q", r.Notice())
	}
}

// The warning comes while starting fresh is still the user's choice, not our
// decision.
func TestTheUserIsWarnedBeforeAnythingIsDropped(t *testing.T) {
	// 4 chars per token, so 20 exchanges of 80 chars ≈ 800 tokens.
	msgs := exchange(20, 80)
	_, r := BuildWindow(msgs, WindowPolicy{BudgetTokens: 1000})

	if r.DroppedExchanges != 0 {
		t.Fatalf("nothing should be dropped yet: %+v", r)
	}
	if !r.NearLimit {
		t.Errorf("800 of 1000 is past the soft line: %+v", r)
	}
	if !strings.Contains(r.Notice(), "/new") {
		t.Errorf("the warning should offer the way out: %q", r.Notice())
	}
}

// One step, well past the budget. Trimming to exactly the budget would trim
// again next turn, and every trim costs a cold re-read of what is left.
func TestTrimmingGoesWellPastTheBudgetInOneStep(t *testing.T) {
	msgs := exchange(40, 100) // ≈ 2000 tokens
	got, r := BuildWindow(msgs, WindowPolicy{BudgetTokens: 1000})

	if r.DroppedExchanges == 0 {
		t.Fatal("something should have been dropped")
	}
	if r.EstimatedTokens > 700 {
		t.Errorf("one trim should land near 60%% of budget, got %d", r.EstimatedTokens)
	}
	// And the result must now fit, with room — the next turn should not trim.
	_, again := BuildWindow(got, WindowPolicy{BudgetTokens: 1000})
	if again.DroppedExchanges != 0 {
		t.Error("trimming twice in a row means the first trim was too timid")
	}
}

// Whole exchanges, so the conversation never starts with an answer to a
// question that is no longer there.
func TestWholeExchangesGoNotHalfOfOne(t *testing.T) {
	msgs := exchange(40, 100)
	got, _ := BuildWindow(msgs, WindowPolicy{BudgetTokens: 1000})

	if got[0].Role != "user" {
		t.Errorf("a window should open on a question, not an answer: %q", got[0].Role)
	}
}

// A window with nothing in it is not a smaller conversation, it is a
// different one.
func TestTheQuestionJustAskedIsNeverDropped(t *testing.T) {
	huge := []chatMessage{
		{Role: "user", Content: strings.Repeat("x", 100000)},
		{Role: "assistant", Content: strings.Repeat("y", 100000)},
		{Role: "user", Content: "and now this one"},
	}
	got, r := BuildWindow(huge, WindowPolicy{BudgetTokens: 50})

	if len(got) == 0 {
		t.Fatal("the window must never be empty")
	}
	if got[len(got)-1].Content != "and now this one" {
		t.Errorf("the question just asked was dropped: %+v", got)
	}
	if r.DroppedWords == 0 {
		t.Error("the report should say what went")
	}
}

// The notice has to say what it cost, or the next answer's slow start reads
// as a fault.
func TestTheDropNoticeExplainsTheCost(t *testing.T) {
	msgs := exchange(40, 100)
	_, r := BuildWindow(msgs, WindowPolicy{BudgetTokens: 1000})

	notice := r.Notice()
	if !strings.Contains(notice, "read again from scratch") {
		t.Errorf("the notice must explain the slow next turn: %q", notice)
	}
	if !strings.Contains(notice, "words") {
		t.Errorf("say how much went, in words — the one unit we can count: %q", notice)
	}
}

// A budget of zero means no budget, not "drop everything".
func TestNoBudgetMeansNoTrimming(t *testing.T) {
	msgs := exchange(50, 500)
	got, r := BuildWindow(msgs, WindowPolicy{BudgetTokens: 0})

	if len(got) != len(msgs) || r.DroppedExchanges != 0 || r.NearLimit {
		t.Errorf("a zero budget must not trim: %d of %d, %+v", len(got), len(msgs), r)
	}
}

// Trimming is per-request. The transcript on disk stays whole, so a
// conversation trimmed today is still complete when read tomorrow.
func TestTrimmingDoesNotTouchTheCallersSlice(t *testing.T) {
	msgs := exchange(40, 100)
	before := len(msgs)
	first := msgs[0].Content

	BuildWindow(msgs, WindowPolicy{BudgetTokens: 1000})

	if len(msgs) != before || msgs[0].Content != first {
		t.Error("BuildWindow modified what it was given")
	}
}

// The bug this pins down, found by trimming a real conversation: dropping
// from the front can strip every question and leave the window opening on an
// answer — the model reading its own words with nothing to have prompted
// them.
func TestAWindowNeverOpensOnAnOrphanedAnswer(t *testing.T) {
	// Ends on an assistant message, which is what the history looks like
	// between turns — when /context is asked.
	msgs := exchange(3, 400)
	got, _ := BuildWindow(msgs, WindowPolicy{BudgetTokens: 60})

	if len(got) > 1 && got[0].Role == "assistant" {
		t.Errorf("window opens on an orphaned answer: %d messages, first is %q",
			len(got), got[0].Role)
	}
}

// When one exchange is bigger than the whole budget, trimming has run out of
// things it is allowed to drop. That is not a failure, but the report must
// say so rather than print a number plainly over the budget.
func TestOverBudgetIsReportedNotHidden(t *testing.T) {
	msgs := []chatMessage{
		{Role: "user", Content: strings.Repeat("q", 400)},
		{Role: "assistant", Content: strings.Repeat("a", 400)},
		{Role: "user", Content: strings.Repeat("z", 4000)},
	}
	_, r := BuildWindow(msgs, WindowPolicy{BudgetTokens: 100})

	if !r.OverBudget {
		t.Errorf("what is left is over the budget and the report should say so: %+v", r)
	}
	if r.DroppedExchanges == 0 {
		t.Error("it should still have dropped what it could")
	}
}
