package main

import (
	"fmt"
	"strings"
)

// How much of a conversation gets re-read each turn.
//
// Every turn sends the whole conversation again — that is what makes it a
// conversation — so it grows, and something has to give. Three things are
// true here and the third is the one that shapes the policy:
//
//  1. A growing conversation is the *best* case for a one-slot prefix cache.
//     Turn N+1's prompt is turn N's plus two messages, which is the ideal
//     extend case: the cached prefix is reused whole and only the new tokens
//     are read. Measured inputs, from the benchmark: 1,900 tok/s warm
//     against 131 cold.
//
//  2. The slot is shared with every other client on the port — including
//     `nightsmith status`, which sends a real completion of its own. So a
//     status check between two turns costs the conversation its head start.
//
//  3. **Trimming destroys the prefix outright.** Drop the oldest exchange and
//     the prompt no longer *starts* with the cached bytes, so the entire
//     remaining conversation is read again cold. Trimming is therefore not
//     free, and that is the whole argument for doing it rarely and in one
//     large step rather than a message at a time.
//
// What is deliberately not done: summarising the conversation with the model
// to make it fit. That is a full generation at 12 tok/s, and it invents
// content. A user who wanted a summary would ask for one.

// estimateTokens is characters ÷ 4, the usual rough figure for English prose.
//
// It is never printed as a token count. Nothing here has the model's
// tokenizer, so this is a budget, not a measurement — the only numbers this
// file shows a user are words, which it can count, and the server's own
// cached_tokens, which it is told.
func estimateTokens(s string) int { return (len(s) + 3) / 4 }

type WindowPolicy struct {
	BudgetTokens int
}

type TrimReport struct {
	DroppedExchanges int
	DroppedWords     int
	EstimatedTokens  int  // what is left, after any trimming
	NearLimit        bool // crossed the soft line, and nothing was dropped
	// OverBudget means trimming ran out of things it was allowed to drop:
	// what remains is one exchange, and it is bigger than the budget on its
	// own. Nothing is wrong, but the budget is not being honoured and saying
	// so is better than printing a number that is plainly over it.
	OverBudget bool
}

// softLine is where the user is told the conversation is getting long, while
// starting a fresh one is still their choice rather than our decision.
const softLine = 0.75

// afterTrim is how far below the budget one trim goes. Trimming to exactly
// the budget would trim again next turn, and every trim costs a cold re-read
// of everything left.
const afterTrim = 0.60

// BuildWindow renders the messages actually sent.
//
// It never rewrites the session file: what is dropped is dropped from this
// request, not from the record on disk. The transcript stays whole, so a
// conversation trimmed today is still complete when it is read tomorrow.
func BuildWindow(msgs []chatMessage, p WindowPolicy) ([]chatMessage, TrimReport) {
	var r TrimReport
	total := 0
	for _, m := range msgs {
		total += estimateTokens(m.Content)
	}
	r.EstimatedTokens = total

	if p.BudgetTokens <= 0 || total <= p.BudgetTokens {
		r.NearLimit = p.BudgetTokens > 0 && float64(total) >= softLine*float64(p.BudgetTokens)
		return msgs, r
	}

	// One step, down past the budget, dropping whole exchanges from the
	// oldest end. The last message is the question just asked and is never
	// dropped — a window with nothing in it is not a smaller conversation,
	// it is a different one.
	target := int(afterTrim * float64(p.BudgetTokens))
	cut := 0
	drop := func() {
		total -= estimateTokens(msgs[cut].Content)
		r.DroppedWords += len(strings.Fields(msgs[cut].Content))
		if msgs[cut].Role == "user" {
			r.DroppedExchanges++
		}
		cut++
	}

	for cut < len(msgs)-1 && total > target {
		drop()
	}
	// Never open on an answer whose question has gone: the model would be
	// reading its own words with nothing to have prompted them. Only while
	// there is something later to keep — the final message stays whatever it
	// is, because a window with nothing in it is not a smaller conversation.
	for cut < len(msgs)-1 && msgs[cut].Role == "assistant" {
		drop()
	}

	r.EstimatedTokens = total
	r.OverBudget = total > p.BudgetTokens
	return msgs[cut:], r
}

// Notice is what the user is told, and only when there is something to tell.
// It says what it cost, because the next answer will be visibly slower to
// start and an unexplained pause reads as a fault.
func (r TrimReport) Notice() string {
	switch {
	case r.DroppedExchanges > 0:
		return fmt.Sprintf("\n  ⚠  Dropped the first %d exchange%s to make room — about %d words.\n"+
			"     The next answer will take longer to start: the rest of the\n"+
			"     conversation has to be read again from scratch.\n\n",
			r.DroppedExchanges, plural(r.DroppedExchanges), r.DroppedWords)
	case r.NearLimit:
		return "\n  ⚠  This conversation is getting long. '/new' starts a fresh one,\n" +
			"     and it will be quicker off the mark.\n\n"
	}
	return ""
}
