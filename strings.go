package main

import (
	"fmt"
	"strings"
)

func shortRepo(repo string) string {
	if i := strings.LastIndex(repo, "/"); i >= 0 {
		return repo[i+1:]
	}
	return repo
}

// humanBytes prints sizes the way they must always be printed: the name
// gives no warning at all, so the size has to.
func humanBytes(b int64) string {
	gb := float64(b) / bytesPerGB
	if gb >= 1 {
		// Always one decimal, including at 111.5 GB. Rounding that to "112 GB"
		// loses nothing numerically and costs the row its precision, and this
		// row's whole job is to be believed.
		return fmt.Sprintf("%.1f GB", gb)
	}
	if b >= 1e6 {
		return fmt.Sprintf("%.0f MB", float64(b)/1e6)
	}
	return fmt.Sprintf("%.0f KB", float64(b)/1e3) // a config file is not "0 MB"
}

// humanDuration renders a wait the way a person would say it.
func humanDuration(seconds float64) string {
	switch {
	case seconds < 90:
		return fmt.Sprintf("%.0fs", seconds)
	case seconds < 3600:
		return fmt.Sprintf("%dm%02ds", int(seconds)/60, int(seconds)%60)
	default:
		return fmt.Sprintf("%dh%02dm", int(seconds)/3600, (int(seconds)%3600)/60)
	}
}

// humanSpeed uses human units: "9 words a second", not "12.6
// tok/s". Tokens are not words: English runs ~0.75 words per token on these
// tokenizers, and printing tokens as words overstated speed by a third.
func humanSpeed(tokPerSec float64) string {
	return fmt.Sprintf("about %.0f words a second", tokPerSec*wordsPerToken)
}

const wordsPerToken = 0.75
