package main

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// A conversation on disk.
//
// JSON Lines, appended to, never rewritten. Three reasons, and the third is
// the one that will matter later:
//
//   - appending is one write per message, so a crash or a Ctrl-C loses at
//     most the turn in progress;
//   - a transcript is a log, not a config file, and unknown fields are free;
//   - **a reader skips what it does not understand rather than failing.**
//     Nothing here emits a tool call today. When something does, a transcript
//     containing one must still open in a binary that predates it — so
//     `tool_calls`, `tool_call_id` and `name` are reserved, and a record with
//     an unknown kind or role is passed over silently. That rule is tested
//     against a fixture now, while it is cheap, rather than discovered later.
//
// The store never rewrites a file, which is what makes that promise hold: a
// transcript written by a newer version cannot be mangled by an older one.

const sessionFormat = 1

type sessionHeader struct {
	V       int       `json:"v"`
	Kind    string    `json:"kind"` // "session"
	ID      string    `json:"id"`
	Created time.Time `json:"created"`
	Model   string    `json:"model"`
}

type SessionMessage struct {
	V           int       `json:"v"`
	Kind        string    `json:"kind"` // "message"
	Role        string    `json:"role"` // user | assistant | system   (later: tool)
	Content     string    `json:"content"`
	At          time.Time `json:"at"`
	Tokens      int       `json:"tokens,omitempty"`
	MS          int64     `json:"ms,omitempty"`
	Interrupted bool      `json:"interrupted,omitempty"`
}

type Session struct {
	ID       string
	Path     string
	Model    string
	Created  time.Time
	Messages []SessionMessage
}

type SessionInfo struct {
	ID    string
	When  time.Time
	Title string // the first thing the user said, shortened — derived, never stored
	Turns int
}

// knownRoles is deliberately a closed set. Anything else belongs to a version
// that knows more than this one, and is skipped rather than guessed at.
var knownRoles = map[string]bool{"user": true, "assistant": true, "system": true}

func NewSession(model string) (*Session, error) {
	if err := os.MkdirAll(chatsDir(), 0o700); err != nil {
		return nil, err
	}
	var b [2]byte
	if _, err := rand.Read(b[:]); err != nil {
		return nil, err
	}
	now := time.Now()
	s := &Session{
		ID:      hex.EncodeToString(b[:]),
		Created: now,
		Model:   model,
	}
	// The name sorts by time so the newest is found without reading any file;
	// the id a person types is the short suffix.
	s.Path = filepath.Join(chatsDir(), fmt.Sprintf("%s-%s.jsonl", now.Format("2006-01-02-150405"), s.ID))

	line, err := json.Marshal(sessionHeader{
		V: sessionFormat, Kind: "session", ID: s.ID, Created: now, Model: model})
	if err != nil {
		return nil, err
	}
	if err := appendLine(s.Path, line); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Session) Append(m SessionMessage) error {
	m.V, m.Kind = sessionFormat, "message"
	if m.At.IsZero() {
		m.At = time.Now()
	}
	line, err := json.Marshal(m)
	if err != nil {
		return err
	}
	if err := appendLine(s.Path, line); err != nil {
		return err
	}
	s.Messages = append(s.Messages, m)
	return nil
}

func appendLine(path string, line []byte) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(line, '\n'))
	return err
}

// Conversation is what gets sent to the model: the messages this version
// understands, in order.
func (s *Session) Conversation() []chatMessage {
	var out []chatMessage
	for _, m := range s.Messages {
		out = append(out, chatMessage{Role: m.Role, Content: m.Content})
	}
	return out
}

// readSession is tolerant on purpose. A file killed mid-write ends in a
// partial line, and every complete record before it is still good.
func readSession(path string) (*Session, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	s := &Session{Path: path}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if len(strings.TrimSpace(string(line))) == 0 {
			continue
		}
		var probe struct {
			Kind string `json:"kind"`
			Role string `json:"role"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			continue // a half-written final line, or a record from the future
		}
		switch probe.Kind {
		case "session":
			var h sessionHeader
			if json.Unmarshal(line, &h) == nil {
				s.ID, s.Model, s.Created = h.ID, h.Model, h.Created
			}
		case "message":
			if !knownRoles[probe.Role] {
				continue // a role this version does not have: skip, don't fail
			}
			var m SessionMessage
			if json.Unmarshal(line, &m) == nil {
				s.Messages = append(s.Messages, m)
			}
		}
	}
	if s.ID == "" {
		return nil, fmt.Errorf("%s isn't a conversation", shortPath(path))
	}
	return s, nil
}

// sessionFiles lists the transcripts newest first. The filename carries the
// timestamp, so ordering costs no reads.
func sessionFiles() ([]string, error) {
	entries, err := os.ReadDir(chatsDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
			names = append(names, e.Name())
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(names)))
	paths := make([]string, len(names))
	for i, n := range names {
		paths[i] = filepath.Join(chatsDir(), n)
	}
	return paths, nil
}

func LatestSession() (*Session, error) {
	paths, err := sessionFiles()
	if err != nil {
		return nil, err
	}
	for _, p := range paths {
		if s, err := readSession(p); err == nil {
			return s, nil
		}
	}
	return nil, fmt.Errorf("there are no conversations on this Mac yet")
}

// OpenSession resolves an id the way git resolves a short sha: any unique
// prefix will do, and an ambiguous one says so rather than picking.
func OpenSession(prefix string) (*Session, error) {
	prefix = strings.ToLower(strings.TrimSpace(prefix))
	if prefix == "" {
		return LatestSession()
	}
	paths, err := sessionFiles()
	if err != nil {
		return nil, err
	}
	var hits []*Session
	for _, p := range paths {
		s, err := readSession(p)
		if err != nil {
			continue
		}
		if strings.HasPrefix(s.ID, prefix) {
			hits = append(hits, s)
		}
	}
	switch len(hits) {
	case 0:
		return nil, fmt.Errorf("no conversation starts with %q", prefix)
	case 1:
		return hits[0], nil
	default:
		ids := make([]string, len(hits))
		for i, h := range hits {
			ids[i] = h.ID
		}
		return nil, fmt.Errorf("%q matches %s — say which", prefix, strings.Join(ids, ", "))
	}
}

func ListSessions() ([]SessionInfo, error) {
	paths, err := sessionFiles()
	if err != nil {
		return nil, err
	}
	var out []SessionInfo
	for _, p := range paths {
		s, err := readSession(p)
		if err != nil {
			continue
		}
		info := SessionInfo{ID: s.ID, When: s.Created}
		for _, m := range s.Messages {
			if m.Role == "user" {
				info.Turns++
				if info.Title == "" {
					info.Title = firstLine(m.Content, 48)
				}
			}
		}
		if info.Turns == 0 {
			continue // opened and never used; not worth offering
		}
		out = append(out, info)
	}
	return out, nil
}

// firstLine is how a conversation gets a name. Deliberately not generated:
// asking the model for a title costs a whole completion at 12 tok/s, which is
// three seconds of someone's life spent on a label.
func firstLine(s string, max int) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > max {
		return strings.TrimSpace(s[:max]) + "…"
	}
	return s
}
