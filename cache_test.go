package qwen

import (
	"testing"

	"webtyp.com/context"
	"webtyp.com/llm"
)

// cacheModel is a toy model whose stepper answers <|im_end|> at once (an empty answer), so a
// test can count exactly how many prompt ids each Generate reads.
func cacheModel(t *testing.T) (*Model, *scriptedStepper) {
	t.Helper()
	cfg := Config{Merges: []byte("H o\nHo l\nHol a"), Decoder: Qwen35_08B}
	m, err := newModel(cfg, []string{"H", "o", "l", "a", "<|im_end|>", "\n", "Ċ", " "}, &scriptedStepper{})
	if err != nil {
		t.Fatal(err)
	}
	s := &scriptedStepper{}
	m.setStepper(s)
	return m, s
}

func promptLen(t *testing.T, m *Model, req llm.Request) int {
	t.Helper()
	segs, err := render(req)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, s := range segs {
		if s.IsSpecial {
			n++
		} else {
			n += len(m.bpe.EncodeOrdinary(nil, s.Text))
		}
	}
	return n
}

// One generated token (<|im_end|>) is not fed: Generate stops before stepping it.
func readFor(t *testing.T, m *Model, s *scriptedStepper, req llm.Request) int {
	t.Helper()
	before := s.fed
	if _, err := m.Generate(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	return s.fed - before
}

// The next turn reads only what comes after the previous request's last message.
func TestCache_NextTurnReadsOnlyWhatIsNew(t *testing.T) {
	m, s := cacheModel(t)
	turn1 := llm.Request{System: "Hola", Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hola"}}}
	turn2 := llm.Request{System: "Hola", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: "Hola"},
		{Role: llm.RoleAssistant, Content: "Hola"},
		{Role: llm.RoleUser, Content: "Hola"},
	}}

	if got, want := readFor(t, m, s, turn1), promptLen(t, m, turn1); got != want {
		t.Fatalf("first turn read %d ids, want all %d", got, want)
	}
	// Everything up to the end of turn 1's user message is cached; the new part starts there.
	cached := len(m.cache.last.ids)
	if cached == 0 {
		t.Fatal("no snapshot after the last message")
	}
	if got, want := readFor(t, m, s, turn2), promptLen(t, m, turn2)-cached; got != want {
		t.Fatalf("second turn read %d ids, want only the %d new ones", got, want)
	}
}

// A request that shares only the system block resumes from it.
func TestCache_SameSystemResumesAfterSystemBlock(t *testing.T) {
	m, s := cacheModel(t)
	a := llm.Request{System: "Hola", Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hola"}}}
	b := llm.Request{System: "Hola", Messages: []llm.Message{{Role: llm.RoleUser, Content: "la"}}}
	readFor(t, m, s, a)
	system := len(m.cache.system.ids)
	if system == 0 {
		t.Fatal("no snapshot after the system block")
	}
	if got, want := readFor(t, m, s, b), promptLen(t, m, b)-system; got != want {
		t.Fatalf("read %d ids, want %d (everything after the system block)", got, want)
	}
}

// A different system block shares nothing: everything is read.
func TestCache_DifferentSystemReadsEverything(t *testing.T) {
	m, s := cacheModel(t)
	readFor(t, m, s, llm.Request{System: "Hola", Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hola"}}})
	other := llm.Request{System: "la", Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hola"}}}
	if got, want := readFor(t, m, s, other), promptLen(t, m, other); got != want {
		t.Fatalf("read %d ids, want all %d", got, want)
	}
}

// Resuming gives the decoder exactly the state reading the whole prompt would: the ids it holds
// after a cached read equal the full prompt.
func TestCache_ResumedStateHoldsTheWholePrompt(t *testing.T) {
	m, s := cacheModel(t)
	turn1 := llm.Request{System: "Hola", Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hola"}}}
	turn2 := llm.Request{System: "Hola", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: "Hola"}, {Role: llm.RoleAssistant, Content: "Hola"}, {Role: llm.RoleUser, Content: "la"},
	}}
	readFor(t, m, s, turn1)
	readFor(t, m, s, turn2)
	if got, want := len(m.cache.last.state.(*fakeState).ids), len(m.cache.last.ids); got != want {
		t.Fatalf("the saved state holds %d ids but claims %d", got, want)
	}
	if got := m.cache.last.state.(*fakeState).ids; !equalInts(got, toInts(m.cache.last.ids)) {
		t.Fatalf("the saved state is not the prompt prefix it claims")
	}
}

func toInts(ids []int32) []int {
	out := make([]int, len(ids))
	for i, v := range ids {
		out[i] = int(v)
	}
	return out
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
