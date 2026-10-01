package qwen

import (
	"math"
	"testing"

	"webtyp.com/context"
	"webtyp.com/llm"
)

func byteVocab() []string {
	var bs []int
	for b := '!'; b <= '~'; b++ {
		bs = append(bs, int(b))
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, b)
	}
	sym := make([]string, 256)
	n := 0
	for b := 0; b < 256; b++ {
		in := false
		for _, x := range bs {
			if x == b {
				in = true
				break
			}
		}
		if in {
			sym[b] = string(rune(b))
		} else {
			sym[b] = string(rune(256 + n))
			n++
		}
	}
	return sym
}

type letterStepper struct {
	want []float32
	fed  []int
}

func (s *letterStepper) NewState() any { return &fakeState{} }

func (s *letterStepper) CopyState(dst, src any) error {
	d, o := dst.(*fakeState), src.(*fakeState)
	d.ids = append(d.ids[:0], o.ids...)
	return nil
}

func (s *letterStepper) Step(state any, token int, logits []float32) error {
	st := state.(*fakeState)
	st.ids = append(st.ids, token)
	s.fed = append(s.fed, token)

	for i := range logits {
		logits[i] = -100
	}
	for i, w := range s.want {
		if 65+i < len(logits) {
			logits[65+i] = w
		}
	}
	return nil
}

func decodeIDs(vocab [][]byte, ids []int32) string {
	var res string
	for _, id := range ids {
		if int(id) < len(vocab) {
			res += string(vocab[id])
		}
	}
	return res
}

func TestDecide_ReadoutAndTemperature(t *testing.T) {
	vocab := append(byteVocab(), "Ho")
	cfg := Config{
		Merges:            []byte("H o"),
		Decoder:           Qwen35_08B,
		DecideTemperature: 1,
	}

	ls := &letterStepper{want: []float32{0, 2}}
	m, err := newModel(cfg, vocab, ls)
	if err != nil {
		t.Fatal(err)
	}

	q := llm.Question{
		Context: "Some context",
		Text:    "Which letter?",
		Options: []string{"a", "b"},
	}

	dec, err := m.Decide(context.Background(), q)
	if err != nil {
		t.Fatalf("Decide failed: %v", err)
	}

	if dec.Choice != 1 {
		t.Errorf("Choice = %d, want 1", dec.Choice)
	}

	z0, z1 := 0.0, 2.0
	expSum := math.Exp(z0) + math.Exp(z1)
	p0Want := math.Exp(z0) / expSum
	p1Want := math.Exp(z1) / expSum

	if math.Abs(dec.Probs[0]-p0Want) > 1e-6 {
		t.Errorf("Probs[0] = %v, want %v", dec.Probs[0], p0Want)
	}
	if math.Abs(dec.Probs[1]-p1Want) > 1e-6 {
		t.Errorf("Probs[1] = %v, want %v", dec.Probs[1], p1Want)
	}
	if math.Abs(dec.Confidence-p1Want) > 1e-6 {
		t.Errorf("Confidence = %v, want %v", dec.Confidence, p1Want)
	}

	// Temperature = 2
	m.cfg.DecideTemperature = 2
	decT2, err := m.Decide(context.Background(), q)
	if err != nil {
		t.Fatalf("Decide with T=2 failed: %v", err)
	}

	tz0, tz1 := 0.0, 1.0
	tExpSum := math.Exp(tz0) + math.Exp(tz1)
	tp0Want := math.Exp(tz0) / tExpSum
	tp1Want := math.Exp(tz1) / tExpSum

	if math.Abs(decT2.Probs[0]-tp0Want) > 1e-6 {
		t.Errorf("T=2 Probs[0] = %v, want %v", decT2.Probs[0], tp0Want)
	}
	if math.Abs(decT2.Probs[1]-tp1Want) > 1e-6 {
		t.Errorf("T=2 Probs[1] = %v, want %v", decT2.Probs[1], tp1Want)
	}
}

func TestDecide_Layouts(t *testing.T) {
	vocab := append(byteVocab(), "Ho")
	cfg := Config{
		Merges:  []byte("H o"),
		Decoder: Qwen35_08B,
	}

	t.Run("state-first for yes/no", func(t *testing.T) {
		ls := &letterStepper{want: []float32{1, 0}}
		m, err := newModel(cfg, vocab, ls)
		if err != nil {
			t.Fatal(err)
		}

		q := llm.Question{
			Context: "Context text",
			Text:    "Is it real?",
			Options: []string{"no", "yes"},
		}

		_, err = m.Decide(context.Background(), q)
		if err != nil {
			t.Fatal(err)
		}

		fedString := decodeIDs(m.tok.vocab, toInt32s(ls.fed))
		if len(fedString) < 9 || fedString[:9] != "Context:\n" {
			t.Errorf("state-first prompt should start with 'Context:\\n', got %q", fedString)
		}
	})

	t.Run("schema-first for other options", func(t *testing.T) {
		ls := &letterStepper{want: []float32{1, 0, 0}}
		m, err := newModel(cfg, vocab, ls)
		if err != nil {
			t.Fatal(err)
		}

		q := llm.Question{
			Context: "Context text",
			Text:    "Which tool?",
			Options: []string{"x", "y", "z"},
		}

		_, err = m.Decide(context.Background(), q)
		if err != nil {
			t.Fatal(err)
		}

		fedString := decodeIDs(m.tok.vocab, toInt32s(ls.fed))
		if len(fedString) < 10 || fedString[:10] != "Question: " {
			t.Errorf("schema-first prompt should start with 'Question: ', got %q", fedString)
		}
	})
}

func TestDecide_OptionCountErrors(t *testing.T) {
	vocab := append(byteVocab(), "Ho")
	cfg := Config{
		Merges:  []byte("H o"),
		Decoder: Qwen35_08B,
	}
	ls := &letterStepper{want: []float32{1}}
	m, err := newModel(cfg, vocab, ls)
	if err != nil {
		t.Fatal(err)
	}

	q1 := llm.Question{Context: "c", Text: "q", Options: []string{"a"}}
	if _, err := m.Decide(context.Background(), q1); err == nil || err.Error() != "qwen: a decision takes 2 to 10 options, got 1" {
		t.Errorf("1 option: got %v", err)
	}

	opts11 := make([]string, 11)
	for i := range opts11 {
		opts11[i] = "opt"
	}
	q11 := llm.Question{Context: "c", Text: "q", Options: opts11}
	if _, err := m.Decide(context.Background(), q11); err == nil || err.Error() != "qwen: a decision takes 2 to 10 options, got 11" {
		t.Errorf("11 options: got %v", err)
	}
}

func TestDecide_GenerateCacheUntouched(t *testing.T) {
	vocab := append(byteVocab(), "Ho")
	cfg := Config{
		Merges:  []byte("H o"),
		Decoder: Qwen35_08B,
	}
	ls := &letterStepper{want: []float32{1, 0}}
	m, err := newModel(cfg, vocab, ls)
	if err != nil {
		t.Fatal(err)
	}

	q := llm.Question{Context: "ctx", Text: "q", Options: []string{"no", "yes"}}
	_, err = m.Decide(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}

	if m.cache.system.state != nil || len(m.cache.system.ids) != 0 {
		t.Errorf("Generate system cache modified by Decide")
	}
	if m.cache.last.state != nil || len(m.cache.last.ids) != 0 {
		t.Errorf("Generate last cache modified by Decide")
	}
}

func TestDecide_SchemaFirstPrefixReuse(t *testing.T) {
	vocab := append(byteVocab(), "Ho")
	cfg := Config{
		Merges:  []byte("H o"),
		Decoder: Qwen35_08B,
	}
	ls := &letterStepper{want: []float32{1, 0, 0}}
	m, err := newModel(cfg, vocab, ls)
	if err != nil {
		t.Fatal(err)
	}

	q1 := llm.Question{
		Context: "Context 1",
		Text:    "Tool question",
		Options: []string{"tool1", "tool2", "tool3"},
	}
	q2 := llm.Question{
		Context: "Context 2",
		Text:    "Tool question",
		Options: []string{"tool1", "tool2", "tool3"},
	}

	ids1, prefixLen1 := m.decideIDs(q1)
	ids2, prefixLen2 := m.decideIDs(q2)

	if prefixLen1 == 0 || prefixLen1 != prefixLen2 {
		t.Fatalf("expected non-zero equal prefixLen, got %d and %d", prefixLen1, prefixLen2)
	}

	ls.fed = nil
	dec1, err := m.Decide(context.Background(), q1)
	if err != nil {
		t.Fatal(err)
	}
	fed1 := len(ls.fed)
	if fed1 != len(ids1) {
		t.Fatalf("first Decide fed %d tokens, want %d", fed1, len(ids1))
	}

	ls.fed = nil
	dec2, err := m.Decide(context.Background(), q2)
	if err != nil {
		t.Fatal(err)
	}
	fed2 := len(ls.fed)
	wantFed2 := len(ids2) - prefixLen2
	if fed2 != wantFed2 {
		t.Fatalf("second Decide fed %d tokens, want %d", fed2, wantFed2)
	}

	if dec1.Choice != dec2.Choice {
		t.Errorf("decisions differ: %d vs %d", dec1.Choice, dec2.Choice)
	}
}

func toInt32s(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, v := range ids {
		out[i] = int32(v)
	}
	return out
}
