package qwen

import (
	"testing"

	"webtyp.com/context"
	"webtyp.com/llm"
)

// emitStepper predicts seq, one id per step that asks for logits, then <|im_end|>.
type emitStepper struct {
	letterStepper
	seq []int
	i   int
}

func (s *emitStepper) Step(state any, token int, logits []float32) error {
	if logits == nil {
		return nil
	}
	for i := range logits {
		logits[i] = -100
	}
	next := lookupSpecialTokenID(tokenImEnd)
	if s.i < len(s.seq) {
		next = s.seq[s.i]
		s.i++
	}
	logits[next] = 100
	return nil
}

// "ñ" is two byte-level tokens (Ã = 0xC3, ± = 0xB1): the stream shows it once, whole, never half.
func TestGenerateStream_WholeCharacters(t *testing.T) {
	cfg := Config{Merges: []byte("H o"), Decoder: Qwen35_08B}
	m, err := newModel(cfg, []string{"H", "o", "Ã", "±", "<|im_end|>"}, &emitStepper{seq: []int{0, 2, 3, 1}})
	if err != nil {
		t.Fatal(err)
	}
	var chunks []string
	resp, err := m.GenerateStream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: "Ho"}}}, func(s string) {
		chunks = append(chunks, s)
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"H", "ñ", "o"}
	if len(chunks) != len(want) {
		t.Fatalf("chunks %q, want %q", chunks, want)
	}
	for i := range want {
		if chunks[i] != want[i] {
			t.Fatalf("chunks %q, want %q", chunks, want)
		}
	}
	if resp.Text != "Hño" {
		t.Fatalf("Text %q, want %q", resp.Text, "Hño")
	}
}
