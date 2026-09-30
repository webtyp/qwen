package qwen

import (
	"testing"
	"webtyp.com/decoder"

	"webtyp.com/context"
	"webtyp.com/llm"
)

type scriptedStepper struct {
	tokensToEmit []int
	stepIdx      int
}

func (s *scriptedStepper) NewState() any {
	return s
}

func (s *scriptedStepper) Step(state any, token int, logits []float32) error {
	for i := range logits {
		logits[i] = -100
	}

	if s.stepIdx < len(s.tokensToEmit) {
		nextTok := s.tokensToEmit[s.stepIdx]
		s.stepIdx++
		if nextTok < len(logits) {
			logits[nextTok] = 100
		}
	} else {
		if 248046 < len(logits) {
			logits[248046] = 100
		}
	}
	return nil
}

func TestGenerateWithFakeStepper(t *testing.T) {
	cfg := Config{
		Merges:  []byte("H o\nHo l\nHol a"),
		Decoder: Qwen35_08B,
	}

	m, err := newModel(cfg, []string{"H", "o", "l", "a", "<|im_end|>"}, &scriptedStepper{})
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}

	t.Run("normal response generation", func(t *testing.T) {
		script := make([]int, 200)
		for i := 0; i < 100; i++ {
			script[i] = 0
		}
		script[100] = 0 // H
		script[101] = 1 // o
		script[102] = 2 // l
		script[103] = 3 // a
		script[104] = 248046

		m.stepper = &scriptedStepper{
			tokensToEmit: script,
		}

		req := llm.Request{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hola"},
			},
		}

		ctx := context.Background()
		resp, err := m.Generate(ctx, req)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}

		if resp.Text == "" {
			t.Errorf("expected non-empty Text, got %q", resp.Text)
		}
		if resp.StopReason != llm.StopEndTurn {
			t.Errorf("expected StopReason 'end_turn', got %q", resp.StopReason)
		}
	})

	t.Run("streaming response text pieces", func(t *testing.T) {
		script := make([]int, 200)
		for i := 0; i < 100; i++ {
			script[i] = 0
		}
		script[100] = 0 // H
		script[101] = 1 // o
		script[102] = 2 // l
		script[103] = 3 // a
		script[104] = 248046

		m.stepper = &scriptedStepper{
			tokensToEmit: script,
		}

		req := llm.Request{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "Hola"},
			},
		}

		var chunks []string
		ctx := context.Background()
		resp, err := m.GenerateStream(ctx, req, func(text string) {
			chunks = append(chunks, text)
		})
		if err != nil {
			t.Fatalf("GenerateStream failed: %v", err)
		}

		if resp.Text == "" {
			t.Errorf("expected non-empty Text, got %q", resp.Text)
		}
		if len(chunks) == 0 {
			t.Error("expected streamed chunks, got none")
		}
	})
}

func TestNew_RequiresWeights(t *testing.T) {
	if _, err := New(Config{Merges: []byte("H o")}); err == nil || err.Error() != errWeightsRequired {
		t.Fatalf("New without weights: got %v, want %q", err, errWeightsRequired)
	}
}

func TestQwen35_08B_LayerPattern(t *testing.T) {
	full := 0
	for i, k := range Qwen35_08B.Layers {
		if want := (i+1)%4 == 0; (k == decoder.FullAttention) != want {
			t.Errorf("layer %d: kind %v, full attention expected %v", i, k, want)
		}
		if k == decoder.FullAttention {
			full++
		}
	}
	if len(Qwen35_08B.Layers) != 24 || full != 6 {
		t.Errorf("want 24 layers with 6 full attention, got %d with %d", len(Qwen35_08B.Layers), full)
	}
	if err := Qwen35_08B.Validate(); err != nil {
		t.Errorf("Qwen35_08B.Validate: %v", err)
	}
}

// The grammar reads the bytes a token writes: "Ċ" in the vocabulary is a newline.
func TestNewModel_DecodesByteLevelTokens(t *testing.T) {
	m, err := newModel(Config{Merges: []byte("a b")}, []string{"Ċ", "Ġa", "<tool_call>"}, &scriptedStepper{})
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []string{"\n", " a", "<tool_call>"} {
		if got := string(m.tok.vocab[i]); got != want {
			t.Errorf("token %d decodes to %q, want %q", i, got, want)
		}
	}
}
