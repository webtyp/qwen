package tests

import (
	"testing"

	"webtyp.com/context"
	"webtyp.com/llm"
	"webtyp.com/qwen"
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
	cfg := qwen.Config{
		Decoder: qwen.Qwen35_08B,
	}
	m, err := qwen.New(cfg)
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}

	vocab := map[int]string{
		65:     "H",
		66:     "o",
		67:     "l",
		68:     "a",
		248046: "<|im_end|>",
	}
	qwen.SetVocabForTest(m, vocab)

	t.Run("normal response generation", func(t *testing.T) {
		// First few steps are prompt tokens. We provide enough scripted logits.
		script := make([]int, 200)
		for i := 0; i < 100; i++ {
			script[i] = 65 // emit 'H' or initial token
		}
		script[100] = 65 // H
		script[101] = 66 // o
		script[102] = 67 // l
		script[103] = 68 // a
		script[104] = 248046

		stepper := &scriptedStepper{
			tokensToEmit: script,
		}
		qwen.SetStepperForTest(m, stepper)

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
			script[i] = 65
		}
		script[100] = 65 // H
		script[101] = 66 // o
		script[102] = 67 // l
		script[103] = 68 // a
		script[104] = 248046

		stepper := &scriptedStepper{
			tokensToEmit: script,
		}
		qwen.SetStepperForTest(m, stepper)

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
