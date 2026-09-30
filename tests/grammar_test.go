package tests

import (
	"math"
	"math/rand"
	"testing"

	"webtyp.com/llm"
	"webtyp.com/qwen"
)

func TestGrammar(t *testing.T) {
	tools := []llm.ToolDef{
		{
			Name:        "clinic_hours",
			Description: "Opening hours",
			InputSchema: `{"type": "object", "properties": {"day": {"type": "string", "enum": ["lunes", "martes"]}}, "required": ["day"]}`,
		},
	}

	vocab := [][]byte{
		[]byte("a"),
		[]byte("b"),
		[]byte("1"),
		[]byte("<|im_start|>"),
		[]byte("<|im_end|>"),
		[]byte("<tool_call>"),
		[]byte("\n</parameter>"),
	}

	g := qwen.NewGrammarForTest(tools, vocab)

	t.Run("masking special tokens", func(t *testing.T) {
		logits := make([]float32, len(vocab))
		g.MaskLogits(logits)

		if !math.IsInf(float64(logits[3]), -1) {
			t.Errorf("expected index 3 (<|im_start|>) to be masked, got %f", logits[3])
		}
		if math.IsInf(float64(logits[4]), -1) {
			t.Errorf("expected index 4 (<|im_end|>) to be allowed, got %f", logits[4])
		}
	})

	t.Run("random trial simulation", func(t *testing.T) {
		rng := rand.New(rand.NewSource(42))
		for seed := 0; seed < 1000; seed++ {
			_ = rng.Intn(100)
		}
	})
}
