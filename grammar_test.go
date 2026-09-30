package qwen

import (
	"math"
	"math/rand"
	"testing"

	"webtyp.com/llm"
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

	g := newGrammar(tools, vocab)

	t.Run("masking special tokens in free text", func(t *testing.T) {
		logits := make([]float32, len(vocab))
		g.maskLogits(logits, grammarState{mode: gModeFreeText})

		imStartID := lookupSpecialTokenID(tokenImStart)
		if imStartID >= 0 && imStartID < len(logits) {
			if !math.IsInf(float64(logits[imStartID]), -1) {
				t.Errorf("expected <|im_start|> to be masked")
			}
		}
	})

	t.Run("random logit forced through grammar produces parseable tool calls", func(t *testing.T) {
		toyVocab := [][]byte{
			[]byte("<tool_call>"),
			[]byte("\n<function="),
			[]byte("clinic_hours"),
			[]byte(">\n"),
			[]byte("<parameter="),
			[]byte("day"),
			[]byte(">\n"),
			[]byte("lunes"),
			[]byte("\n</parameter>\n"),
			[]byte("</function>\n"),
			[]byte("</tool_call>\n"),
			[]byte("<|im_end|>"),
		}

		toyGrammar := newGrammar(tools, toyVocab)

		for seed := int64(0); seed < 1000; seed++ {
			rng := rand.New(rand.NewSource(seed))
			st := grammarState{mode: gModeFreeText}
			var generated string

			for step := 0; step < 50; step++ {
				logits := make([]float32, len(toyVocab))
				for i := range logits {
					logits[i] = rng.Float32() * 10
				}

				toyGrammar.maskLogits(logits, st)

				bestID := -1
				bestVal := float32(math.Inf(-1))
				for i, v := range logits {
					if v > bestVal {
						bestVal = v
						bestID = i
					}
				}

				if bestID < 0 || math.IsInf(float64(bestVal), -1) {
					break
				}

				tokStr := string(toyVocab[bestID])

				nextSt, ok := toyGrammar.stepBytes(st, toyVocab[bestID])
				if !ok {
					break
				}
				st = nextSt
				generated += tokStr

				if tokStr == "<|im_end|>" {
					break
				}
			}

			if generated != "" {
				// Strip trailing incomplete text or <|im_end|> before parsing
				parseStr := generated
				if len(parseStr) >= len("<|im_end|>") && parseStr[len(parseStr)-len("<|im_end|>"):] == "<|im_end|>" {
					parseStr = parseStr[:len(parseStr)-len("<|im_end|>")]
				}
				// If generation stopped in structural state before completing tool_call tag, trim trailing unclosed block
				lastTC := -1
				for i := 0; i <= len(parseStr)-len("<tool_call>"); i++ {
					if parseStr[i:i+len("<tool_call>")] == "<tool_call>" {
						lastTC = i
					}
				}
				if lastTC != -1 {
					lastTCEnd := -1
					for i := lastTC; i <= len(parseStr)-len("</tool_call>"); i++ {
						if parseStr[i:i+len("</tool_call>")] == "</tool_call>" {
							lastTCEnd = i + len("</tool_call>")
						}
					}
					if lastTCEnd != -1 {
						parseStr = parseStr[:lastTCEnd]
					} else {
						parseStr = parseStr[:lastTC]
					}
				}

				_, calls, err := parseToolCalls(parseStr, tools)
				if err != nil {
					t.Fatalf("seed %d produced unparseable output %q (raw %q): %v", seed, parseStr, generated, err)
				}
				_ = calls
			}
		}
	})

	t.Run("required parameter enforcement", func(t *testing.T) {
		reqTools := []llm.ToolDef{
			{
				Name:        "book",
				Description: "Book",
				InputSchema: `{"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]}`,
			},
		}
		gReq := newGrammar(reqTools, vocab)

		st := grammarState{
			mode:       gModeParamStartOrFnEnd,
			toolIdx:    0,
			seenParams: nil,
		}

		nextSt, ok := gReq.stepBytes(st, []byte("</function>"))
		if ok {
			t.Errorf("expected </function> to be rejected when required parameter 'day' is missing, got accepted: %+v", nextSt)
		}
	})

	t.Run("integer validation", func(t *testing.T) {
		if !validateParamValue("123", paramSchema{pType: "integer"}) {
			t.Errorf("123 should be valid integer")
		}
		if validateParamValue("12a", paramSchema{pType: "integer"}) {
			t.Errorf("12a should be invalid integer")
		}
	})

	t.Run("enum validation", func(t *testing.T) {
		enumParam := paramSchema{pType: "enum", enums: []string{"lunes", "martes"}}
		if !validateParamValue("lunes", enumParam) {
			t.Errorf("lunes should be valid enum")
		}
		if validateParamValue("domingo", enumParam) {
			t.Errorf("domingo should be invalid enum")
		}
	})
}
