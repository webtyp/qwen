package qwen

import (
	"math"

	"webtyp.com/context"
	"webtyp.com/fmt"
	"webtyp.com/llm"
)

// Ensure Model implements llm.Client, llm.Streamer, llm.TokenCounter
var (
	_ llm.Client       = (*Model)(nil)
	_ llm.Streamer     = (*Model)(nil)
	_ llm.TokenCounter = (*Model)(nil)
)

// CountTokens returns the token count for a text string using EncodeOrdinary.
func (m *Model) CountTokens(text string) int {
	return len(m.bpe.EncodeOrdinary(nil, text))
}

// Generate generates a complete response for a prompt request.
func (m *Model) Generate(ctx *context.Context, req llm.Request) (llm.Response, error) {
	var fullText string
	var toolCalls []llm.ToolCall
	var stopReason llm.StopReason
	var promptTokens int
	var genTokens int

	err := m.generateStreamInternal(ctx, req, func(chunk string) {
		fullText += chunk
	}, &fullText, &toolCalls, &stopReason, &promptTokens, &genTokens)

	if err != nil {
		return llm.Response{}, err
	}

	text := fullText
	if len(req.Tools) > 0 {
		parsedText, parsedCalls, parseErr := parseToolCalls(fullText, req.Tools)
		if parseErr == nil && len(parsedCalls) > 0 {
			text = parsedText
			toolCalls = parsedCalls
			stopReason = llm.StopToolUse
		}
	}

	return llm.Response{
		Text:       text,
		StopReason: stopReason,
		ToolCalls:  toolCalls,
		Usage: llm.Usage{
			InputTokens:  promptTokens,
			OutputTokens: genTokens,
		},
	}, nil
}

// GenerateStream streams generated text tokens as they are produced.
func (m *Model) GenerateStream(ctx *context.Context, req llm.Request, onText func(string)) (llm.Response, error) {
	var fullText string
	var toolCalls []llm.ToolCall
	var stopReason llm.StopReason
	var promptTokens int
	var genTokens int

	err := m.generateStreamInternal(ctx, req, onText, &fullText, &toolCalls, &stopReason, &promptTokens, &genTokens)
	if err != nil {
		return llm.Response{}, err
	}

	text := fullText
	if len(req.Tools) > 0 {
		parsedText, parsedCalls, parseErr := parseToolCalls(fullText, req.Tools)
		if parseErr == nil && len(parsedCalls) > 0 {
			text = parsedText
			toolCalls = parsedCalls
			stopReason = llm.StopToolUse
		}
	}

	return llm.Response{
		Text:       text,
		StopReason: stopReason,
		ToolCalls:  toolCalls,
		Usage: llm.Usage{
			InputTokens:  promptTokens,
			OutputTokens: genTokens,
		},
	}, nil
}

func (m *Model) generateStreamInternal(
	ctx *context.Context,
	req llm.Request,
	onText func(string),
	fullText *string,
	toolCalls *[]llm.ToolCall,
	stopReason *llm.StopReason,
	promptTokenCount *int,
	genTokenCount *int,
) error {
	segs, err := render(req)
	if err != nil {
		return err
	}

	var promptIds []int32

	for _, seg := range segs {
		if seg.IsSpecial {
			id := lookupSpecialTokenID(seg.Text)
			if id < 0 {
				return fmt.Errf("qwen: render emitted %q, which has no token id", seg.Text)
			}
			promptIds = append(promptIds, int32(id))
		} else {
			promptIds = m.bpe.EncodeOrdinary(promptIds, seg.Text)
		}
	}

	*promptTokenCount = len(promptIds)

	st := m.stepper.NewState()

	logits := make([]float32, m.logitsSize())

	// Feed prompt
	for _, id := range promptIds {
		if err := m.stepper.Step(st, int(id), logits); err != nil {
			return err
		}
	}

	maxGen := req.MaxOutputTokens
	if maxGen <= 0 {
		maxGen = 4096
	}

	g := newGrammar(req.Tools, m.tok.vocab)
	gState := grammarState{}

	*genTokenCount = 0
	imEndID := lookupSpecialTokenID(tokenImEnd)

	for *genTokenCount < maxGen {
		g.maskLogits(logits, gState)

		nextID := argMax(logits)
		if nextID == imEndID {
			*stopReason = llm.StopEndTurn
			break
		}

		*genTokenCount++

		tokStr := m.decodeToken(nextID)
		*fullText += tokStr
		gState = g.nextState(gState, tokStr)

		if onText != nil && !gInToolCallMarkup(gState) {
			onText(tokStr)
		}

		if err := m.stepper.Step(st, nextID, logits); err != nil {
			return err
		}
	}

	if *genTokenCount >= maxGen && *stopReason == "" {
		*stopReason = llm.StopMaxTokens
	}

	return nil
}

func lookupSpecialTokenID(name string) int {
	switch name {
	case tokenEndOfText:
		return 248044
	case tokenImStart:
		return 248045
	case tokenImEnd:
		return 248046
	case tagToolCall:
		return 248058
	case tagToolCallEnd:
		return 248059
	case tagToolResponse:
		return 248066
	case tagToolResponseEnd:
		return 248067
	case tagThink:
		return 248068
	case tagThinkEnd:
		return 248069
	default:
		return -1
	}
}

// logitsSize is the width of the decoder's output: its embedding rows, which can exceed the
// tokenizer's vocabulary (Qwen3.5 pads the table to 248 320).
func (m *Model) logitsSize() int {
	if m.cfg.Decoder.Vocab > len(m.tok.vocab) {
		return m.cfg.Decoder.Vocab
	}
	return len(m.tok.vocab)
}

func argMax(logits []float32) int {
	bestID := -1
	bestVal := float32(math.Inf(-1))
	for id, val := range logits {
		if val > bestVal {
			bestVal = val
			bestID = id
		}
	}
	if bestID < 0 {
		return 0
	}
	return bestID
}

func (m *Model) decodeToken(id int) string {
	if id >= 0 && id < len(m.tok.vocab) {
		return string(m.tok.vocab[id])
	}
	return ""
}

func gInToolCallMarkup(st grammarState) bool {
	return st.mode != gModeFreeText
}
