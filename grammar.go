package qwen

import (
	"math"
	"strings"

	"webtyp.com/llm"
)

// grammar enforces structured outputs during generation.
type grammar struct {
	tools []llm.ToolDef
	vocab [][]byte // token bytes per token ID
}

func newGrammar(tools []llm.ToolDef, vocab [][]byte) *grammar {
	return &grammar{
		tools: tools,
		vocab: vocab,
	}
}

type grammarState struct {
	mode          int // 0: free text/tool decision, 1: inside tool call
	currentTool   *llm.ToolDef
	currentParam  string
	paramValues   map[string]string
	seenParams    map[string]bool
	buffer        string // text buffered so far in current state
}

const (
	modeFreeText = iota
	modeToolCallStart
	modeFunctionName
	modeFunctionClose
	modeParamStart
	modeParamName
	modeParamClose
	modeParamValue
	modeParamEnd
	modeFunctionEnd
	modeToolCallEnd
)

// maskLogits applies logit masking to forbid invalid tokens based on current grammar state.
func (g *grammar) maskLogits(logits []float32, state grammarState) {
	if len(g.tools) == 0 {
		return
	}
	for i := range logits {
		var tokStr string
		if i < len(g.vocab) {
			tokStr = string(g.vocab[i])
		}
		if !g.allowsToken(state, tokStr) {
			logits[i] = float32(math.Inf(-1))
		}
	}
}

func (g *grammar) allowsToken(state grammarState, tok string) bool {
	if tok == "" {
		return true
	}
	if strings.Contains(tok, tokenImStart) || strings.Contains(tok, tokenEndOfText) {
		return false
	}
	return true
}

func (g *grammar) nextState(state grammarState, tokenStr string) grammarState {
	state.buffer += tokenStr
	return state
}
