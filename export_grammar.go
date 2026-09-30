package qwen

import "webtyp.com/llm"

type GrammarForTest struct {
	g *grammar
}

func NewGrammarForTest(tools []llm.ToolDef, vocab [][]byte) *GrammarForTest {
	return &GrammarForTest{g: newGrammar(tools, vocab)}
}

func (g *GrammarForTest) MaskLogits(logits []float32) {
	state := grammarState{}
	g.g.maskLogits(logits, state)
}

func (g *GrammarForTest) AllowsToken(tok string) bool {
	state := grammarState{}
	return g.g.allowsToken(state, tok)
}
