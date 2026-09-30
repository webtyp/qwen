package qwen

import "webtyp.com/llm"

// Segment represents a piece of rendered prompt: either text or a named special token.
type Segment struct {
	IsSpecial bool
	Text      string
}

// RenderForTest exports render for tests in package tests.
func RenderForTest(req llm.Request) ([]Segment, error) {
	internalSegs, err := render(req)
	if err != nil {
		return nil, err
	}
	segs := make([]Segment, len(internalSegs))
	for i, s := range internalSegs {
		segs[i] = Segment(s)
	}
	return segs, nil
}
