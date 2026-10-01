package qwen

import (
	"math"

	"webtyp.com/context"
	"webtyp.com/fmt"
	"webtyp.com/llm"
)

var _ llm.Decider = (*Model)(nil)

// Decide answers a closed question by reading the probability of each option's letter.
func (m *Model) Decide(ctx *context.Context, q llm.Question) (llm.Decision, error) {
	nOpts := len(q.Options)
	if nOpts < 2 || nOpts > 10 {
		return llm.Decision{}, fmt.Errf("qwen: a decision takes 2 to 10 options, got %d", nOpts)
	}

	for i := 0; i < nOpts; i++ {
		if m.letterIDs[i] < 0 {
			letter := string(rune('A' + i))
			return llm.Decision{}, fmt.Errf("qwen: the option letter %q is not a single token", letter)
		}
	}

	ids, prefixLen := m.decideIDs(q)

	m.mu.Lock()
	defer m.mu.Unlock()

	letters := make([]int, nOpts)
	for i := range letters {
		letters[i] = int(m.letterIDs[i])
	}
	letterLogits := make([]float32, nOpts)
	if err := m.readDecidePrompt(ids, prefixLen, letters, letterLogits); err != nil {
		return llm.Decision{}, err
	}

	temp := m.cfg.DecideTemperature
	if temp == 0 {
		temp = 1.0
	}

	z := make([]float64, nOpts)
	maxZ := math.Inf(-1)
	for i := 0; i < nOpts; i++ {
		val := float64(letterLogits[i]) / temp
		z[i] = val
		if val > maxZ {
			maxZ = val
		}
	}

	sumExp := 0.0
	probs := make([]float64, nOpts)
	for i := 0; i < nOpts; i++ {
		p := math.Exp(z[i] - maxZ)
		probs[i] = p
		sumExp += p
	}

	bestChoice := 0
	maxP := -1.0
	for i := 0; i < nOpts; i++ {
		probs[i] /= sumExp
		if probs[i] > maxP {
			maxP = probs[i]
			bestChoice = i
		}
	}

	return llm.Decision{
		Choice:     bestChoice,
		Confidence: probs[bestChoice],
		Probs:      probs,
	}, nil
}

func isYesNo(options []string) bool {
	return len(options) == 2 && options[0] == "no" && options[1] == "yes"
}

func (m *Model) decideIDs(q llm.Question) (ids []int32, prefixLen int) {
	if isYesNo(q.Options) {
		p1 := "Context:\n" + q.Context
		p2 := "\n\nQuestion: " + q.Text + "\nOptions:"
		for i, opt := range q.Options {
			letter := string(rune('A' + i))
			p2 += "\n(" + letter + ") " + opt
		}
		p2 += "\nAnswer: ("

		ids1 := m.bpe.EncodeOrdinary(nil, p1)
		ids2 := m.bpe.EncodeOrdinary(nil, p2)
		ids = append(ids, ids1...)
		ids = append(ids, ids2...)
		return ids, 0
	}

	p1 := "Question: " + q.Text + "\nOptions:"
	var p2 string
	for i, opt := range q.Options {
		letter := string(rune('A' + i))
		p2 += "\n(" + letter + ") " + opt
	}
	p3 := "\n\nContext:\n"
	p4 := q.Context
	p5 := "\n\nAnswer: ("

	ids1 := m.bpe.EncodeOrdinary(nil, p1)
	ids2 := m.bpe.EncodeOrdinary(nil, p2)
	ids3 := m.bpe.EncodeOrdinary(nil, p3)
	ids4 := m.bpe.EncodeOrdinary(nil, p4)
	ids5 := m.bpe.EncodeOrdinary(nil, p5)

	prefixLen = len(ids1) + len(ids2)

	ids = append(ids, ids1...)
	ids = append(ids, ids2...)
	ids = append(ids, ids3...)
	ids = append(ids, ids4...)
	ids = append(ids, ids5...)

	return ids, prefixLen
}

func (m *Model) readDecidePrompt(ids []int32, prefixLen int, letters []int, out []float32) error {
	st := m.stepper.NewState()
	start := 0

	if prefixLen > 0 {
		cached := &m.cache.decision
		if cached.state != nil && len(cached.ids) == prefixLen && hasPrefix(ids[:prefixLen], cached.ids) {
			if err := m.stepper.CopyState(st, cached.state); err != nil {
				return err
			}
			start = prefixLen
		}
	}

	for i := start; i < len(ids); i++ {
		// No prompt token needs the whole vocabulary's logits; only the letters, at the end.
		if err := m.stepper.Step(st, int(ids[i]), nil); err != nil {
			return err
		}
		pos := i + 1
		if prefixLen > 0 && pos == prefixLen {
			if err := m.save(&m.cache.decision, ids[:prefixLen], st); err != nil {
				return err
			}
		}
	}
	return m.stepper.LogitsFor(st, letters, out)
}
