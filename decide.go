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

// DecidePrompt returns the text the model reads for q, in pieces. Each piece is tokenized on its
// own and the ids are joined in order, so no token spans two pieces. A yes/no question puts the
// context first; any other puts the question and options first, the prefix kept between
// decisions (D27). agenteval sends the same pieces to llama-server, so what it measures is the
// prompt the browser reads.
func DecidePrompt(q llm.Question) []string {
	parts, _ := decideParts(q)
	return parts
}

// decideParts returns DecidePrompt's pieces and how many of them form the cached prefix.
func decideParts(q llm.Question) (parts []string, prefixParts int) {
	var options string
	for i, opt := range q.Options {
		options += "\n(" + string(rune('A'+i)) + ") " + opt
	}
	if isYesNo(q.Options) {
		return []string{
			"Context:\n" + q.Context,
			"\n\nQuestion: " + q.Text + "\nOptions:" + options + "\nAnswer: (",
		}, 0
	}
	return []string{
		"Question: " + q.Text + "\nOptions:",
		options,
		"\n\nContext:\n",
		q.Context,
		"\n\nAnswer: (",
	}, 2
}

func (m *Model) decideIDs(q llm.Question) (ids []int32, prefixLen int) {
	parts, prefixParts := decideParts(q)
	for i, p := range parts {
		ids = m.bpe.EncodeOrdinary(ids, p)
		if i == prefixParts-1 {
			prefixLen = len(ids)
		}
	}
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
