package qwen

// snapshot is the decoder state after reading ids, exactly.
type snapshot struct {
	ids   []int32
	state any
}

// prefixCache keeps two snapshots of what the model read last: after the system block
// (identity and tools, the same for a whole conversation) and after the last complete message
// before the answer. The next request of the same turn, and the first request of the next turn,
// both start with that last message: the chat template drops the <think> block of earlier
// assistant turns, so a request rarely starts with the whole previous prompt, but it always
// starts with everything up to the previous request's last message.
type prefixCache struct {
	system snapshot
	last   snapshot
}

// reset forgets both snapshots (a new stepper means a different model state).
func (c *prefixCache) reset() { *c = prefixCache{} }

// resumeFrom returns the longest snapshot whose ids are a prefix of prompt and shorter than it
// (at least the last prompt id must be read, to get the logits of the first answer token), or
// nil.
func (c *prefixCache) resumeFrom(prompt []int32) *snapshot {
	var best *snapshot
	for _, s := range []*snapshot{&c.system, &c.last} {
		if s.state == nil || len(s.ids) >= len(prompt) || !hasPrefix(prompt, s.ids) {
			continue
		}
		if best == nil || len(s.ids) > len(best.ids) {
			best = s
		}
	}
	return best
}

func hasPrefix(ids, prefix []int32) bool {
	if len(prefix) > len(ids) {
		return false
	}
	for i := range prefix {
		if ids[i] != prefix[i] {
			return false
		}
	}
	return true
}

// messageEnds returns the positions just after each "<|im_end|>" followed by newlineID: the end
// of every complete message in prompt, in order.
func messageEnds(prompt []int32, newlineID int32) []int {
	imEnd := int32(lookupSpecialTokenID(tokenImEnd))
	var ends []int
	for i := 0; i+1 < len(prompt); i++ {
		if prompt[i] == imEnd && prompt[i+1] == newlineID {
			ends = append(ends, i+2)
		}
	}
	return ends
}
