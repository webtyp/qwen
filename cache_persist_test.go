package qwen

import (
	"testing"

	"webtyp.com/context"
	"webtyp.com/llm"
	"webtyp.com/weights"
)

func persistModel(t *testing.T, ls *letterStepper, id string) *Model {
	t.Helper()
	cfg := Config{Merges: []byte("H o"), Decoder: Qwen35_08B, Weights: &weights.Artifact{ID: id, Version: 1}}
	m, err := newModel(cfg, append(byteVocab(), "Ho"), ls)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

var toolQuestion = llm.Question{Context: "Context 1", Text: "Tool question", Options: []string{"tool1", "tool2", "tool3"}}

// A Worker that starts again loads the saved tool list and its first decision reads only the new
// context, not the whole question again, and decides the same.
func TestDecisionCache_SavedAndLoadedSkipsThePrefix(t *testing.T) {
	first := &letterStepper{want: []float32{0, 2, 1}}
	m := persistModel(t, first, "decider-0.8b")
	if _, ok, err := m.SaveDecisionCache(); ok || err != nil {
		t.Fatalf("before any decision: ok=%v err=%v, want nothing to save", ok, err)
	}
	want, err := m.Decide(context.Background(), toolQuestion)
	if err != nil {
		t.Fatal(err)
	}
	data, ok, err := m.SaveDecisionCache()
	if !ok || err != nil {
		t.Fatalf("ok=%v err=%v", ok, err)
	}

	second := &letterStepper{want: []float32{0, 2, 1}}
	restarted := persistModel(t, second, "decider-0.8b")
	if err := restarted.LoadDecisionCache(data); err != nil {
		t.Fatal(err)
	}
	q := toolQuestion
	q.Context = "Context 2"
	ids, prefixLen := restarted.decideIDs(q)
	got, err := restarted.Decide(context.Background(), q)
	if err != nil {
		t.Fatal(err)
	}
	if len(second.fed) != len(ids)-prefixLen {
		t.Fatalf("fed %d tokens, want %d (the prefix comes from the saved cache)", len(second.fed), len(ids)-prefixLen)
	}
	if got.Choice != want.Choice {
		t.Errorf("choice %d, want %d", got.Choice, want.Choice)
	}
}

// Bytes that are not a saved cache, or come from other weights, are refused.
func TestDecisionCache_LoadRejectsBadBytesAndOtherWeights(t *testing.T) {
	m := persistModel(t, &letterStepper{want: []float32{1, 0, 0}}, "decider-0.8b")
	if _, err := m.Decide(context.Background(), toolQuestion); err != nil {
		t.Fatal(err)
	}
	data, _, err := m.SaveDecisionCache()
	if err != nil {
		t.Fatal(err)
	}
	for name, bad := range map[string][]byte{"empty": nil, "not a cache": []byte("hello"), "cut short": data[:12]} {
		if err := m.LoadDecisionCache(bad); err == nil || err.Error() != errDecisionCacheFormat {
			t.Errorf("%s: got %v, want %q", name, err, errDecisionCacheFormat)
		}
	}
	other := persistModel(t, &letterStepper{want: []float32{1, 0, 0}}, "qwen3.5-0.8b")
	want := "qwen: the decision cache was saved by decider-0.8b v1, this model is qwen3.5-0.8b v1"
	if err := other.LoadDecisionCache(data); err == nil || err.Error() != want {
		t.Fatalf("got %v, want %q", err, want)
	}
}
