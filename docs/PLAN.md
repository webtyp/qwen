---
PLAN: "feat: Model implements llm.Decider — decider-0.8b answers closed questions from the option letters' probabilities"
TAG: v0.3.0
EXECUTOR: jules
REVIEWER: none
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.

# Plan — `webtyp.com/qwen`: closed questions (`llm.Decider`)

## 0. Context

`qwen` runs Qwen3.5 models in Go for the browser (`model.go`, `generate.go`): `Generate` writes
text, `CountTokens` counts. The webtyp agent is becoming **hybrid**: a *decision model* picks the
tool, answers yes/no questions from data and flags injections, and code does the rest. That model
is **decider-0.8b**, a Qwen3.5-0.8B fine-tune with the same architecture and tensor names this
package already runs. It does not write text. It answers a closed question by reading, in **one
pass over the prompt**, the probability of each option's letter as the next token.

The contract is `webtyp.com/llm` v0.2.0:

```go
type Decider interface {
	Decide(ctx *context.Context, q Question) (Decision, error)
}
type Question struct {
	Context string   // what the model reads before the question
	Text    string   // the question itself
	Options []string // 2 to 10 possible answers; yes/no questions use exactly {"no", "yes"}
}
type Decision struct {
	Choice     int       // index into Question.Options of the most probable option
	Confidence float64   // the probability of Choice
	Probs      []float64 // one per option, in the order given; they sum to 1
}
```

### The prompt (measured on 2026-09-30 with decider-0.8b on `llama-server`)

decider-0.8b was trained on two layouts, and each is better for one kind of question:

| Layout | Route among 9 tools | Injection yes/no | Yes/no from data |
|---|---|---|---|
| **state-first** | 14/18 | **10/10** | **8/8** |
| **schema-first** | **18/18** | 6/10 | 6/8 |

**The rule:** options exactly `["no", "yes"]` → state-first; any other options → schema-first.
Schema-first also puts the question and options **before** the context, so a list of tools that
does not change can be read once and reused (stage 3).

Each piece is encoded **separately** with `m.bpe.EncodeOrdinary` (no special tokens), and the id
slices are concatenated in this order. Letters are `A`, `B`, … `J`. `\n` is a newline.

- **state-first:** `"Context:\n" + Context`, then
  `"\n\nQuestion: " + Text + "\nOptions:" + "\n(A) " + Options[0] + "\n(B) " + Options[1] + … + "\nAnswer: ("`.
- **schema-first:** `"Question: " + Text + "\nOptions:"`, then
  `"\n(A) " + Options[0] + "\n(B) " + Options[1] + …`, then `"\n\nContext:\n"`, then `Context`,
  then `"\n\nAnswer: ("`.

**The answer:** after feeding the prompt, the logits of the next token. For option `i`,
`z_i = logits[letterID(i)] / T`, and `p = softmax(z)` over the options only. `Choice` is the
argmax, `Confidence = p[Choice]`, and `Probs = p`. `T` is the decision temperature from
decider-0.8b's `decider_config.json`: **1.03**.

## Development rules (inline)

- Primary runtime: browser, TinyGo/WASM (`GOOS=js GOARCH=wasm go build ./...`). Never import
  `strings`, `fmt`, `errors`, `strconv` (use `webtyp.com/fmt`), `sort` or `map[K]V` in non-test
  files.
- Tests in this repository live in the root as `package qwen` (see `AGENTS.md`). Add the new test
  file's name to `AGENTS.md`'s list with its reason.
- `go get webtyp.com/llm@v0.2.0`.
- No `TODO`. `gotest` green.

## Design gate (api-design — five answers)

1. **Prior art.** `decider.infer.Decider.decide` (the model author's library) reads option-letter
   logits at an answer slot. OpenAI's `logprobs` and llama.cpp's `n_probs` expose the same
   readout. Constrained classification in `outlines` picks among choices by likelihood. We
   implement the contract webtyp already has (`llm.Decider`).
2. **Novice-name test.** `model.Decide(ctx, llm.Question{…})`. `Config.DecideTemperature`:
   "the temperature of decisions".
3. **Complexity ledger.** Concepts +1 (`DecideTemperature`), call site +0, ways +0.
4. **Where it belongs.** The runtime that holds the model reads its logits. `qwen` owns the
   tokenizer and the decoder state.
5. **What it deletes.** Nothing. It adds capability.

## Stage 1 — `Decide` (`decide.go`, `model.go`)

- `Config` gains
  `DecideTemperature float64 // the decision temperature (decider-0.8b: 1.03); 0 means 1`.
- `var _ llm.Decider = (*Model)(nil)`.
- `func (m *Model) Decide(ctx *context.Context, q llm.Question) (llm.Decision, error)`:
  1. `len(q.Options)` not in 2..10 → error
     `qwen: a decision takes 2 to 10 options, got %d` (constant, `fmt.Errf`).
  2. Build the ids with the layout rule of §0 (`decideIDs(q) (ids []int32, prefixLen int)`, where
     `prefixLen` is the length of the first two pieces for schema-first and 0 for state-first).
  3. `m.mu.Lock()`, then read the ids with the cache of stage 3, and read the logits after the
     last id.
  4. Letter ids: `EncodeOrdinary(nil, "A")` … `"J"`, computed once and kept in `Model`
     (`letterIDs []int32`). Each must be exactly one token, or Decide returns
     `qwen: the option letter %q is not a single token`.
  5. Softmax with `T = DecideTemperature` (or 1 when it is 0) over the first `len(Options)`
     letters. Return the `llm.Decision`.
- `Decide` never changes what `Generate` does: it does not write `m.cache.system` or `m.cache.last`.

## Stage 2 — tests of the readout (`decide_test.go`)

A toy model whose vocabulary is the 256 byte-level symbols (so any text encodes, one id per
byte), plus `"Ho"` (so the one merge `"H o"` is valid). The letter `A` is then id 65, `B` 66,
and so on.

```go
// byteVocab is GPT-2's byte-to-unicode table: index b is the symbol of byte b.
func byteVocab() []string {
	var bs []int
	for b := '!'; b <= '~'; b++ { bs = append(bs, int(b)) }
	for b := 0xA1; b <= 0xAC; b++ { bs = append(bs, b) }
	for b := 0xAE; b <= 0xFF; b++ { bs = append(bs, b) }
	sym := make([]string, 256)
	n := 0
	for b := 0; b < 256; b++ {
		in := false
		for _, x := range bs { if x == b { in = true; break } }
		if in { sym[b] = string(rune(b)) } else { sym[b] = string(rune(256 + n)); n++ }
	}
	return sym
}
```

`newModel(Config{Merges: []byte("H o"), DecideTemperature: 1}, append(byteVocab(), "Ho"), stepper)`.

The fake stepper for decisions (`letterStepper`) has a `want []float32` of letter logits. On every
`Step` it records the id fed (in its state, as `scriptedStepper` does) and sets
`logits[65+i] = want[i]`, and every other logit to −100. It implements `CopyState` like
`scriptedStepper`.

- **Readout:** `want = {0, 2}`, options `["a", "b"]`, T = 1 → Choice 1,
  `Probs = softmax(0, 2)` within 1e-6, and `Confidence = Probs[1]`.
- **Temperature:** the same with T = 2 → `Probs = softmax(0, 1)`.
- **Layouts:** options `["no", "yes"]` → the decoded fed ids start with `"Context:\n"`. Options
  `["x", "y", "z"]` → they start with `"Question: "`. Decode with `m.tok.vocab` bytes joined, as
  `decodeToken` does.
- **Errors:** 1 option and 11 options → the exact error.
- **Generate is untouched:** after `Decide`, `m.cache.system` and `m.cache.last` are empty.

## Stage 3 — reuse the question and options (`cache.go`, `decide.go`)

`prefixCache` gains `decision snapshot`. For schema-first ids with `prefixLen > 0`:

- if `m.cache.decision.ids` equals `ids[:prefixLen]` and its state is set: copy that state
  (`stepper.CopyState`) and feed only `ids[prefixLen:]`;
- otherwise, feed from a fresh state, and when `prefixLen` ids have been fed, save the snapshot
  (`ids[:prefixLen]` and a copy of the state), the same way `save` does.

State-first decisions read everything and save nothing.

Test (`decide_test.go`): two schema-first `Decide` calls with the same `Text` and `Options` and
different `Context`. The second feeds exactly `len(ids2) - prefixLen` ids (count them in the fake
stepper), and its decision is the same as on a fresh model.

## Stage 4 — docs

- `README.md`: a section "Closed questions (`llm.Decider`)". Explain in plain words that a decision
  model answers by the probability of each option's letter. Include the layout rule with the
  measurement table of §0, and a 6-line example with `qwen.Config{…, DecideTemperature: 1.03}` and
  `model.Decide`.
- `AGENTS.md`: list `decide_test.go`.

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | `decide.go`, `model.go` | builds; `var _ llm.Decider` |
| 2 | `decide_test.go` | readout, temperature, layout, error tests pass |
| 3 | `cache.go`, `decide.go` | prefix reuse test passes |
| 4 | `README.md`, `AGENTS.md` | documented |
| all | — | `gotest` green; `GOOS=js GOARCH=wasm go build ./...` |
