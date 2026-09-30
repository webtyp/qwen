---
PLAN: "feat: reuse the state of the prompt's fixed prefix — each turn reads only what is new"
TAG: v0.2.0
EXECUTOR: jules
REVIEWER: none
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.

# Plan — `webtyp.com/qwen`: do not read the same prompt twice

## 0. Context

`qwen` runs Qwen3.5 in Go for the browser. `Generate` renders an `llm.Request` into prompt ids
(`generate.go`: render → encode), feeds **every** prompt id to a fresh decoder state with
`stepper.Step`, then generates. Measured with the real Qwen3.5-0.8B (native Go, 2026-09-30):
**0.53 s per token**, and reading the prompt costs the same per token as generating. An agent
turn sends the system block (identity plus every offered tool's definition, a few hundred
tokens) on every request, and the conversation grows turn after turn. So most of each request
is identical to the previous one, and reading it again costs minutes.

The ecosystem already keeps that part stable on purpose: `webtyp/agentcontext` never changes the
system block during a conversation, and user turns carry a timestamp that never changes. And
`webtyp.com/decoder` v0.3.0 added:

```go
// CopyFrom makes s the same sequence state as src: position, KV caches and recurrent states.
// The two states share no memory afterwards, and s reuses the capacity it already has.
func (s *State) CopyFrom(src *State) error
```

**The fix:** the model remembers the ids it last read and the decoder state after them. When a
new prompt starts with ids it already read, it copies the saved state and reads only the rest.

## Development rules (inline)

- Primary runtime: browser, TinyGo/WASM (`GOOS=js GOARCH=wasm go build ./...`). Never import
  `strings`, `fmt`, `errors`, `strconv` (use `webtyp.com/fmt`), `sort` or `map[K]V` in non-test
  files.
- Tests in this repository live in the root as `package qwen` (see `AGENTS.md`: they reach the
  unexported stepper). New tests follow that rule, and add their names to `AGENTS.md`.
- `go get webtyp.com/decoder@v0.3.0` first.
- No `TODO`. No behavior change except speed: a cached answer is **identical**, id for id, to an
  uncached one.
- `gotest` green.

## Design gate (api-design — five answers)

1. **Prior art.** llama.cpp's server `cache_prompt` reuses the KV cache of the longest common
   prefix with the previous request. vLLM's automatic prefix caching does the same per block.
   The Anthropic and OpenAI APIs offer prompt caching of a stable prefix. All of them key the
   cache on the exact token prefix. Qwen3.5 mixes attention with recurrent DeltaNet layers,
   whose state cannot be rewound to an arbitrary earlier position, so we keep **snapshots** at
   the ends of what we read instead of truncating a cache.
2. **Novice-name test.** No new public name. `Generate` is faster.
3. **Complexity ledger.** Concepts +0. Files +0. Call site +0. Ways +0.
4. **Where it belongs.** `qwen` owns the prompt ids and when they are fed. The copy is
   `decoder`'s (done).
5. **What it deletes.** Nothing. It adds capability.

## Stage 1 — the stepper can copy a state (`model.go`)

- `stepper` gains `CopyState(dst, src any) error`. `realStepper.CopyState` calls
  `dst.(*decoder.State).CopyFrom(src.(*decoder.State))`.
- The test fake (`scriptedStepper` in `generate_test.go`) implements it as well. Its state must
  record how many tokens were fed through it (see stage 3).

## Stage 2 — the prefix cache (`cache.go`, `generate.go`)

```go
// prefixCache is what the model last read: the prompt ids and the decoder state after them.
type prefixCache struct {
	ids   []int32
	state any // nil until the first Generate
}
```

`Model` gains `cache prefixCache` and a `mu sync.Mutex` held for the whole of `Generate`/
`GenerateStream`. One model serves one agent; the mutex makes concurrent calls safe rather than
wrong. `sync` is allowed; it compiles under TinyGo.

In `generateStreamInternal`, after encoding `promptIds`:

1. `n := commonPrefix(m.cache.ids, promptIds)`, the number of leading ids that are equal.
2. Always read at least the last prompt id, because its step produces the logits of the first
   answer token: `if n >= len(promptIds) { n = len(promptIds) - 1 }`.
3. If `n > 0` **and** `n == len(m.cache.ids)`: `st := m.stepper.NewState()`, then
   `m.stepper.CopyState(st, m.cache.state)`, and feed only `promptIds[n:]`. Otherwise (no cache,
   or the prompt diverges inside the cached ids, where a DeltaNet state cannot be rewound): a
   fresh state, feed everything.
4. After feeding the prompt, save the snapshot: `m.cache.ids = append(m.cache.ids[:0], promptIds...)`
   and copy the state into `m.cache.state`, which is allocated once and reused. Save it **before**
   generating, so the saved state is exactly "after the prompt".

Why after the whole prompt? The next request of an agent turn is this prompt plus the model's
answer, the tool results or the next user turn, so it starts with exactly these ids. When the
next request instead drops something in the middle (a compaction, or a critic retry adding a
system note), the prefix check in step 3 fails and it reads everything once. That is correct,
only not faster.

`commonPrefix(a, b []int32) int` is a small helper in `cache.go`.

## Stage 3 — tests (`cache_test.go`)

Use the scripted stepper and toy vocabulary of `generate_test.go`. The fake counts every token
fed through `Step` (prompt and generation) in a field `fed`.

- **Second turn reads only what is new:** `Generate(req1)`, then `Generate(req2)`, where
  `req2 = req1` plus the assistant's answer and a new user message. The ids fed during the second
  call equal `len(ids(req2)) - len(ids(req1))` plus the generated tokens, not `len(ids(req2))`.
  Compute the ids with `render` + the model's encoder, as `holaAfterPrompt` does.
- **Identical results:** the same two calls on a model whose cache was reset between them
  (a new model) give the same `llm.Response`.
- **Divergence reads everything:** `req3` with a different `System` → every id of `req3` is fed.
- **The cached state is not shared:** after a cached `Generate`, generating again from the same
  request gives the same answer (the saved snapshot was not advanced by the generation).
- `GOOS=js GOARCH=wasm go build ./...` still passes.

## Stage 4 — docs

`README.md`: one paragraph "Reading only what is new". It explains the cache in two sentences,
says it is automatic, and says that it helps most when the system prompt and tools stay the
same during a conversation (what `webtyp/agentcontext` guarantees).

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | `model.go`, `generate_test.go` | builds |
| 2 | `cache.go`, `generate.go` | builds; existing tests green |
| 3 | `cache_test.go`, `AGENTS.md` | new tests pass |
| 4 | `README.md` | mentions the cache |
| all | — | `gotest` green; wasm build |
