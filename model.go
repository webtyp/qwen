package qwen

import (
	"sync"

	"webtyp.com/decoder"
	"webtyp.com/fmt"
	"webtyp.com/tokenizer"
	"webtyp.com/weights"
)

// Config configures a Qwen model instance.
type Config struct {
	Weights           *weights.Artifact // from webtyp/weightsc -quant int8-block32 -prefix model.language_model.; its Tokenizer.Vocab is the vocabulary
	Merges            []byte            // the companion .merges file (one "left right" pair per line, rank order)
	Decoder           decoder.Config    // the checkpoint's shape; Qwen35_08B for the 0.8B model
	DecideTemperature float64           // the decision temperature (decider-0.8b: 1.03); 0 means 1
}

// Qwen35_08B is the shape of Qwen3.5-0.8B, from the text_config of its config.json:
// 24 layers where every fourth is full attention (18 Gated DeltaNet + 6 gated attention),
// grouped-query attention 8/2 with 256-wide heads, RoPE on the first 64 dims
// (partial_rotary_factor 0.25) with theta 1e7.
var Qwen35_08B = decoder.Config{
	Arch:             decoder.Qwen35,
	Vocab:            248320,
	Hidden:           1024,
	Intermediate:     3584,
	Layers:           qwen35Layers(24, 4),
	Heads:            8,
	KVHeads:          2,
	HeadDim:          256,
	RotaryDim:        64,
	RopeTheta:        10000000,
	LinearKeyHeads:   16,
	LinearValueHeads: 16,
	LinearKeyDim:     128,
	LinearValueDim:   128,
	ConvKernel:       4,
	Eps:              1e-6,
}

// qwen35Layers is config.json's layer_types: layer i is full attention when (i+1) % interval == 0.
func qwen35Layers(n, interval int) []decoder.LayerKind {
	layers := make([]decoder.LayerKind, n)
	for i := range layers {
		if (i+1)%interval == 0 {
			layers[i] = decoder.FullAttention
		}
	}
	return layers
}

// stepper abstracts model state stepping for generation loops and testing with fakes.
type stepper interface {
	NewState() any
	Step(state any, token int, logits []float32) error
	// CopyState makes dst the same sequence state as src.
	CopyState(dst, src any) error
}

// realStepper implements stepper wrapping decoder.Model and decoder.State.
type realStepper struct {
	model *decoder.Model
}

func (s *realStepper) NewState() any {
	return s.model.NewState()
}

func (s *realStepper) CopyState(dst, src any) error {
	return dst.(*decoder.State).CopyFrom(src.(*decoder.State))
}

func (s *realStepper) Step(state any, token int, logits []float32) error {
	ds := state.(*decoder.State)
	return s.model.Step(ds, token, logits)
}

// Model represents the Qwen3.5 language model.
type Model struct {
	cfg     Config
	stepper stepper
	tok     *qwenTokenizer
	bpe     *tokenizer.BPE

	mu        sync.Mutex  // one Generate at a time: they share the cache
	cache     prefixCache // what was read last (cache.go)
	newlineID int32       // the token id of "\n", for finding where messages end
	letterIDs []int32     // token ids for letters A through J
}

// setStepper replaces the stepper and forgets the cache, whose states belong to the old one.
func (m *Model) setStepper(s stepper) {
	m.stepper = s
	m.cache.reset()
}

type qwenTokenizer struct {
	vocab [][]byte
}

const (
	errWeightsRequired = "qwen: Config.Weights is required"
	errVocabRequired   = "qwen: Config.Weights has no tokenizer vocabulary"
	errMergesRequired  = "qwen: Config.Merges is required"
	weightsPrefix      = "model.language_model."
)

// New creates a Qwen model from its weights artifact and merges.
func New(cfg Config) (*Model, error) {
	if cfg.Weights == nil {
		return nil, fmt.Err(errWeightsRequired)
	}
	if len(cfg.Weights.Tokenizer.Vocab) == 0 {
		return nil, fmt.Err(errVocabRequired)
	}
	dec, err := decoder.New(cfg.Decoder, cfg.Weights, weightsPrefix)
	if err != nil {
		return nil, err
	}
	return newModel(cfg, cfg.Weights.Tokenizer.Vocab, &realStepper{model: dec})
}

// newModel builds the tokenizer and the grammar's view of the vocabulary around a stepper.
// New passes the real decoder; tests pass a scripted one.
func newModel(cfg Config, vocab []string, s stepper) (*Model, error) {
	if len(cfg.Merges) == 0 {
		return nil, fmt.Err(errMergesRequired)
	}
	bpe, err := tokenizer.New(tokenizer.Config{
		Scheme: tokenizer.QwenScheme{},
		Vocab:  vocab,
		Merges: splitLines(string(cfg.Merges)),
	})
	if err != nil {
		return nil, err
	}
	// The grammar matches the bytes a token writes, not its vocabulary spelling: byte-level
	// BPE stores "\n" as "Ċ" and " " as "Ġ".
	decoded := make([][]byte, len(vocab))
	for i, v := range vocab {
		decoded[i] = tokenizer.QwenScheme{}.DecodeToken(nil, v)
	}
	// Qwen's vocabulary has one token for "\n" (Ċ); without it, message ends cannot be found and
	// the prefix cache stays empty (every request is read in full, which is only slower).
	newlineID := int32(-1)
	if nl := bpe.EncodeOrdinary(nil, "\n"); len(nl) == 1 {
		newlineID = nl[0]
	}
	letterIDs := make([]int32, 10)
	for i := 0; i < 10; i++ {
		letter := string(rune('A' + i))
		enc := bpe.EncodeOrdinary(nil, letter)
		if len(enc) == 1 {
			letterIDs[i] = enc[0]
		} else {
			letterIDs[i] = -1
		}
	}
	return &Model{cfg: cfg, stepper: s, tok: &qwenTokenizer{vocab: decoded}, bpe: bpe, newlineID: newlineID, letterIDs: letterIDs}, nil
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			line := s[start:i]
			if len(line) > 0 && line[len(line)-1] == '\r' {
				line = line[:len(line)-1]
			}
			lines = append(lines, line)
			start = i + 1
		}
	}
	if start < len(s) {
		line := s[start:]
		if len(line) > 0 && line[len(line)-1] == '\r' {
			line = line[:len(line)-1]
		}
		lines = append(lines, line)
	}
	return lines
}
