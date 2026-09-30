package qwen

import (
	"webtyp.com/decoder"
	"webtyp.com/tokenizer"
	"webtyp.com/weights"
)

// Config configures a Qwen model instance.
type Config struct {
	Weights *weights.Artifact // from webtyp/weightsc -quant int8-block32 -prefix model.language_model.
	Vocab   []byte            // vocabulary file bytes (newline separated tokens)
	Merges  []byte            // the companion .merges file bytes (newline separated)
	Decoder decoder.Config    // the checkpoint's shape; Qwen35_08B for the 0.8B model
}

// Qwen35_08B defines the shape/config for Qwen3.5-0.8B.
var Qwen35_08B = decoder.Config{
	Vocab:            248320,
	Hidden:           1024,
	Intermediate:     3584,
	Heads:            16,
	KVHeads:          8,
	HeadDim:          128,
	RotaryDim:        64,
	RopeTheta:        1000000.0,
	LinearKeyHeads:   16,
	LinearValueHeads: 16,
	LinearKeyDim:     128,
	LinearValueDim:   128,
	ConvKernel:       4,
	Eps:              1e-6,
	Layers:           make([]decoder.LayerKind, 24),
}

// stepper abstracts model state stepping for generation loops and testing with fakes.
type stepper interface {
	NewState() any
	Step(state any, token int, logits []float32) error
}

// realStepper implements stepper wrapping decoder.Model and decoder.State.
type realStepper struct {
	model *decoder.Model
}

func (s *realStepper) NewState() any {
	return s.model.NewState()
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
}

type qwenTokenizer struct {
	vocab [][]byte
}

// New creates a new Qwen model.
func New(cfg Config) (*Model, error) {
	var decModel *decoder.Model
	if cfg.Weights != nil {
		m, err := decoder.New(cfg.Decoder, cfg.Weights, "model.language_model.")
		if err != nil {
			return nil, err
		}
		decModel = m
	}

	if len(cfg.Vocab) == 0 {
		return nil, fmtErrf("qwen: vocab is required")
	}
	if len(cfg.Merges) == 0 {
		return nil, fmtErrf("qwen: merges is required")
	}

	vocabLines := splitLines(string(cfg.Vocab))
	mergeLines := splitLines(string(cfg.Merges))

	bpe, err := tokenizer.New(tokenizer.Config{
		Scheme: tokenizer.QwenScheme{},
		Vocab:  vocabLines,
		Merges: mergeLines,
	})
	if err != nil {
		return nil, err
	}

	vocabBytes := make([][]byte, len(vocabLines))
	for i, v := range vocabLines {
		vocabBytes[i] = []byte(v)
	}

	m := &Model{
		cfg:     cfg,
		tok:     &qwenTokenizer{vocab: vocabBytes},
		bpe:     bpe,
	}
	if decModel != nil {
		m.stepper = &realStepper{model: decModel}
	}

	return m, nil
}

type customErr struct {
	msg string
}

func (e *customErr) Error() string {
	return e.msg
}

func fmtErrf(msg string) error {
	return &customErr{msg: msg}
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
