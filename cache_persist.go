package qwen

import "webtyp.com/fmt"

// decisionCacheMagic opens every saved decision cache; decisionCacheFormat is its layout version.
const (
	decisionCacheMagic  = "WTQD"
	decisionCacheFormat = 1

	errDecisionCacheFormat = "qwen: the bytes are not a saved decision cache"
	errDecisionCacheModel  = "qwen: the decision cache was saved by %s v%d, this model is %s v%d"
)

// SaveDecisionCache returns the decision cache as bytes: the question and options a closed
// question starts with (the tool list), already read by the model. Reading that prefix is the
// slowest part of the first decision (22–44 s in the browser), and it is the same every time the
// Worker starts, so the Worker keeps these bytes (in OPFS) and gives them to LoadDecisionCache.
// ok is false while no decision has filled the cache.
//
// Layout, little-endian: "WTQD", format, the weights artifact's ID (length + bytes) and version,
// the prefix token ids (count + int32 each), then the decoder state.
func (m *Model) SaveDecisionCache() (data []byte, ok bool, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	c := &m.cache.decision
	if c.state == nil {
		return nil, false, nil
	}
	state, err := m.stepper.SaveState(c.state)
	if err != nil {
		return nil, false, err
	}
	id, version := m.artifact()
	out := append([]byte{}, decisionCacheMagic...)
	out = appendUint32(out, decisionCacheFormat)
	out = appendUint32(out, uint32(len(id)))
	out = append(out, id...)
	out = appendUint32(out, version)
	out = appendUint32(out, uint32(len(c.ids)))
	for _, v := range c.ids {
		out = appendUint32(out, uint32(v))
	}
	return append(out, state...), true, nil
}

// LoadDecisionCache restores bytes from SaveDecisionCache. Bytes saved with other weights are
// refused, since their state means nothing to this model.
func (m *Model) LoadDecisionCache(data []byte) error {
	r := byteReader{data: data}
	if string(r.next(len(decisionCacheMagic))) != decisionCacheMagic || r.uint32() != decisionCacheFormat {
		return fmt.Err(errDecisionCacheFormat)
	}
	id := string(r.next(int(r.uint32())))
	version := r.uint32()
	n := int(r.uint32())
	if r.bad || n > len(data)/4 {
		return fmt.Err(errDecisionCacheFormat)
	}
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(r.uint32())
	}
	if r.bad {
		return fmt.Err(errDecisionCacheFormat)
	}
	if wantID, wantVersion := m.artifact(); id != wantID || version != wantVersion {
		return fmt.Errf(errDecisionCacheModel, id, version, wantID, wantVersion)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	st := m.stepper.NewState()
	if err := m.stepper.LoadState(st, data[r.pos:]); err != nil {
		return err
	}
	m.cache.decision = snapshot{ids: ids, state: st}
	return nil
}

// artifact names the weights, so a saved cache is only loaded by the model that saved it.
func (m *Model) artifact() (string, uint32) {
	if m.cfg.Weights == nil {
		return "", 0
	}
	return m.cfg.Weights.ID, m.cfg.Weights.Version
}

func appendUint32(b []byte, v uint32) []byte {
	return append(b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
}

type byteReader struct {
	data []byte
	pos  int
	bad  bool
}

func (r *byteReader) next(n int) []byte {
	if n < 0 || r.pos+n > len(r.data) {
		r.bad = true
		return nil
	}
	b := r.data[r.pos : r.pos+n]
	r.pos += n
	return b
}

func (r *byteReader) uint32() uint32 {
	b := r.next(4)
	if b == nil {
		return 0
	}
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}
