package npCorr

import (
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"
)

func Correlate(a, v []float64, mode CORRELATE_MODE) ([]float64, error) {
	n, m := len(a), len(v)
	if n <= 2 || m <= 2 {
		return nil, errorx.New(errCode.INVALID_VALUE, "input length is not enough")
	}
	if mode != FULL_MODE && mode != VALID_MODE && mode != SAME_MODE {
		return nil, errorx.New(errCode.INVALID_VALUE, "invalid mode, expected 'full', 'same' or 'valid'")
	}
	if mode == VALID_MODE && m > n {
		return []float64{}, errorx.New(errCode.INVALID_VALUE, "np.correlate([1,2],[1,2,3],mode='valid') → []")
	}
	var outLen int
	switch mode {
	case FULL_MODE:
		outLen = n + m - 1
	case VALID_MODE:
		outLen = n
	case SAME_MODE:
		outLen = n - m + 1
	}

	out := make([]float64, outLen)
	start := 0
	if mode == SAME_MODE {
		start = (m - 1) / 2
	} else if mode == VALID_MODE {
		start = m - 1
	}

	for i := 0; i < outLen; i++ {
		sum := 0.0
		for j := 0; j < m; j++ {
			ai := i + start - j
			if ai >= 0 && ai < n {
				sum += a[ai] * v[j]
			}
		}
		out[i] = sum
	}

	return out, nil
}

type CORRELATE_MODE uint

const (
	FULL_MODE CORRELATE_MODE = iota
	VALID_MODE
	SAME_MODE
)
