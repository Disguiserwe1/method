package npCorr

import (
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"
	"strategyCrypto/pkg/utils/myTools"
)

func Convolve(a, v []float64, mode CORRELATE_MODE) ([]float64, error) {
	n, m := len(a), len(v)
	if n <= 2 || m <= 2 {
		return nil, errorx.New(errCode.INVALID_VALUE, "input length is not enough")
	}

	// flip v (np.convolve 卷积核需要翻转)
	vRev := myTools.ReverseSliceF64(v)

	// 复用我们已经实现的 Correlate
	return Correlate(a, vRev, mode)
}
