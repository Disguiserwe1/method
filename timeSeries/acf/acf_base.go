package acf

import (
	"method/numpy/npCorr"
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"
	"strategyCrypto/pkg/utils/myTools"
)

// 单一序列自相关函数
func AutoCorrSingeSegment(series []float64, maxLag int) ([]float64, error) {
	n := len(series)
	if n == 0 {
		return nil, errorx.New(errCode.EMPTY_VALUE, "input series empty")
	}
	if maxLag <= 0 {
		return nil, errorx.New(errCode.INVALID_VALUE, "maxLag must be > 0")
	}

	mean := myTools.ArrMean(series)

	// subtract mean
	u := make([]float64, n)
	for i := range series {
		u[i] = series[i] - mean
	}

	// full correlate
	acfFull, err := npCorr.Correlate(u, u, npCorr.FULL_MODE)
	if err != nil {
		return nil, err
	}

	// take positive lags: acf[n-1:]
	acf := acfFull[n-1:]
	if len(acf) > maxLag {
		acf = acf[:maxLag]
	}

	// compute variance
	var v2 float64
	for _, x := range u {
		v2 += x * x
	}
	varValue := v2 / float64(n)

	// normalize: acf[k] /= var * (n-k)
	for k := 0; k < len(acf); k++ {
		acf[k] /= varValue * float64(n-k)
	}

	return acf, nil
}
