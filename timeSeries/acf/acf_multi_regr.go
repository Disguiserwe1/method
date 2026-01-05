package acf

import (
	"fmt"
	"math"
	"method/ml/ols"
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"
)

// 拟合（ACF ~ lag^{-gamma}）
func FitLogACF(acf []float64, minPoints int) (gamma, intercept, r2 float64, model ols.MultiLinearModel, err error) {
	n := len(acf)
	if n < 3 {
		return math.NaN(), math.NaN(), math.NaN(), ols.MultiLinearModel{}, errorx.New(errCode.INVALID_VALUE, "fitStart/fitEnd 不合法")
	}

	// ---------------------------
	// 1. 找到 ACF>0 的最大连续区间
	// ---------------------------
	start := -1
	for i := 1; i < n; i++ {
		if acf[i] > 0 && !math.IsNaN(acf[i]) {
			start = i
			break
		}
	}
	if start == -1 {
		return math.NaN(), math.NaN(), math.NaN(), ols.MultiLinearModel{}, errorx.New(errCode.INVALID_VALUE, "ACF 没有正值点")
	}

	end := start
	for end < n && acf[end] > 0 && !math.IsNaN(acf[end]) {
		end++
	}

	length := end - start
	if length < minPoints {
		return math.NaN(), math.NaN(), math.NaN(), ols.MultiLinearModel{},
			errorx.New(errCode.INVALID_VALUE, fmt.Sprintf("有效 ACF 正区间太短 (%d)，需要至少 %d 点用于回归", length, minPoints))
	}

	// ---------------------------
	// 2. 构造 log-log 回归数据
	//    X = log(lag)
	//    Y = log(acf[lag])
	// ---------------------------
	X := make([][]float64, 0, length)
	Y := make([]float64, 0, length)

	for lag := start; lag < end; lag++ {
		c := acf[lag]
		if c <= 0 || math.IsNaN(c) {
			continue // 跳过无效
		}
		X = append(X, []float64{math.Log(float64(lag))})
		Y = append(Y, math.Log(c))
	}

	if len(X) < minPoints {
		return math.NaN(), math.NaN(), math.NaN(), ols.MultiLinearModel{},
			errorx.New(errCode.INVALID_VALUE, fmt.Sprintf("有效 log-log 拟合点不足：只有 %d 个, 需要 >= %d", len(X), minPoints))
	}

	// ---------------------------
	// 3. 执行线性回归 log(C) = a + b log(lag)
	// ---------------------------
	model, err = ols.MultiRegression(X, Y, true)
	if err != nil {
		return math.NaN(), math.NaN(), math.NaN(), ols.MultiLinearModel{},
			errorx.New(errCode.INVALID_VALUE, fmt.Sprintf("拟合失败: %v", err))
	}

	if len(model.Coeffs) < 2 {
		return math.NaN(), math.NaN(), math.NaN(), ols.MultiLinearModel{},
			errorx.New(errCode.INVALID_VALUE, fmt.Sprintf("回归无 slope 项, Coeffs=%v", model.Coeffs))
	}
	intercept = model.Coeffs[0]
	gamma = -model.Coeffs[1]
	r2 = model.RSquared

	return gamma, intercept, r2, model, nil
}
