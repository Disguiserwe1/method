package ols

import (
	"math"
	"strategyCrypto/pkg/utils/myTools"
)

// 定义线性回归模型的参数
type LinearRegressionModel struct {
	Slope     float64
	Intercept float64
}

// Regression 返回ols斜率项和截距项
func Regression(x, y []float64) LinearRegressionModel {
	if maskX, maskY, ok := paramsValidate(x, y); ok {
		n := len(maskX)
		m := (myTools.DotProduct(maskX, maskY) - float64(n)*myTools.ArrMean(maskX)*myTools.ArrMean(maskY)) / (myTools.DotProduct(maskX, maskX) - float64(n)*math.Pow(myTools.ArrMean(maskX), 2))
		b := myTools.ArrMean(maskY) - m*myTools.ArrMean(maskX)
		return LinearRegressionModel{Slope: m, Intercept: b}
	} else {
		return LinearRegressionModel{Slope: math.NaN(), Intercept: math.NaN()}
	}
}

func paramsValidate(x, y []float64) ([]float64, []float64, bool) {
	if len(x) != len(y) {
		return nil, nil, false
	}
	mask_x, mask_y := myTools.MaskIsNaNBoth(x, y)
	if len(mask_x) != len(mask_y) {
		return nil, nil, false
	}
	return mask_x, mask_y, true
}

//func main() {
//	// 示例数据
//	x := []float64{1, 2, 3, 4, 5}
//	y := []float64{2, 3, 5, 7, 11}
//
//	// 计算OLS参数
//	PredictModel := Regression(x, y)
//
//	// 打印模型参数
//	fmt.Printf("Slope (m): %v\n", PredictModel.Slope)
//	fmt.Printf("Intercept (b): %v\n", PredictModel.Intercept)
//}
