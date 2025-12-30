package ols

import (
	"fmt"
	"math"
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"
	"ofeisInfra/infra/observe/log/staticLog"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type MultiLinearModel struct {
	Coeffs      []float64 // 回归系数
	SE          []float64 // 标准误
	TStats      []float64 // t统计量
	PValues     []float64 // p值（双尾）
	Resids      []float64 // 残差
	AIC         float64
	BIC         float64
	Sigma2      float64 // 残差方差
	RSquared    float64
	AdjRSquared float64
}

func MultiRegressionMat(matX *mat.Dense, matY *mat.VecDense) (MultiLinearModel, error) {
	n, k := matX.Dims()
	// 计算 (X'X)
	var XT mat.Dense
	XT.CloneFrom(matX.T())

	var XTX mat.Dense
	XTX.Mul(&XT, matX)

	// (X'X)^(-1)
	var invXTX mat.Dense
	// err := invXTX.Inverse(&XTX)
	// if err != nil {
	// 	return MultiLinearModel{Coeffs: nil, Intercept: math.NaN()}, errorx.New(errCode.CodeInvalidValue, "矩阵不可逆，请检查自变量是否共线", "")
	// }

	err := invXTX.Inverse(&XTX) // 求逆矩阵消耗太大
	if err != nil {
		// staticLog.Log.Infof("warning XTX矩阵不可逆 %s", err)
		pinv, errSVD := pseudoInverse(&XTX)
		if errSVD != nil {
			return MultiLinearModel{}, errSVD
		}
		invXTX.CloneFrom(pinv)
	}

	// (X'Y)
	var XTY mat.VecDense
	XTY.MulVec(&XT, matY)

	// β = (X'X)^(-1) * (X'Y)
	var beta mat.VecDense
	beta.MulVec(&invXTX, &XTY)

	// 预测值 & 残差
	Yhat := mat.NewVecDense(n, nil)
	Yhat.MulVec(matX, &beta)
	resid := mat.NewVecDense(n, nil)
	resid.SubVec(matY, Yhat)

	// RSS
	RSS := mat.Dot(resid, resid)

	// 残差方差 σ² = RSS / (n - k)
	sigma2 := RSS / float64(n-k)

	// 标准误 SE = sqrt( diag(σ² * (X'X)^(-1)) )
	SE := make([]float64, k)
	for i := 0; i < k; i++ {
		SE[i] = math.Sqrt(sigma2 * invXTX.At(i, i))
	}

	// t统计量
	tStats := make([]float64, k)
	for i := 0; i < k; i++ {
		tStats[i] = beta.AtVec(i) / SE[i]
	}

	// p值（双尾），使用 Student-t 分布
	df := float64(n - k)
	if df <= 0 {
		return MultiLinearModel{}, errorx.New(errCode.INVALID_VALUE, fmt.Sprintf("自由度 df=%v 非法：样本数 n 必须大于参数数 k", df))
	}

	tdist := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: df}

	pValues := make([]float64, k)
	for i := 0; i < k; i++ {
		t := math.Abs(tStats[i])
		p := 2 * tdist.Survival(t)
		pValues[i] = p
	}

	// R² & 调整后R²
	Ymean := mat.Sum(matY) / float64(n)
	TSS := 0.0
	for i := 0; i < n; i++ {
		diff := matY.AtVec(i) - Ymean
		TSS += diff * diff
	}
	RSq := 1 - RSS/TSS
	AdjRSq := 1 - (1-RSq)*float64(n-1)/float64(n-k)

	// AIC / BIC
	logLik := -0.5 * float64(n) * (1 + math.Log(2*math.Pi*RSS/float64(n)))
	AIC := -2*logLik + 2*float64(k)
	BIC := -2*logLik + float64(k)*math.Log(float64(n))

	// 提取 β
	coeffs := make([]float64, k)
	for i := 0; i < k; i++ {
		coeffs[i] = beta.AtVec(i)
	}
	return MultiLinearModel{
		Coeffs:      coeffs,
		SE:          SE,
		TStats:      tStats,
		PValues:     pValues,
		Resids:      resid.RawVector().Data,
		AIC:         AIC,
		BIC:         BIC,
		Sigma2:      sigma2,
		RSquared:    RSq,
		AdjRSquared: AdjRSq,
	}, nil
}

func MultiRegression(X [][]float64, Y []float64, withConst bool) (MultiLinearModel, error) {
	n := len(Y)
	if n == 0 || len(X) == 0 {
		return MultiLinearModel{}, errorx.New(errCode.EMPTY_VALUE, "输入数据为空")
	}

	if withConst {
		X = addConstantColumn(X)
	}

	k := len(X[0])
	if n != len(X) {
		return MultiLinearModel{}, errorx.New(errCode.INVALID_VALUE, "数据长度不匹配")
	}

	// 转成矩阵
	dataX := make([]float64, n*k)
	for i := 0; i < n; i++ {
		copy(dataX[i*k:(i+1)*k], X[i])
	}
	matX := mat.NewDense(n, k, dataX)
	matY := mat.NewVecDense(n, Y)

	// 计算 (X'X)
	var XT mat.Dense
	XT.CloneFrom(matX.T())

	var XTX mat.Dense
	XTX.Mul(&XT, matX)

	// (X'X)^(-1)
	var invXTX mat.Dense
	// err := invXTX.Inverse(&XTX)
	// if err != nil {
	// 	return MultiLinearModel{Coeffs: nil, Intercept: math.NaN()}, errorx.New(errCode.CodeInvalidValue, "矩阵不可逆，请检查自变量是否共线", "")
	// }

	err := invXTX.Inverse(&XTX)
	if err != nil {
		staticLog.Log.Infof("warning XTX矩阵不可逆 %s", err)
		pinv, errSVD := pseudoInverse(&XTX)
		if errSVD != nil {
			return MultiLinearModel{}, errSVD
		}
		invXTX.CloneFrom(pinv)
	}

	// (X'Y)
	var XTY mat.VecDense
	XTY.MulVec(&XT, matY)

	// β = (X'X)^(-1) * (X'Y)
	var beta mat.VecDense
	beta.MulVec(&invXTX, &XTY)

	// 预测值 & 残差
	Yhat := mat.NewVecDense(n, nil)
	Yhat.MulVec(matX, &beta)
	resid := mat.NewVecDense(n, nil)
	resid.SubVec(matY, Yhat)

	// RSS
	RSS := mat.Dot(resid, resid)

	// 残差方差 σ² = RSS / (n - k)
	sigma2 := RSS / float64(n-k)

	// 标准误 SE = sqrt( diag(σ² * (X'X)^(-1)) )
	SE := make([]float64, k)
	for i := 0; i < k; i++ {
		SE[i] = math.Sqrt(sigma2 * invXTX.At(i, i))
	}

	// t统计量
	tStats := make([]float64, k)
	for i := 0; i < k; i++ {
		tStats[i] = beta.AtVec(i) / SE[i]
	}

	// p值（双尾），使用 Student-t 分布
	df := float64(n - k)
	if df <= 0 {
		return MultiLinearModel{}, errorx.New(errCode.INVALID_VALUE, fmt.Sprintf("自由度 df=%v 非法：样本数 n 必须大于参数数 k", df))
	}

	tdist := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: df}

	pValues := make([]float64, k)
	for i := 0; i < k; i++ {
		t := math.Abs(tStats[i])
		p := 2 * tdist.Survival(t)
		pValues[i] = p
	}

	// R² & 调整后R²
	Ymean := mat.Sum(matY) / float64(n)
	TSS := 0.0
	for i := 0; i < n; i++ {
		diff := matY.AtVec(i) - Ymean
		TSS += diff * diff
	}
	RSq := 1 - RSS/TSS
	AdjRSq := 1 - (1-RSq)*float64(n-1)/float64(n-k)

	// AIC / BIC
	logLik := -0.5 * float64(n) * (1 + math.Log(2*math.Pi*RSS/float64(n)))
	AIC := -2*logLik + 2*float64(k)
	BIC := -2*logLik + float64(k)*math.Log(float64(n))

	// 提取 β
	coeffs := make([]float64, k)
	for i := 0; i < k; i++ {
		coeffs[i] = beta.AtVec(i)
	}
	return MultiLinearModel{
		Coeffs:      coeffs,
		SE:          SE,
		TStats:      tStats,
		PValues:     pValues,
		Resids:      resid.RawVector().Data,
		AIC:         AIC,
		BIC:         BIC,
		Sigma2:      sigma2,
		RSquared:    RSq,
		AdjRSquared: AdjRSq,
	}, nil
}

// 用SVD 求解广义逆矩阵
func pseudoInverse(A *mat.Dense) (*mat.Dense, error) {
	var svd mat.SVD
	ok := svd.Factorize(A, mat.SVDThin)
	if !ok {
		return nil, errorx.New(errCode.INVALID_VALUE, "SVD分解失败")
	}

	// 提取 U, Σ, Vᵀ
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)

	// 取 Σ 的倒数
	sigma := svd.Values(nil)
	m, n := A.Dims()
	sInv := mat.NewDense(n, m, nil)

	tol := 1e-12 // 小奇异值截断阈值
	for i, val := range sigma {
		if val > tol {
			sInv.Set(i, i, 1.0/val)
		}
	}

	// 计算伪逆 A⁺ = V * Σ⁺ * Uᵀ
	var temp mat.Dense
	temp.Mul(&v, sInv)
	var uT mat.Dense
	uT.CloneFrom(u.T())

	var pinv mat.Dense
	pinv.Mul(&temp, &uT)

	return &pinv, nil
}

// 添加常数项
func addConstantColumn(X [][]float64) [][]float64 {
	n := len(X)
	if n == 0 {
		return X
	}
	k := len(X[0])

	// 新矩阵 n × (k+1)
	newX := make([][]float64, n)
	for i := 0; i < n; i++ {
		newRow := make([]float64, k+1)
		newRow[0] = 1.0 // 第一列为常数项
		copy(newRow[1:], X[i])
		newX[i] = newRow
	}
	return newX
}
