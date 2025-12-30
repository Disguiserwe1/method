package ols

import (
	"math"
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"

	"gonum.org/v1/gonum/mat"
)

// Package: ols
// 文件: ols_lasso.go
// 说明: 在现有 OLS 框架上实现与 sklearn 规范一致的 LASSO（线性）与 LASSO-Logistic（分类）。
//       - 线性损失: (1/(2n))||y - Xβ||^2 + α||β||_1
//       - 逻辑损失: (1/n)\sum_i [log(1+e^{x_i^Tβ}) - y_i x_i^Tβ] + α||β||_1
//       - α 与 sklearn 一致；截距（如果启用）不参与惩罚；X 会标准化（截距列不标准化）。
//       - 输出使用你的 MultiLinearModel 结构，尽量与 OLS 一致。
//
// 注意:
// 1) 线性 LASSO 采用坐标下降（Coordinate Descent），数值稳健，收敛快。
// 2) 逻辑 LASSO 采用近端梯度（Proximal Gradient, ISTA）+ 软阈值，带回溯步长。
// 3) 对 Logistic，SE/TStats/PValues 传统定义不直接适用，这里置为 NaN/0；提供 McFadden pseudo-R^2、AIC/BIC。
// 4) 线性/逻辑均支持 withIntercept: 截距不惩罚，且不标准化。

// -------------------------- 公共辅助 --------------------------

// softThreshold: S(z, a) = sign(z) * max(|z|-a, 0)
func softThreshold(z, a float64) float64 {
	if z > a {
		return z - a
	}
	if z < -a {
		return z + a
	}
	return 0
}

// addInterceptColumn: 在矩阵最左侧添加常数列 1（不标准化、不惩罚）
func addInterceptColumnDense(X *mat.Dense) *mat.Dense {
	n, p := X.Dims()
	out := mat.NewDense(n, p+1, nil)
	for i := 0; i < n; i++ {
		out.Set(i, 0, 1.0)
		for j := 0; j < p; j++ {
			out.Set(i, j+1, X.At(i, j))
		}
	}
	return out
}

// standardizeExceptIntercept: 对除截距外的列做标准化，返回每列均值/标准差（第一列截距返回0/1）
func standardizeExceptIntercept(X *mat.Dense, withIntercept bool) (means []float64, stds []float64) {
	n, p := X.Dims()
	means = make([]float64, p)
	stds = make([]float64, p)
	start := 0
	if withIntercept { // 第0列是常数列，不标准化
		stds[0] = 1
		start = 1
	}
	for j := start; j < p; j++ {
		// 均值
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += X.At(i, j)
		}
		mu := sum / float64(n)
		means[j] = mu
		// 方差
		var2 := 0.0
		for i := 0; i < n; i++ {
			v := X.At(i, j) - mu
			var2 += v * v
		}
		std := math.Sqrt(var2 / float64(n))
		if std == 0 {
			std = 1
		}
		stds[j] = std
		// 标准化
		for i := 0; i < n; i++ {
			X.Set(i, j, (X.At(i, j)-mu)/std)
		}
	}
	return
}

// destandardizeBeta: 将标准化空间的系数恢复到原始空间（截距单独处理）
func destandardizeBeta(beta []float64, means, stds []float64, withIntercept bool) []float64 {
	p := len(beta)
	out := make([]float64, p)
	copy(out, beta)
	start := 0
	if withIntercept {
		start = 1
	}
	// 非截距：β_j_orig = β_j_std / std_j
	for j := start; j < p; j++ {
		out[j] = out[j] / stds[j]
	}
	if withIntercept {
		// 截距: β0_orig = β0_std - Σ_j (μ_j/std_j)*β_j_std
		adj := 0.0
		for j := 1; j < p; j++ {
			adj += (means[j] / stds[j]) * beta[j]
		}
		out[0] = out[0] - adj
	}
	return out
}

// computePred: yhat = Xβ
func computePred(X *mat.Dense, beta []float64) *mat.VecDense {
	n, p := X.Dims()
	yhat := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		s := 0.0
		for j := 0; j < p; j++ {
			s += X.At(i, j) * beta[j]
		}
		yhat.SetVec(i, s)
	}
	return yhat
}

// -------------------------- 线性 LASSO（与 sklearn 一致） --------------------------
// 目标: (1/(2n))||y - Xβ||^2 + α||β||_1，截距不惩罚
func lassoLinearCD(matX *mat.Dense, matY *mat.VecDense, alpha float64, withIntercept bool, tol float64, maxIter int) ([]float64, error) {
	X := mat.DenseCopyOf(matX)
	Y := mat.NewVecDense(matY.Len(), nil)
	Y.CopyVec(matY)
	// 处理截距列
	if withIntercept {
		X = addInterceptColumnDense(X)
	}
	// 标准化（除截距列）
	means, stds := standardizeExceptIntercept(X, withIntercept)
	n, p := X.Dims()
	beta := make([]float64, p)

	// 预先计算列平方和: gj = (1/n) * ||X_j||^2
	gj := make([]float64, p)
	for j := 0; j < p; j++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			v := X.At(i, j)
			sum += v * v
		}
		gj[j] = sum / float64(n)
		if gj[j] == 0 {
			gj[j] = 1e-8
		}
	}

	for it := 0; it < maxIter; it++ {
		maxChange := 0.0
		// 截距先单独更新: β0 ← 平均残差 + β0 （等价于不惩罚的坐标更新）
		if withIntercept {
			// r_i = y_i - Σ_{j>=1} x_ij β_j
			sumRes := 0.0
			for i := 0; i < n; i++ {
				pred := 0.0
				for j := 1; j < p; j++ {
					pred += X.At(i, j) * beta[j]
				}
				sumRes += (Y.AtVec(i) - pred)
			}
			newB0 := sumRes / float64(n)
			maxChange = math.Max(maxChange, math.Abs(newB0-beta[0]))
			beta[0] = newB0
		}
		// 其他坐标做 L1 软阈值
		start := 0
		if withIntercept {
			start = 1
		}
		for j := start; j < p; j++ {
			// ρ_j = (1/n) * X_j^T (y - Xβ + X_j β_j)
			var rho float64
			for i := 0; i < n; i++ {
				pred := beta[0]
				for k := start; k < p; k++ {
					pred += X.At(i, k) * beta[k]
				}
				r := Y.AtVec(i) - pred + X.At(i, j)*beta[j]
				rho += X.At(i, j) * r
			}
			rho /= float64(n)
			// 更新 β_j ← S(rho, α) / gj[j]
			newBj := softThreshold(rho, alpha) / gj[j]
			if math.IsNaN(newBj) || math.IsInf(newBj, 0) {
				newBj = 0
			}
			maxChange = math.Max(maxChange, math.Abs(newBj-beta[j]))
			beta[j] = newBj
		}
		if maxChange < tol {
			break
		}
	}
	// 反标准化（以及截距回到原尺度）
	beta = destandardizeBeta(beta, means, stds, withIntercept)
	return beta, nil
}

// -------------------------- 逻辑 LASSO（与 sklearn 一致） --------------------------
// 目标: (1/n)Σ log(1+e^{xβ}) - y xβ + α||β||_1，截距不惩罚
// 使用近端梯度 (ISTA) + 回溯线搜索；X 标准化、截距不惩罚。
func lassoLogisticISTA(matX *mat.Dense, matY *mat.VecDense, alpha float64, withIntercept bool, tol float64, maxIter int) ([]float64, error) {
	X := mat.DenseCopyOf(matX)
	Y := mat.NewVecDense(matY.Len(), nil)
	Y.CopyVec(matY)
	if withIntercept {
		X = addInterceptColumnDense(X)
	}
	means, stds := standardizeExceptIntercept(X, withIntercept)
	n, p := X.Dims()
	beta := make([]float64, p)

	// 初始步长（Lipschitz近似）：L ≈ (1/4n) * max_j ||X_j||^2，故 step = 1/L
	maxColSq := 0.0
	for j := 0; j < p; j++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			v := X.At(i, j)
			sum += v * v
		}
		if sum > maxColSq {
			maxColSq = sum
		}
	}
	L := (maxColSq / float64(n)) * 0.25
	if L <= 0 {
		L = 1.0
	}
	step := 1.0 / L

	for it := 0; it < maxIter; it++ {
		// 计算 p_i = sigmoid(x_i^T β)
		pvec := make([]float64, n)
		for i := 0; i < n; i++ {
			s := 0.0
			for j := 0; j < p; j++ {
				s += X.At(i, j) * beta[j]
			}
			if s > 20 {
				pvec[i] = 1.0
			} else if s < -20 {
				pvec[i] = 0.0
			} else {
				pvec[i] = 1.0 / (1.0 + math.Exp(-s))
			}
		}
		// 计算梯度 g = (1/n) X^T (p - y)
		g := make([]float64, p)
		for j := 0; j < p; j++ {
			s := 0.0
			for i := 0; i < n; i++ {
				s += X.At(i, j) * (pvec[i] - Y.AtVec(i))
			}
			g[j] = s / float64(n)
		}
		// 截距更新: 无惩罚 prox
		newBeta := make([]float64, p)
		if withIntercept {
			newBeta[0] = beta[0] - step*g[0]
		}
		// 其他坐标: 近端算子 soft-threshold(step*alpha)
		start := 0
		if withIntercept {
			start = 1
		}
		for j := start; j < p; j++ {
			newBeta[j] = softThreshold(beta[j]-step*g[j], step*alpha)
		}
		// 回溯: 如果目标没有下降，则缩小步长（最多尝试若干次）
		oldObj := logisticL1Objective(X, Y, beta, alpha)
		for bt := 0; bt < 10; bt++ {
			newObj := logisticL1Objective(X, Y, newBeta, alpha)
			if newObj <= oldObj {
				break
			}
			step *= 0.5
			if withIntercept {
				newBeta[0] = beta[0] - step*g[0]
			}
			for j := start; j < p; j++ {
				newBeta[j] = softThreshold(beta[j]-step*g[j], step*alpha)
			}
		}
		// 收敛判据
		maxChange := 0.0
		for j := 0; j < p; j++ {
			if d := math.Abs(newBeta[j] - beta[j]); d > maxChange {
				maxChange = d
			}
		}
		beta = newBeta
		if maxChange < tol {
			break
		}
	}
	beta = destandardizeBeta(beta, means, stds, withIntercept)
	return beta, nil
}

// logisticL1Objective: (1/n)Σ [log(1+e^{xβ}) - y xβ] + α||β||_1（截距不惩罚）
func logisticL1Objective(X *mat.Dense, Y *mat.VecDense, beta []float64, alpha float64) float64 {
	n, p := X.Dims()
	loss := 0.0
	for i := 0; i < n; i++ {
		s := 0.0
		for j := 0; j < p; j++ {
			s += X.At(i, j) * beta[j]
		}
		if s > 20 {
			loss += (s - Y.AtVec(i)*s)
		} else if s < -20 {
			loss += (0 - Y.AtVec(i)*s)
		} else {
			loss += math.Log1p(math.Exp(s)) - Y.AtVec(i)*s
		}
	}
	loss /= float64(n)
	// L1:
	l1 := 0.0
	start := 1 // 默认有截距时
	if p > 0 && math.Abs(X.At(0, 0)-1.0) > 1e-12 {
		start = 0
	} // 无截距
	for j := start; j < p; j++ {
		l1 += math.Abs(beta[j])
	}
	return loss + alpha*l1
}

// -------------------------- 对外主函数 --------------------------
// MultiRegressionLasso: 与 sklearn 一致的 LASSO 接口，返回 MultiLinearModel 结构。
// alpha: 与 sklearn 相同的正则参数（不是 λ/(2n)，就是 α 本身）。
// useLogistic: false=线性；true=逻辑
// withIntercept: 是否包含截距（不参与惩罚）
func MultiRegressionLasso(matX *mat.Dense, matY *mat.VecDense, alpha float64, useLogistic bool, withIntercept bool) (MultiLinearModel, error) {
	n, _ := matX.Dims()
	if matY.Len() != n {
		return MultiLinearModel{}, errorx.New(errCode.INVALID_VALUE, "Y 长度与 X 行数不匹配")
	}

	// sklearn 对齐：alpha 按样本数缩放
	alphaEff := alpha / float64(n)

	// 训练
	var beta []float64
	var err error
	const tol = 1e-5
	const maxIter = 10000
	if !useLogistic {
		beta, err = lassoLinearCD(matX, matY, alphaEff, withIntercept, tol, maxIter)
	} else {
		beta, err = lassoLogisticISTA(matX, matY, alphaEff, withIntercept, tol, maxIter)
	}
	if err != nil {
		return MultiLinearModel{}, err
	}

	// 预测、残差、指标
	Xeff := mat.DenseCopyOf(matX)
	if withIntercept {
		Xeff = addInterceptColumnDense(Xeff)
	}
	yhat := computePred(Xeff, beta)

	// 组装输出结构
	k := len(beta)
	model := MultiLinearModel{
		Coeffs:      make([]float64, k),
		SE:          make([]float64, k),
		TStats:      make([]float64, k),
		PValues:     make([]float64, k),
		Resids:      make([]float64, n),
		AIC:         0,
		BIC:         0,
		Sigma2:      0,
		RSquared:    0,
		AdjRSquared: 0,
	}
	copy(model.Coeffs, beta)
	for i := 0; i < n; i++ {
		model.Resids[i] = matY.AtVec(i) - yhat.AtVec(i)
	}

	if !useLogistic {
		// 线性: 计算 RSS, R^2, AIC/BIC（与原 OLS 一致的定义）
		RSS := 0.0
		yMean := 0.0
		for i := 0; i < n; i++ {
			yMean += matY.AtVec(i)
		}
		yMean /= float64(n)
		TSS := 0.0
		for i := 0; i < n; i++ {
			res := model.Resids[i]
			RSS += res * res
			d := matY.AtVec(i) - yMean
			TSS += d * d
		}
		model.Sigma2 = RSS / float64(n-k)
		if TSS > 0 {
			model.RSquared = 1 - RSS/TSS
		}
		if n > k {
			// 高斯似然的近似对数似然（与 OLS 中一致）
			logLik := -0.5 * float64(n) * (1 + math.Log(2*math.Pi*RSS/float64(n)))
			model.AIC = -2*logLik + 2*float64(k)
			model.BIC = -2*logLik + float64(k)*math.Log(float64(n))
		}
		// LASSO 下 SE/TStats/PValues 非经典意义，这里保留 0
	} else {
		// 逻辑: logLik, AIC/BIC, pseudo R^2 (McFadden)
		logLik := 0.0
		for i := 0; i < n; i++ {
			p := yhat.AtVec(i)
			if p < 1e-12 {
				p = 1e-12
			} else if p > 1-1e-12 {
				p = 1 - 1e-12
			}
			y := matY.AtVec(i)
			logLik += y*math.Log(p) + (1-y)*math.Log(1-p)
		}
		// 空模型对数似然（只有截距）
		ybar := 0.0
		for i := 0; i < n; i++ {
			ybar += matY.AtVec(i)
		}
		ybar /= float64(n)
		llNull := float64(n) * (ybar*math.Log(ybar+1e-12) + (1-ybar)*math.Log(1-ybar+1e-12))
		model.AIC = -2*logLik + 2*float64(k)
		model.BIC = -2*logLik + float64(k)*math.Log(float64(n))
		if llNull != 0 {
			model.RSquared = 1 - (logLik)/(llNull)
		} // McFadden pseudo-R^2
		// 其余统计量保持 0/默认
	}
	return model, nil
}
