package adfuller

import (
	"math"
	"math/rand"
	"method/ml/ols"
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"
	"strategyCrypto/pkg/utils/myTools"
	"time"

	"github.com/gonum/stat"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type ADFResult struct {
	Gamma     float64            // 单位根系数
	TStat     float64            // ADF统计量 (t值)
	PValue    float64            // 对应p值
	UsedLag   int                // 选用的滞后阶数
	NObs      int                // 有效样本量
	AIC       float64            // Akaike信息准则
	BIC       float64            // 贝叶斯信息准则
	Method    LagMode            // autolag选择方法
	Trend     string             // 趋势类型 ("nc"、"c"、"ct")
	Criticals map[string]float64 // 临界值（1%, 5%, 10%）
	Tail      string             // 左尾or右尾
	Resid     []float64          // 残差
	Coeffs    []float64          // 回归系数
}

// adf单位根检验, H0: 非平稳(存在单位根); H1: 序列平稳(无单位根)
// input: price; output: bool
// 用于右尾ADF（Explosive ADF）的数据构造
func buildExplosiveData(series []float64, lag int) (y, ylag []float64) {
	y = series[1:]
	ylag = series[:len(series)-1]
	if lag > 0 {
		y = y[lag:]
		ylag = ylag[lag:]
	}
	return
}

func diff(x []float64) []float64 {
	d := make([]float64, len(x)-1)
	for i := 1; i < len(x); i++ {
		d[i-1] = x[i] - x[i-1]
	}
	return d
}

// 构造ADF输入矩阵
func buildADFData(series []float64, lag int) (dy, ylag []float64) {
	dy = diff(series)
	ylag = series[:len(series)-1]
	if lag > 0 {
		dy = dy[lag:]
		ylag = ylag[lag:]
	}
	return dy, ylag
}

// 方差
func variance(x []float64) float64 {
	mean := myTools.ArrMean(x)
	sum := 0.0
	for _, v := range x {
		sum += math.Pow(v-mean, 2)
	}
	return sum / float64(len(x)-1)
}

// t统计量
func calcTStat(y, x []float64, gamma float64) float64 {
	n := float64(len(y))
	meanX := myTools.ArrMean(x)
	ssx := 0.0
	sse := 0.0
	for i := range y {
		yhat := gamma * x[i]
		sse += math.Pow(y[i]-yhat, 2)
		ssx += math.Pow(x[i]-meanX, 2)
	}
	se := math.Sqrt((sse / (n - 2)) / ssx)
	return gamma / se
}

// ADF检验主函数
// input: logPrice 对数价格序列; regr: 趋势类型 "n"、"c"、"ct"; maxLag: 最大滞后阶数 0 为1阶; autolag: 滞后阶数选择方法; tail: "LEFT_TAIL" or "RIGHT_TAIL"
func AdfTest(logPrice []float64, regr string, maxLag int, autolag LagMode, tail string) (ADFResult, error) {

	result := ADFResult{
		AIC:       math.Inf(1),
		BIC:       math.Inf(1),
		Tail:      tail,
		Method:    autolag,
		Criticals: make(map[string]float64),
		Resid:     make([]float64, 0),
		Trend:     regr,
	}

	dy := diff(logPrice) // len = n-1
	ylag := logPrice[:len(logPrice)-1]

	switch tail {
	case LEFT_TAIL:
		result.Criticals = adfLeftTailCriticalValues[regr]
		for lag := 0; lag <= maxLag; lag++ {
			dy1 := dy[maxLag:]
			if len(dy1) < 10 {
				continue
			}
			ylag1 := ylag[:len(ylag)-maxLag]

			// 构建endog
			nRow := len(dy1)
			var nCol int
			if regr == "c" || regr == "n" {
				nCol = lag + 2
			}
			if regr == "ct" {
				nCol = lag + 3
			}
			X := make([]float64, nRow*nCol)
			pos := 0
			for i := 0; i < nRow; i++ {
				X[pos] = ylag1[i]
				pos++
				if regr != "n" {
					X[pos] = 1
					pos++
				}
				if regr == "ct" {
					X[pos] = float64(i + 1)
					pos++
				}
				if lag != 0 {
					for j := lag; j > 0; j-- {
						X[pos] = dy[maxLag-j : len(dy)-j][i]
						pos++
					}
				}

			}

			matX := mat.NewDense(nRow, nCol, X)
			matY := mat.NewVecDense(nRow, dy1)
			model, err := ols.MultiRegressionMat(matX, matY)
			if err != nil {
				continue
			}
			gamma := model.Coeffs[0]
			tStat := model.TStats[0]
			AIC := model.AIC
			BIC := model.BIC
			nobs := nRow
			pValues := model.PValues[0]

			switch autolag {
			case LAG_MODE_AIC:
				if model.AIC < result.AIC {
					result.Gamma = gamma
					result.TStat = tStat
					result.AIC = AIC
					result.BIC = BIC
					result.UsedLag = lag
					result.NObs = nobs
					result.Resid = model.Resids
					result.PValue = pValues
					result.Coeffs = model.Coeffs
				}
			case LAG_MODE_BIC:
				if model.BIC < result.BIC {
					result.Gamma = gamma
					result.TStat = tStat
					result.AIC = AIC
					result.BIC = BIC
					result.UsedLag = lag
					result.NObs = nobs
					result.Resid = model.Resids
					result.PValue = pValues
					result.Coeffs = model.Coeffs
				}
			case LAG_MODE_TSTAT:
				if tStat < result.TStat || result.TStat == 0 {
					result.Gamma = gamma
					result.TStat = tStat
					result.AIC = AIC
					result.BIC = BIC
					result.UsedLag = lag
					result.NObs = nobs
					result.Resid = model.Resids
					result.PValue = pValues
					result.Coeffs = model.Coeffs
				}

			}
		}
	case RIGHT_TAIL:
		result.Criticals = adfRightTailCriticalValues[regr]
		for lag := 0; lag <= maxLag; lag++ {
			dy1 := dy[maxLag:]
			if len(dy1) < 10 {
				continue
			}
			ylag1 := ylag[:len(ylag)-maxLag]

			// 构建endog
			nRow := len(dy1)
			var nCol int
			if regr == "c" || regr == "n" {
				nCol = lag + 2
			}
			if regr == "ct" {
				nCol = lag + 3
			}
			X := make([]float64, nRow*nCol)
			pos := 0
			for i := 0; i < nRow; i++ {
				X[pos] = ylag1[i]
				pos++
				if regr != "n" {
					X[pos] = 1
					pos++
				}
				if regr == "ct" {
					X[pos] = float64(i + 1)
					pos++
				}
				if lag != 0 {
					for j := lag; j > 0; j-- {
						X[pos] = dy[maxLag-j : len(dy)-j][i]
						pos++
					}
				}

			}

			matX := mat.NewDense(nRow, nCol, X)
			matY := mat.NewVecDense(nRow, dy1)
			model, err := ols.MultiRegressionMat(matX, matY)
			if err != nil {
				continue
			}
			gamma := model.Coeffs[0]
			tStat := model.TStats[0]
			AIC := model.AIC
			BIC := model.BIC
			nobs := nRow
			pValues := model.PValues[0]

			switch autolag {
			case LAG_MODE_AIC:
				if model.AIC < result.AIC {
					result.Gamma = gamma
					result.TStat = tStat
					result.AIC = AIC
					result.BIC = BIC
					result.UsedLag = lag
					result.NObs = nobs
					result.Resid = model.Resids
					result.PValue = pValues
					result.Coeffs = model.Coeffs
				}
			case LAG_MODE_BIC:
				if model.BIC < result.BIC {
					result.Gamma = gamma
					result.TStat = tStat
					result.AIC = AIC
					result.BIC = BIC
					result.UsedLag = lag
					result.NObs = nobs
					result.Resid = model.Resids
					result.PValue = pValues
					result.Coeffs = model.Coeffs
				}
			case LAG_MODE_TSTAT:
				if tStat < result.TStat || result.TStat == 0 {
					result.Gamma = gamma
					result.TStat = tStat
					result.AIC = AIC
					result.BIC = BIC
					result.UsedLag = lag
					result.NObs = nobs
					result.Resid = model.Resids
					result.PValue = pValues
					result.Coeffs = model.Coeffs
				}

			}
		}
	}

	if math.IsInf(result.AIC, 1) || math.IsInf(result.BIC, 1) || result.TStat == 0 || math.IsNaN(result.TStat) {
		return result, errorx.New(errCode.INVALID_VALUE, "ADF检验失败, 可能样本量过小或数据异常")
	}

	// 输出结果判定
	// if tail == LEFT_TAIL {
	// 	if result.TStat < result.Criticals["5%"] {
	// 		fmt.Println("序列平稳（拒绝单位根假设）", result.TStat, result.Criticals["5%"])
	// 	} else {
	// 		fmt.Println("非平稳（存在单位根）", result.TStat, result.Criticals["5%"])
	// 	}
	// } else {
	// 	if result.TStat > result.Criticals["5%"] {
	// 		fmt.Println("检测到爆炸性（价格泡沫）", result.TStat, result.Criticals["5%"])
	// 	} else {
	// 		fmt.Println("无爆炸性迹象", result.TStat, result.Criticals["5%"])
	// 	}
	// }

	return result, nil
}

var adfLeftTailCriticalValues = map[string]map[string]float64{
	"n":  {"1%": -2.58, "5%": -1.95, "10%": -1.62},
	"c":  {"1%": -3.43, "5%": -2.86, "10%": -2.57},
	"ct": {"1%": -3.96, "5%": -3.41, "10%": -3.13},
}

// 右尾ADF临界值
var adfRightTailCriticalValues = map[string]map[string]float64{
	"n":  {"1%": 2.58, "5%": 1.95, "10%": 1.62},
	"c":  {"1%": 3.43, "5%": 2.86, "10%": 2.57},
	"ct": {"1%": 3.96, "5%": 3.41, "10%": 3.13},
}

type ARResult struct {
	P       int
	Coeffs  []float64
	AIC     float64
	BIC     float64
	PValues []float64
}

// 检验残差是否存在AR(p)结构，并选出最佳p
func DetectAR(resid []float64, pMax int) (bestP int, info []ARResult) {
	n := len(resid)
	results := []ARResult{}

	// 0️⃣ AR(0) 基准模型（白噪声）
	mean := stat.Mean(resid, nil)
	var ss float64
	for _, v := range resid {
		diff := v - mean
		ss += diff * diff
	}
	sigma2 := ss / float64(n)
	aic0 := float64(n)*math.Log(sigma2) + 2*1
	bic0 := float64(n)*math.Log(sigma2) + math.Log(float64(n))*1

	results = append(results, ARResult{
		P:       0,
		Coeffs:  []float64{mean},
		AIC:     aic0,
		BIC:     bic0,
		PValues: nil,
	})

	// 1️⃣ 后续 AR(p)
	bestAIC := aic0
	bestP = 0

	for p := 1; p <= pMax; p++ {
		T := n - p
		matX := mat.NewDense(T, p+1, nil)
		matY := mat.NewVecDense(T, nil)

		for t := p; t < n; t++ {
			for j := 0; j < p; j++ {
				matX.Set(t-p, j, resid[t-j-1])
			}
			matX.Set(t-p, p, 1) // 常数项
			matY.SetVec(t-p, resid[t])
		}

		model, err := ols.MultiRegressionMat(matX, matY)
		if err != nil {
			continue
		}

		results = append(results, ARResult{
			P:       p,
			Coeffs:  model.Coeffs,
			AIC:     model.AIC,
			BIC:     model.BIC,
			PValues: model.PValues,
		})

		if model.AIC < bestAIC {
			bestAIC = model.AIC
			bestP = p
		}
	}

	return bestP, results
}

// 从adf检验结果获取data的估计
func (f *ADFResult) GetEstmate() (regr string, muHat, tauHat float64) {
	switch f.Trend {
	case "n":
		muHat = 0
		tauHat = 0
	case "c":
		if len(f.Coeffs) >= 2 {
			muHat = f.Coeffs[1]
			tauHat = 0
		}
	case "ct":
		if len(f.Coeffs) >= 3 {
			muHat = f.Coeffs[1]
			tauHat = f.Coeffs[2]
		}
	}
	return f.Trend, muHat, tauHat
}

// 残差白噪声采样模拟
func SimulateWhiteNoise(resid []float64, length int, method whiteNoiseSampleMethod) ([]float64, error) {

	switch method {
	case PARAMETIC_BOOTSTRAP:
		return nil, errorx.New(errCode.INVALID_VALUE, "未知的残差采样方法")

	case NONPARAMETIC_BOOTSTRAP:
		result := make([]float64, length)
		for i := range result {
			r := rand.New(rand.NewSource(time.Now().UnixNano()))
			pos := r.Intn(len(resid))
			result[i] = resid[pos]
		}
		return result, nil

	case WILD_BOOTSTRAP:
		return nil, errorx.New(errCode.INVALID_VALUE, "未知的残差采样方法")

	default:
		return nil, errorx.New(errCode.INVALID_VALUE, "未知的残差采样方法")
	}
}

// Ljung-Box检验
// 给定时序rt，样本长度n，滞后阶数lags
// 样本自相关系数: rk = Σ((rt - rmean)(rt-k - rmean)) / Σ((rt - rmean)^2)
// Ljung-Box统计量: Q = n(n+2)Σ(rk^2/(n-k))  k=1~lags
// Q服从自由度为lags的卡方分布
// input: resid 残差序列; lags 滞后阶数; alpha 显著性水平
// output: result 是否拒绝原假设(残差白噪声); Q 统计量; pValue p值; err 错误
func LjungBoxTest(resid []float64, lags int, alpha float64) (result bool, Q float64, pValue float64, err error) {
	n := float64(len(resid))
	if n <= float64(lags) {
		return false, 0, 0, errorx.New(errCode.INVALID_VALUE, "样本量过小, 无法进行Ljung-Box检验")
	}
	rmean := myTools.ArrMean(resid)
	r := make([]float64, lags+1)
	var denom float64
	for _, v := range resid {
		denom += (v - rmean) * (v - rmean)
	}
	for k := 10; k <= lags; k++ {
		var num float64
		for t := k; t < len(resid); t++ {
			num += (resid[t] - rmean) * (resid[t-k] - rmean)
		}
		r[k] = num / denom
	}

	for k := 1; k <= lags; k++ {
		Q += r[k] * r[k] / (n - float64(k))
	}
	Q = n * (n + 2) * Q

	// 计算 p-value
	chi2 := distuv.ChiSquared{K: float64(lags)}
	pValue = 1 - chi2.CDF(Q)
	if pValue < alpha {
		result = true // 拒绝原假设, 残差存在自相关
	} else {
		result = false // 不拒绝原假设, 残差白噪声
	}
	return result, Q, pValue, nil
}
