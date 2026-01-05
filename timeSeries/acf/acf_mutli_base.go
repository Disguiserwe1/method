// n段agg序列，自相关+gamma估计
// 多段自相关函数为:
//
//	                         ∑∑(εt - μ)⋅(εt+τ - μ)
//		                 C(τ) = —————————————————————
//								   σ**2⋅∑j(T - τ)
//
// 按lag把多段、同数据分布的segment合并计算autoCorr
package acf

import (
	"fmt"
	"math"
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"
	"runtime"
	"strategyCrypto/pkg/utils/myTools"
	"sync"
)

type MultiSegments struct {
	eps      [][]float64 // 分段样本
	totalN   int         // 样本长度
	mean     float64     // 全局均值
	variance float64     // 全局方差
}

func NewMultiSeg(epsSegments [][]float64) (*MultiSegments, error) {
	// 1) 校验数据
	if len(epsSegments) == 0 {
		return nil, errorx.New(errCode.EMPTY_VALUE, "segments is empty")
	}

	N := 0
	allSegments := make([]float64, 0, len(epsSegments)*len(epsSegments[0]))
	for _, seg := range epsSegments {
		N += len(seg)
		allSegments = append(allSegments, seg...)
	}
	if N == 0 {
		return nil, errorx.New(errCode.EMPTY_VALUE, "all segments is empty")
	}
	// 全局均值
	meanValue := myTools.ArrMean(allSegments)
	// 全局方差
	varValue := myTools.WelfordVariancePopulation(allSegments)
	if varValue == 0 {
		return nil, errorx.New(errCode.INVALID_VALUE, "variance is zero")
	}
	return &MultiSegments{eps: epsSegments, totalN: N, mean: meanValue, variance: varValue}, nil
}

// 计算segments的自相关函数
func (s *MultiSegments) AutoCorrSegments(maxLag int) ([]float64, error) {

	if maxLag <= 0 {
		return nil, errorx.New(errCode.INVALID_VALUE, "maxLag must be > 0")
	}

	acf := make([]float64, maxLag)
	mean := s.mean
	variance := s.variance
	eps := s.eps

	// 逐 lag 计算自相关
	for k := 0; k < maxLag; k++ {
		num := 0.0
		cnt := 0

		// 遍历 segment
		for _, seg := range eps {
			n := len(seg)
			nk := n - k
			if nk <= 0 {
				continue
			}

			// 避免重复 len(seg)-k，推到循环外
			// 并减少 bounds checking
			segk := seg[k:]  // t+k
			seg0 := seg[:nk] // t

			for i := 0; i < nk; i++ {
				dx := seg0[i] - mean
				dy := segk[i] - mean
				num += dx * dy
			}

			cnt += nk
		}

		if cnt == 0 {
			for j := k; j < maxLag; j++ {
				acf[j] = math.NaN()
			}
			break
		}

		acf[k] = num / (variance * float64(cnt))
	}

	return acf, nil
}

func (s *MultiSegments) AutoCorrSegmentsParallel(maxLag int) ([]float64, error) {
	if maxLag <= 0 {
		return nil, fmt.Errorf("maxLag must be > 0")
	}

	mean := s.mean
	variance := s.variance
	eps := s.eps

	// acf := make([]float64, maxLag)

	// CPU 核心数
	numWorkers := runtime.NumCPU()
	wg := sync.WaitGroup{}
	tasks := make(chan int, maxLag)

	// 每个 worker 用独立 buffer 避免 false sharing
	results := make([]float64, maxLag)

	// worker
	worker := func() {
		defer wg.Done()
		for k := range tasks {
			num := 0.0
			cnt := 0

			for _, seg := range eps {
				n := len(seg)
				nk := n - k
				if nk <= 0 {
					continue
				}

				segk := seg[k:]
				seg0 := seg[:nk]

				for i := 0; i < nk; i++ {
					dx := seg0[i] - mean
					dy := segk[i] - mean
					num += dx * dy
				}
				cnt += nk
			}

			if cnt == 0 {
				results[k] = math.NaN()
			} else {
				results[k] = num / (variance * float64(cnt))
			}
		}
	}

	// 启动 worker
	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go worker()
	}

	// 分发任务
	for k := 0; k < maxLag; k++ {
		tasks <- k
	}
	close(tasks)

	// 等待
	wg.Wait()

	return results, nil
}

// 计算segments的后续符号比重
func (s *MultiSegments) GetSegmentsSignalWeight(sumQty float64) ([]float64, []float64, error) {
	if len(s.eps) == 0 {
		return nil, nil, errorx.New(errCode.EMPTY_VALUE, "eps is empty")
	}

	// 找到最长 segment
	maxLen := 0
	for _, seg := range s.eps {
		if len(seg) > maxLen {
			maxLen = len(seg)
		}
	}
	if maxLen == 0 {
		return nil, nil, errorx.New(errCode.EMPTY_VALUE, "all segments are empty")
	}

	N := len(s.eps)
	out := make([]float64, maxLen)
	outNegQty := make([]float64, maxLen)
	for _, seg := range s.eps {
		count := 0.0
		countPos := 0.0
		// signalWeight := make([]float64, 0, len(seg))
		for i, v := range seg {
			count += math.Abs(v)
			if v > 0 {
				countPos += v
			}
			out[i] += countPos / count
			outNegQty[i] += (count - 2*countPos) / sumQty
			// signalWeight = append(signalWeight, float64(countPos)/float64(countN))
		}
	}
	for i, sig := range out {
		out[i] = sig / float64(N)
		outNegQty[i] = outNegQty[i] / float64(N)
	}
	return out, outNegQty, nil
}
