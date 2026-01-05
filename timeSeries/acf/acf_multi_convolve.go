// multi 自相关 => 卷积
// 自相关公式为: C(τ) = ∑(Xt)⋅(Xt+τ)
// 卷积公式为:  (y*h)[n] = ∑(y[k])⋅(h[n-k])
// 取: y[k] = Xk; h[m] = X-m 带入卷积
//  (y*h)[n] = ∑(Xk)⋅(Xk-n)
// 自相关 C(τ) 就是 “x 和翻转后的 x 做卷积” 得到的结果

// 再用FFT对卷积计算进行加速:
// 1) 把序列 x 去均值（减掉 μ）
// 2) FFT(x) 得到 X
// 3) 自相关其实只需要 X * conj(X)（翻转 + 点乘这一块在频域等价于乘以共轭）
// 4) 对 X * conj(X) 做 IFFT，得到一整条自相关序列
// ACF=IFFT(X⋅μ)
// 复杂度 O(N⋅maxLag) => O(NlogN)
package acf

import (
	"math"
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"

	"gonum.org/v1/gonum/dsp/fourier"
)

func (s *MultiSegments) AutoCorrSegmentsFFT(maxLag int) ([]float64, error) {

	if maxLag <= 0 {
		return nil, errorx.New(errCode.INVALID_VALUE, "maxLag must be > 0")
	}

	eps := s.eps
	mean := s.mean
	variance := s.variance

	// 全局 numerator（每段 FFT ACF 的和）
	numerator := make([]float64, maxLag)

	// 全局 pair 数
	pairCnt := make([]int, maxLag)

	for _, seg := range eps {
		T := len(seg)
		if T == 0 {
			continue
		}

		// ---------- Step 1: 去均值 ----------
		// fourier.FFT 操作的是实数序列 []float64
		L := nextPow2(2 * T)      // zero-padding，避免 wrap-around
		seq := make([]float64, L) // 零填充
		for i := 0; i < T; i++ {
			seq[i] = seg[i] - mean
		}

		// ---------- Step 2: 实数 FFT ----------
		fft := fourier.NewFFT(L)
		coeff := fft.Coefficients(nil, seq) // len(coeff) = L/2 + 1

		// ---------- Step 3: 乘以共轭 => |FFT|^2 ----------
		for i, c := range coeff {
			re, im := real(c), imag(c)
			// c * conj(c) = re^2 + im^2 是纯实数
			coeff[i] = complex(re*re+im*im, 0)
		}

		// ---------- Step 4: IFFT 得到自相关 ----------
		acTime := fft.Sequence(nil, coeff) // []float64，长度 L
		// 文档说明：Coefficients 再 Sequence 会乘以长度 L
		// 所以这里要除以 L，得到真正的线性自相关和
		scale := 1.0 / float64(L)

		// acSeg[k] ≈ Σ_{t} (x_t - μ)(x_{t+k} - μ)
		maxK := maxLag
		if maxK > T {
			maxK = T
		}

		for k := 0; k < maxK; k++ {
			val := acTime[k] * scale
			numerator[k] += val
			pairCnt[k] += (T - k)
		}
	}

	// ---------- Step 5: 标准化 ⇒ ACF ----------
	acf := make([]float64, maxLag)

	for k := 0; k < maxLag; k++ {
		if pairCnt[k] == 0 {
			// 后面都没 pair 了，统一 NaN
			for j := k; j < maxLag; j++ {
				acf[j] = math.NaN()
			}
			break
		}
		acf[k] = numerator[k] / (variance * float64(pairCnt[k]))
	}

	return acf, nil
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}
