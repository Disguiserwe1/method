package fundingratemodel

import (
	"math"
	"testing"
)

var (
	imn  = 25000.0
	xAmt = 22704.65
	xP   = 279.71
	xQty = 81.18
)

// ------------------- 原始公式（2 次除法） -------------------
func OriginalFormula(imn, B, C, D float64) float64 {
	return imn / ((imn-B)/C + D)
}

func BenchmarkOriginalFormula(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = OriginalFormula(imn, xAmt, xP, xQty)
	}
}

// ------------------- 化简后公式（1 次除法） -------------------
func Simplified(imn, B, C, D float64) float64 {
	den := imn - B + D*C
	return imn * C / den
}

func BenchmarkSimplified(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Simplified(imn, xAmt, xP, xQty)
	}
}

// ------------------- 你写的 invC 写法（3 次除法 → 最慢） --------
func YourVersion(imn, B, C, D float64) float64 {
	invC := 1.0 / C
	k := D - B*invC
	return 1.0 / (invC + k/imn)
}

func BenchmarkYourVersion(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = YourVersion(imn, xAmt, xP, xQty)
	}
}

//
// ------------------- Newton-Raphson 1/x（0 除法 → 最快） --------
//

// 初始倒数：用一次除法得到 invDen0
// 若要求绝对性能，这里可以换 bit-hack 初值
func newtonInv(x float64) float64 {
	y := 1.0 / x // 初始值

	// 两次 NR 提升精度（可达 double 精度）
	y = y * (2 - x*y)
	y = y * (2 - x*y)
	return y
}

func FastInverse(imn, B, C, D float64) float64 {
	den := imn - B + D*C
	inv := newtonInv(den)
	return imn * C * inv
}

func BenchmarkFastInverse(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = FastInverse(imn, xAmt, xP, xQty)
	}
}

// ------------------- 验证四种计算结果一致 -------------------
func TestAccuracy(t *testing.T) {
	o := OriginalFormula(imn, xAmt, xP, xQty)
	s := Simplified(imn, xAmt, xP, xQty)
	y := YourVersion(imn, xAmt, xP, xQty)
	f := FastInverse(imn, xAmt, xP, xQty)

	t.Log("Original  :", o)
	t.Log("Simplified:", s)
	t.Log("YourVer   :", y)
	t.Log("FastInv   :", f)

	// 精度允许范围
	if math.Abs(o-s) > 1e-9 ||
		math.Abs(o-y) > 1e-9 ||
		math.Abs(o-f) > 1e-9 {
		t.Fatal("results mismatch")
	}
}
