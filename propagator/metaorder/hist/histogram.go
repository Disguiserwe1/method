package hist

import "math"

// HistogramBin 每个分箱的结构
type HistogramBin struct {
	From  float64
	To    float64
	Count int
}

// Hist 按指定 bins 对 data 做分箱统计
func Hist(data []float64, bins int) []HistogramBin {
	if len(data) == 0 || bins <= 0 {
		return nil
	}

	// 1. 求最小值最大值
	minV, maxV := data[0], data[0]
	for _, v := range data {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}

	// 避免 max == min 导致除0
	if maxV == minV {
		maxV = minV + 1e-9
	}

	// 2. 分箱宽度
	width := (maxV - minV) / float64(bins)

	// 3. 初始化 bins
	result := make([]HistogramBin, bins)
	for i := 0; i < bins; i++ {
		result[i] = HistogramBin{
			From:  minV + float64(i)*width,
			To:    minV + float64(i+1)*width,
			Count: 0,
		}
	}

	// 4. 遍历数据并统计
	for _, v := range data {
		idx := int(math.Floor((v - minV) / width))
		if idx == bins { // 处理 v == maxV 的边界
			idx = bins - 1
		}
		result[idx].Count++
	}

	return result
}
