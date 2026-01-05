package acf

// 自动确定 log-log 的拟合区间
func AutoFitRange(acf []float64) (start, end int) {
	n := len(acf)

	// 1) 找到所有 ACF>0 的 lag（因为 log 不能取 <=0）
	valid := make([]int, 0)
	for lag := 2; lag < n; lag++ { // lag 从 2 开始
		if acf[lag] > 0 {
			valid = append(valid, lag)
		}
	}

	if len(valid) < 20 {
		// 区间太短，不够拟合
		return 2, int(float64(n) * 0.3)
	}

	// 2) 去掉前面 20% 和后面 20% —— 保留中间的 60% 为线性区
	s := valid[int(float64(len(valid))*0.2)]
	e := valid[int(float64(len(valid))*0.8)]

	if e <= s+5 {
		e = s + 5
	}
	if e >= n {
		e = n - 1
	}

	return s, e
}
