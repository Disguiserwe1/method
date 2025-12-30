package childAutoCorr

import (
	"fmt"
	"math"
	"method/ml/ols"
	"method/propagator/metaorder/pseudo"
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"
	"strategyCrypto/pkg/container/chunklist"
	"strategyCrypto/pkg/preprocess/aggTradeRecords"
	"strategyCrypto/pkg/utils/myDataUtils"
	"strategyCrypto/pkg/utils/myTools"
)

type cachePseudo1 struct {
	pseudo  []pseudo.PseudoTrade
	eps     []int     // sign符号方向
	logQ    []float64 // logQ成交量
	bidsQ   []float64 // bids成交量
	asksQ   []float64 // asks成交量
	deltaTs []int64   // ΔTs
}

// 此处的pseudo是  deltaTs间隔; 为VD标准化后的 logQ
type enhancePse struct {
	pseudo.PseudoTrade
	deltaTs int64
	logQ    float64
}

type cachePseudo2 struct {
	bidsPseudo  *chunklist.ChunkedList[enhancePse]
	asksPseudo  *chunklist.ChunkedList[enhancePse]
	maxBidsLogQ float64
	maxAsksLogQ float64
}

type savePse struct {
}

type childPseudo struct {
	recordReq *myDataUtils.FileReq
	dateList  []string
	data1     cachePseudo1 // childorder序列
	data2     cachePseudo2
	bins      []int
	edges     []float64
	binIdx    [][]int
}

func NewChildPseudo(symbolName string, rType myDataUtils.RecordsType) *childPseudo {
	return &childPseudo{
		recordReq: myDataUtils.NewFileReq(symbolName, rType),
		dateList:  make([]string, 0),
		data1: cachePseudo1{
			pseudo:  make([]pseudo.PseudoTrade, 0),
			eps:     make([]int, 0),
			logQ:    make([]float64, 0),
			deltaTs: make([]int64, 0),
		},
		data2: cachePseudo2{
			bidsPseudo:  chunklist.NewChunkedList[enhancePse](-1, -1),
			asksPseudo:  chunklist.NewChunkedList[enhancePse](-1, -1),
			maxBidsLogQ: 0,
			maxAsksLogQ: 0,
		},
		bins:   make([]int, 0),
		edges:  make([]float64, 0),
		binIdx: make([][]int, 0),
	}
}

// 从本地存储aggTrade中load records并进行切分
func (cp *childPseudo) LoadRecords(date string, amount float64) error {
	err := cp.recordReq.ReadRecordsByDate(date)
	if err != nil {
		return err
	}
	cp.dateList = append(cp.dateList, date)
	pseudos := make([]pseudo.PseudoTrade, 0)
	qSum := 0.0
	for _, record := range cp.recordReq.GetRecords() {
		agg, ok := aggTradeRecords.RecordToAggData(record)
		if !ok {
			continue
		}
		qSum += agg.Q
		pseudos = append(pseudos, pseudo.ExpandAggTrade(agg)...)
	}

	// 对pseudo做过滤，删除金额小于50的
	// 对pseudo做归一化, qmean = qt / VD(日成交量)
	n := len(pseudos)
	out := make([]pseudo.PseudoTrade, 0, n)
	eps := make([]int, 0, n)
	q := make([]float64, 0, n)
	bq := make([]float64, 0, n)
	aq := make([]float64, 0, n)
	ts := make([]int64, 0, n)

	// 大slice进行边界检查消除
	_ = pseudos[n-1]

	maxBid, maxAsk := 1.0, 1.0
	for i := 0; i < n; i++ {
		pse := pseudos[i]
		// 对于探针类成交全部剔除
		if pse.Q*pseudos[i].P < amount {
			continue
		}
		// 获取时间间隔
		deltaTs := int64(0)
		if i != 0 {
			deltaTs = pse.T - pseudos[i-1].T
		}

		logQ := math.Log1p(pse.Q / qSum)
		if math.IsNaN(logQ) {
			continue
		}
		ts = append(ts, deltaTs)
		out = append(out, pse)
		if pse.BM {
			eps = append(eps, -1)
			bq = append(bq, pse.Q)
			cp.data2.bidsPseudo.Append(enhancePse{pse, deltaTs, logQ})
			if maxBid < logQ {
				maxBid = logQ
			}
		} else {
			eps = append(eps, 1)
			aq = append(aq, pse.Q)
			cp.data2.asksPseudo.Append(enhancePse{pse, deltaTs, logQ})
			if maxAsk < logQ {
				maxAsk = logQ
			}
		}
		q = append(q, logQ)
	}
	// 扩容&append
	cp.data1.pseudo = myTools.AppendGrow(cp.data1.pseudo, out)
	cp.data1.eps = myTools.AppendGrow(cp.data1.eps, eps)
	cp.data1.logQ = myTools.AppendGrow(cp.data1.logQ, q)
	cp.data1.bidsQ = myTools.AppendGrow(cp.data1.bidsQ, bq)
	cp.data1.asksQ = myTools.AppendGrow(cp.data1.asksQ, aq)
	cp.data1.deltaTs = myTools.AppendGrow(cp.data1.deltaTs, ts)

	fmt.Printf("日期%s records已加载.", date)
	return nil
}

// minEpsRequired = 1000 最小eps数据点要求
// q gapTs trade 数据点间隔分位数
func (cp *childPseudo) PreSplitEps(minEpsRequired, q float64) {

	quantile95 := myTools.Quantile(cp.data1.deltaTs, 0.95)
	quantile99 := myTools.Quantile(cp.data1.deltaTs, 0.99)
	quantile995 := myTools.Quantile(cp.data1.deltaTs, 0.995)
	quantile999 := myTools.Quantile(cp.data1.deltaTs, 0.999)
	quantile9999 := myTools.Quantile(cp.data1.deltaTs, 0.9999)
	quantileQ95 := myTools.Quantile(cp.data1.logQ, 0.95)
	quantileQ99 := myTools.Quantile(cp.data1.logQ, 0.99)
	quantileQ995 := myTools.Quantile(cp.data1.logQ, 0.995)
	quantileQ999 := myTools.Quantile(cp.data1.logQ, 0.999)
	quantileQ9999 := myTools.Quantile(cp.data1.logQ, 0.9999)
	fmt.Println(cp.data1.deltaTs[1], cp.data1.deltaTs[10], cp.data1.deltaTs[100])
	fmt.Println(quantileQ95, quantileQ99, quantileQ995, quantileQ999, quantileQ9999, len(cp.data1.deltaTs)/10000)
	fmt.Println(quantile95, quantile99, quantile995, quantile999, quantile9999, len(cp.data1.deltaTs)/1000)
}

// step1: 给q做分bin | 采用log-equal-width分成 K 个 bins, bin有最小样本数的约束
func (cp *childPseudo) LogEqualSplitBins(K, tauMax int) []float64 {
	cp.buildBins(K)
	err := cp.collectBinIndices(K)
	if err != nil {
		fmt.Println(err)
	}
	gamma := cp.estimateGamma(K, tauMax)
	return gamma
}

// 按 logQ 等宽切成 K 个 bins | O(N)
func (cp *childPseudo) buildBins(K int) {
	n := len(cp.data1.logQ)
	Lmin := math.Inf(1)
	Lmax := math.Inf(-1)

	for _, v := range cp.data1.logQ {
		if v < Lmin {
			Lmin = v
		}
		if v > Lmax {
			Lmax = v
		}
	}

	// 构建边界
	edges := make([]float64, K+1)
	delta := (Lmax - Lmin) / float64(K)
	for i := 0; i <= K; i++ {
		edges[i] = Lmin + delta*float64(i)
	}

	// 给每个元素分配 bin index
	bins := make([]int, n)
	for i, v := range cp.data1.logQ {
		if v == Lmax {
			bins[i] = K - 1
			continue
		}
		idx := int((v - Lmin) / delta)
		if idx < 0 {
			idx = 0
		}
		if idx >= K {
			idx = K - 1
		}
		bins[i] = idx
	}
	cp.bins = bins
	cp.edges = edges
}

// 收集属于各 bin 的 trade index 列表 | O(N)
func (cp *childPseudo) collectBinIndices(K int) error {
	var err error
	err = nil
	n := len(cp.bins)
	count := make([]int, K)

	// 先计数分配容量
	for i := range n {
		count[cp.bins[i]]++
	}
	for i := range K {
		if count[i] < max(5000, n/(10*K)) {
			err = errorx.New(errCode.INVALID_VALUE, "bin 内logQ数量不满足最小要去")
		}
	}

	// 分配
	binIdx := make([][]int, K)
	for k := range K {
		binIdx[k] = make([]int, 0, count[k])
	}

	// 填充 index
	for i := range n {
		k := cp.bins[i]
		binIdx[k] = append(binIdx[k], i)
	}

	cp.binIdx = binIdx

	return err
}

// 计算每个 bin 的 C_B(τ)，O(#bin * τMax)
func (cp *childPseudo) computeCB(k, tauMax int) []float64 {
	n := len(cp.data1.eps)
	C := make([]float64, tauMax+1)

	for tau := 1; tau <= tauMax; tau++ {
		var sum float64
		count := 0

		for _, t := range cp.binIdx[k] {
			if t+tau >= n {
				break
			}
			sum += float64(cp.data1.eps[t] * cp.data1.eps[t+tau])
			count++
		}

		if count > 0 {
			C[tau] = sum / float64(count)
		}
	}

	return C
}

// 拟合 C(τ) ~ τ^{-γ} 的 γ，用最小二乘
func fitGammaTau(C []float64) float64 {
	if len(C) < 2 {
		return math.NaN()
	}
	x := make([]float64, 0)
	for i := range C {
		x = append(x, float64(i))
	}
	regr := ols.Regression(x, C)

	return -regr.Slope
}

// logQ + eps → γ_k（每个 bin 对应一个 γ）
func (cp *childPseudo) estimateGamma(K int, tauMax int) []float64 {

	gammas := make([]float64, K)
	for k := range K {
		C := cp.getCB(k)
		gammas[k] = fitGammaTau(C)
	}

	return gammas
}

func (cp *childPseudo) getCB(k int) []float64 {
	C := make([]float64, 0, len(cp.binIdx[k]))
	for _, t := range cp.binIdx[k] {
		C = append(C, float64(cp.data1.eps[t]))
	}
	return C
}
