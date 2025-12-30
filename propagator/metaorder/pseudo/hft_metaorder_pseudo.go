package pseudo

// 1.订单流的簇结构，利用Δt(inter-arrival time) 去识别clustering
// Lillo, Mike & Farmer (2005) | Sato & Kanazawa (2023, 2024) |
// Bouchaud et al. (2018) |
// 2.cluster size 分布与 “trader 活跃度分布” 的关系
// 		metaorder 的长度（child counts）服从幂律分布
// 		- LMF、Bouchaud、TSE 2024、ANcerno 研究
//		交易者参与频率（参与成交的比例）也服从幂律
//		- Sato–Kanazawa (2023, 2024) | Mastromatteo, Toth & Bouchaud (2014)
// 		订单簇（cluster size）是 metaorder 的 noisy 投影
// 		- Tóth et al. (2011)
import (
	"method/propagator/metaorder/hist"
	"ofeisInfra/infra/errorx"
	"ofeisInfra/infra/errorx/errCode"
	"ofeisInfra/quant/market/trade/aggTradeChan"

	"fmt"
	"math"
	"ofeisInfra/pkg/utils/fileUtils"
	"sort"
	"strategyCrypto/pkg/utils/myTools"
)

// 生成 pseudo trades 从aggTrade分割出来的最小子订单
type PseudoTrade struct {
	T        int64   // 时间戳
	P        float64 // 价格
	Q        float64 // 子成交量
	BM       bool    // 买卖方向
	TraderID int     // 映射的交易者ID
}

// 拆分下的元订单结构体
type proxyMetaOrder struct {
	t_start     int64         // 元订单开始时间
	t_end       int64         // 元订单结束时间
	childPseudo []PseudoTrade // 子订单pseudo
	q           float64       // 典型子订单 Q/N 中位数
	Q           float64       // 元订单总量 Q
	N           float64       // 元订单数量
	sign        float64       // 元订单方向
}

// 分割aggTrade 权重符合Uniform(0.5, 1.5)
func ExpandAggTrade(agg aggTradeChan.AggData) []PseudoTrade {

	N := int(agg.N)
	if N <= 1 {
		// 只有 1 笔 trade，不需要拆
		return []PseudoTrade{
			{T: agg.T, P: agg.P, Q: agg.Q, BM: agg.BM},
		}
	}

	// Step 1: 生成随机权重
	weights := make([]float64, N)
	sumW := 0.0
	for i := range N {
		w := myTools.RandomFloat64Variable()*1.0 + 0.5 // Uniform(0.5,1.5)
		weights[i] = w
		sumW += w
	}

	// Step 2: 根据权重生成子成交量 q_i
	q := make([]float64, N)
	for i := range N {
		q[i] = agg.Q * (weights[i] / sumW)
	}

	// Step 3: 生成时间戳
	times := make([]int64, N)
	times[0] = agg.T // 第一笔必须在 T

	for i := 1; i < N; i++ {
		delta := myTools.RandomVariable(100) // 0 ~ 99
		times[i] = agg.T + int64(delta)
	}

	// Step 4: 打包成 trades，并排序
	trades := make([]PseudoTrade, N)
	for i := 0; i < N; i++ {
		trades[i] = PseudoTrade{
			T:  times[i],
			P:  agg.P,
			Q:  q[i],
			BM: agg.BM,
		}
	}

	// 按时间排序
	sort.Slice(trades, func(i, j int) bool {
		return trades[i].T < trades[j].T
	})

	return trades
}

// Burstiness 事件发生不均匀，呈现“爆发—沉寂”模式
// 计算 Δt(inter-arrival time)事件到达的时间间隔:
// · 平均值: μ = E[Δt]
// · 标准差: σ = (E[(Δt - μ)**2])**0.5
// Burstiness定义为:
// 						B = (σ - μ)/(σ + μ)
// B = -1: 完全规律, 间隔恒定;
// B = 0: 泊松过程, 无记忆性;
// B —> 1: 高爆发性

// 经验划分法: 越 burst，簇越明显 → 阈值应该越小，让簇更“紧”。
func chooseDeltaTBt(pseudoS []PseudoTrade) int64 {
	deltaTs := make([]float64, 0, len(pseudoS)-1)
	for i, trade := range pseudoS {
		if i != 0 {
			deltaTs = append(deltaTs, float64(trade.T-pseudoS[i-1].T))
		}
	}
	mu := myTools.ArrMean(deltaTs)
	sigma := myTools.ArrStd(deltaTs)
	hist := hist.Hist(deltaTs, 100)
	quantile60_ := hist[10]
	quantile50_ := hist[20]
	quantile30_ := hist[30]
	burst := (sigma - mu) / (sigma + mu)
	var thresholdTs int64
	switch {
	case burst <= 0.0:
		thresholdTs = int64(quantile60_.Count)
	case burst > 0.0 && burst < 0.5:
		thresholdTs = int64(quantile50_.Count)
	case burst > 0.5:
		thresholdTs = int64(quantile30_.Count)
	default:
		thresholdTs = 0
	}
	fmt.Println("hist", burst, thresholdTs, quantile30_.Count, hist[10].Count, hist[30].Count, hist[50].Count, hist[70].Count)
	return thresholdTs
}

func ClusterByThreshold(pseudoS []PseudoTrade) (float64, error) {
	deltaTs := make([]float64, 0, len(pseudoS)-1)
	for i, trade := range pseudoS {
		if i != 0 {
			deltaTs = append(deltaTs, float64(trade.T-pseudoS[i-1].T))
		}
	}
	mu := myTools.ArrMean(deltaTs)
	sigma := myTools.ArrStd(deltaTs)
	hist := hist.Hist(deltaTs, 100)
	quantile60_ := hist[60]
	quantile50_ := hist[50]
	quantile30_ := hist[10]
	burst := (sigma - mu) / (sigma + mu)
	var thresholdTs int64
	switch {
	case burst <= 0.0:
		thresholdTs = int64(quantile60_.Count)
	case burst > 0.0 && burst < 0.5:
		thresholdTs = int64(quantile50_.Count)
	case burst > 0.5:
		thresholdTs = int64(quantile30_.Count)
	default:
		thresholdTs = 0
		return 0.0, errorx.New(errCode.INVALID_VALUE, "thresholdTs 为零.")
	}

	tmpCluster := 1
	kc := make([]float64, 0)
	for i := 0; i < len(deltaTs); i++ {
		if int64(deltaTs[i]) <= thresholdTs {
			tmpCluster++
		} else {
			kc = append(kc, float64(tmpCluster))
			tmpCluster = 1
		}
	}
	// avgCL := len(pseudoS) / len(kc) // 平均簇长度
	N_min, N_max := 5.0, 500.0
	fmt.Println(burst, thresholdTs, myTools.ArrMean(kc), float64(len(pseudoS))/myTools.ArrMean(kc))
	N_est := min(max(float64(len(pseudoS))/myTools.ArrMean(kc), N_min), N_max)
	return N_est, nil
}

// 对pseudo做过滤,删除金额小于50的
func DeleteMiniOrder(pseudos []PseudoTrade, amount float64) []PseudoTrade {
	out := make([]PseudoTrade, 0)
	for _, pse := range pseudos {
		if pse.Q*pse.P < amount {
			continue
		}
		out = append(out, pse)
	}
	return out
}

func QLogNormal(pseudos []PseudoTrade) (float64, float64) {
	qLogs := make([]float64, 0)
	for _, pse := range pseudos {
		qLogs = append(qLogs, math.Log(pse.Q))
	}
	mu := myTools.ArrMean(qLogs)
	sigma := myTools.ArrStd(qLogs)
	return mu, sigma
}

func MappingTrades(N int, prob []float64, strickiness float64, trades []PseudoTrade) ([]PseudoTrade, error) {
	if len(prob) != N {
		return nil, errorx.New(errCode.INVALID_VALUE, "概率和交易者ID数不匹配")
	}
	if N <= 0 {
		return nil, errorx.New(errCode.INVALID_VALUE, "交易者ID数为正")
	}
	tradesDel := DeleteMiniOrder(trades, 20)
	// 1) normalize f -> p_i
	// sumF := 0.0
	// for _, f := range freq {
	// 	if f < 0 {
	// 		return nil, errorx.New(errCode.INVALID_VALUE, "[]freq 包含负值")
	// 	}
	// 	sumF += f
	// }
	// if sumF == 0 {
	// 	return nil, errorx.New(errCode.INVALID_VALUE, "[]freq 和为零")
	// }

	// p := make([]float64, N)
	// for i := 0; i < N; i++ {
	// 	p[i] = freq[i] / sumF
	// }

	// 按时间排序(可选)
	// sort.Slice(trades, func(i, j int) bool { return trades[i].T < trades[j].T })
	// 3) 对每笔 trade 随机分配 trader ID
	thresholdTs := chooseDeltaTBt(tradesDel)
	out := make([]PseudoTrade, len(tradesDel))
	var tmpProb []float64
	preID := -1
	preTs := int64(0)
	for i, tr := range tradesDel {
		u := myTools.RandomFloat64Variable()
		tmpProb = DynamicProbabilities(prob, preID, float64(tr.T-preTs), strickiness, float64(thresholdTs))
		// 二分查找 cum[]
		l, r := 0, N-1
		for l < r {
			m := (l + r) / 2
			if u <= tmpProb[m] {
				r = m
			} else {
				l = m + 1
			}
		}

		tr.TraderID = l
		out[i] = tr
		preID = tr.TraderID
		preTs = int64(tr.T)
	}

	return out, nil
}

func PowerLawProbabilities(N int, alpha float64) []float64 {
	probs := make([]float64, N)

	// 先计算未归一化分布
	sum := 0.0
	for i := 1; i <= N; i++ {
		p := 1.0 / math.Pow(float64(i), alpha)
		probs[i-1] = p
		sum += p
	}

	// 归一化
	for i := 0; i < N; i++ {
		probs[i] /= sum
	}

	return probs
}

// DynamicProbabilities 返回：
// 对当前 pseudo-trade，结合上一笔 TraderID 及 Δt 后的最终概率分布
func DynamicProbabilities(
	baseProbs []float64, // 基准 power-law 概率（长度 N）
	prevID int, // 上一笔的 TraderID，-1 表示没有上一笔
	deltaT float64, // 当前 pseudo 与上一笔的时间差（单位：秒或毫秒）
	lambda float64, // 最大 stickiness（0~1，如 0.7）
	tau float64, // 时间尺度（如 0.1 秒或 100ms）
) []float64 {

	N := len(baseProbs)
	dyn := make([]float64, N)

	// 如果没有上一笔，则直接返回基准分布
	if prevID < 0 || prevID >= N {
		copy(dyn, baseProbs)
		return dyn
	}

	// stickiness 权重 w(Δt)
	w := lambda * math.Exp(-deltaT/tau)

	// 构造动态分布
	for i := 0; i < N; i++ {
		if i == prevID {
			dyn[i] = w + (1.0-w)*baseProbs[i]
		} else {
			dyn[i] = (1.0 - w) * baseProbs[i]
		}
	}

	return probToCDF(dyn)
}

func probToCDF(prob []float64) []float64 {
	N := len(prob)
	cum := make([]float64, N)
	cum[0] = prob[0]
	for i := 1; i < N; i++ {
		cum[i] = cum[i-1] + prob[i]
	}
	cum[N-1] = 1.0
	return cum
}

type pseudos struct {
	bidsPseudo  []PseudoTrade
	asksPseudo  []PseudoTrade
	metaOrder   []proxyMetaOrder
	pseudoSlice []PseudoTrade
}

func NewPt(bids, asks []PseudoTrade) *pseudos {
	return &pseudos{bidsPseudo: bids, asksPseudo: asks,
		metaOrder: make([]proxyMetaOrder, 0), pseudoSlice: make([]PseudoTrade, 0)}
}

func (p *pseudos) RollingPseudoSelect() {
	for i, bid := range p.bidsPseudo {
		if i == 0 {
			p.pseudoSlice = append(p.pseudoSlice, bid)
		} else {
			if bid.TraderID == p.bidsPseudo[i-1].TraderID {
				p.pseudoSlice = append(p.pseudoSlice, bid)
			} else {
				pse_start := p.pseudoSlice[0]
				pse_end := p.pseudoSlice[len(p.pseudoSlice)-1]
				pse_qty := make([]float64, 0)
				for _, pse := range p.pseudoSlice {
					pse_qty = append(pse_qty, pse.Q)
				}
				qty_median := myTools.ArrMedian(pse_qty)
				qty_sum := myTools.ArrSum(pse_qty)
				metaorder := proxyMetaOrder{
					t_start:     pse_start.T,
					t_end:       pse_end.T,
					childPseudo: p.pseudoSlice,
					q:           qty_median,
					Q:           qty_sum,
					N:           float64(len(pse_qty)),
					sign:        -1,
				}
				p.metaOrder = append(p.metaOrder, metaorder)
				p.pseudoSlice = make([]PseudoTrade, 0)
				p.pseudoSlice = append(p.pseudoSlice, bid)
			}
		}
	}
	p.pseudoSlice = make([]PseudoTrade, 0)

	for i, ask := range p.asksPseudo {
		if i == 0 {
			p.pseudoSlice = append(p.pseudoSlice, ask)
		} else {
			if ask.TraderID == p.asksPseudo[i-1].TraderID {
				p.pseudoSlice = append(p.pseudoSlice, ask)
			} else {
				pse_start := p.pseudoSlice[0]
				pse_end := p.pseudoSlice[len(p.pseudoSlice)-1]
				pse_qty := make([]float64, 0)
				for _, pse := range p.pseudoSlice {
					pse_qty = append(pse_qty, pse.Q)
				}
				qty_median := myTools.ArrMedian(pse_qty)
				qty_sum := myTools.ArrSum(pse_qty)
				metaorder := proxyMetaOrder{
					t_start:     pse_start.T,
					t_end:       pse_end.T,
					childPseudo: p.pseudoSlice,
					q:           qty_median,
					Q:           qty_sum,
					N:           float64(len(pse_qty)),
					sign:        1,
				}
				p.metaOrder = append(p.metaOrder, metaorder)
				p.pseudoSlice = make([]PseudoTrade, 0)
				p.pseudoSlice = append(p.pseudoSlice, ask)
			}
		}
	}

	sort.Slice(p.metaOrder, func(i, j int) bool { return p.metaOrder[i].t_start < p.metaOrder[j].t_start })
}

func (p *pseudos) GetMetaOrder() []proxyMetaOrder {
	return p.metaOrder
}

func (p *pseudos) WriteMO(filepath string) {
	out := make([][]string, 0)
	out = append(out, []string{
		"t_start",
		"t_end",
		"q",
		"Q",
		"N",
		"p_start",
		"p_end",
		"sign",
	})
	for _, mo := range p.metaOrder {
		out = append(out, []string{
			fmt.Sprintf("%v", mo.t_start),
			fmt.Sprintf("%v", mo.t_end),
			fmt.Sprintf("%v", mo.q),
			fmt.Sprintf("%v", mo.Q),
			fmt.Sprintf("%v", mo.N),
			fmt.Sprintf("%v", mo.childPseudo[0].P),
			fmt.Sprintf("%v", mo.childPseudo[len(mo.childPseudo)-1].P),
			fmt.Sprintf("%v", mo.sign),
		})
	}
	fileUtils.WriteDataToDesCsv(out, filepath)
}
