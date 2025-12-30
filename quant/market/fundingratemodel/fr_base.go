package fundingratemodel

// 冲击买卖价格结构体
type implusePrice struct {
	Ts  int64
	bid float64
	ask float64
}

// 溢价指数，保留全量冲击买卖价格序列，markPrice的价格指数以及真实资金费率
type PremiumIndex struct {
}

func (pIdx *PremiumIndex) New() *PremiumIndex {
	return &PremiumIndex{}
}

// 重置溢价指数
func (pIdx *PremiumIndex) Reset() {

}
