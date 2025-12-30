package fundingratemodel

import (
	"strategyCrypto/pkg/container/bitSet/depthBitSet"
	"strategyCrypto/pkg/utils/myDecimal"
)

//
// https://www.binance.com/zh-CN/support/faq/detail/360033525031
// bn的溢价指数P，对每个合约计算以缩小永续和标记价格价差:
// 溢价指数(P) = [Max(0, 冲击买价 - 价格指数) - Max(0, 价格指数 - 冲击卖价)] / 价格指数
// 冲击买价 = 按买价执行冲击保证金额的平均成交价
// 冲击卖价 = 按卖价执行冲击保证金额的平均成交价
// 价格指数是在主要现货交易平台上架的标的资产的加权平均价值。(为了简化先采用marketprice替代)
// U 本位合约的 IMN 是以价值 200 USDT 的保证金（USDT 计价）进行交易时的名义金额
// 冲击保证金额(IMN) = 200 USDT / 最高杠杆下的起始保证金率

const (
	FIX_RATE = 0.0003 // Binance默认固定利率每天0.03%
	// 请注意，这不适用于特定合约，例如 ETHBTC 的利率设为 0%。币安保留根据市场状况不定时调整利率的权利。
	FIX_DEPOSIT = 200.0 // U 本位合约的 IMN 是以价值 200 USDT 的保证金（USDT 计价）进行交易时的名义金额

)

type FrBn struct {
	req frReq
	cacheImnAsk
	cacheImnBid
}
type cacheImnAsk struct {
	bestprice  uint32
	imn        float64
	xcumSumAmt float64 // 累积委托金额
	xcumSumQty float64 // 累计基础数量
	xPrice     float64 // x等级的价格
}

type cacheImnBid struct {
	bestprice  uint32
	imn        float64
	xcumSumAmt float64 // 累积委托金额
	xcumSumQty float64 // 累计基础数量
	xPrice     float64 // x等级的价格
}

type frReq struct {
	maxLeverage float64 // 最高杠杆下的起始保证金率
}

func NewFrReq(leverage float64) *FrBn {
	return &FrBn{req: frReq{leverage}}
}

// 冲击保证金额(IMN) = 200 USDT / 最高杠杆下的起始保证金率
func (fr *FrBn) CalIMN() float64 {
	return FIX_DEPOSIT * fr.req.maxLeverage
}

// 冲击卖价,若非正常则返回false
func (fr *FrBn) CalImpulseBid(bs *depthBitSet.BidSide) (float64, bool) {
	xcumSumAmt := 0.0 // 累积委托金额
	xPrice := 0.0     // x等级的价格
	xcumSumQty := 0.0 // 累计基础数量
	imn := fr.CalIMN()

	stopByBestPrice := false // 是否因为bestprice就超出imn
	ok, err := bs.RangeFromFn(bs.BestPrice, func(p, q uint32) bool {
		pDec := myDecimal.NewPrice(p, bs.PScale())
		qDec := myDecimal.NewQty(q, bs.QScale())
		pF64 := pDec.GetScaleValueF64()
		qF64 := qDec.GetScaleValueF64()
		amtF64 := myDecimal.MulAmount(pDec, qDec).GetScaleValueF64()
		xcumSumQty += qF64
		xcumSumAmt += amtF64
		if xcumSumAmt > imn { // 累积委托金额大于IMN
			xPrice = pF64
			xcumSumQty -= qF64
			xcumSumAmt -= amtF64
			if p == bs.BestPrice {
				stopByBestPrice = true
			}
			return false
		} else {
			return true
		}
	})
	if err != nil || stopByBestPrice {
		ok = false
	}
	if xPrice == 0.0 {
		return 0.0, false
	}
	fr.cacheImnBid = cacheImnBid{
		bestprice:  bs.BestPrice,
		imn:        imn,
		xcumSumAmt: xcumSumAmt,
		xcumSumQty: xcumSumQty,
		xPrice:     xPrice,
	}
	return imn / ((imn-xcumSumAmt)/xPrice + xcumSumQty), ok
}

// 冲击买价
func (fr *FrBn) CalImpulseAsk(bs *depthBitSet.AskSide) (float64, bool) {
	xcumSumAmt := 0.0 // 累积委托金额
	xPrice := 0.0     // x等级的价格
	xcumSumQty := 0.0 // 累计基础数量
	imn := fr.CalIMN()

	stopByBestPrice := false // 是否因为bestprice就超出imn
	ok, err := bs.RangeFromFn(bs.BestPrice, func(p, q uint32) bool {
		pDec := myDecimal.NewPrice(p, bs.PScale())
		qDec := myDecimal.NewQty(q, bs.QScale())
		pF64 := pDec.GetScaleValueF64()
		qF64 := qDec.GetScaleValueF64()
		amtF64 := myDecimal.MulAmount(pDec, qDec).GetScaleValueF64()
		xcumSumQty += qF64
		xcumSumAmt += amtF64
		if xcumSumAmt > imn { // 累积委托金额大于IMN
			xPrice = pF64
			xcumSumQty -= qF64
			xcumSumAmt -= amtF64
			if p == bs.BestPrice {
				stopByBestPrice = true
			}
			return false
		} else {
			return true
		}
	})

	if err != nil || stopByBestPrice {
		ok = false
	}
	if xPrice == 0.0 {
		return 0.0, false
	}
	fr.cacheImnAsk = cacheImnAsk{
		bestprice:  bs.BestPrice,
		imn:        imn,
		xcumSumAmt: xcumSumAmt,
		xcumSumQty: xcumSumQty,
		xPrice:     xPrice,
	}
	return imn / ((imn-xcumSumAmt)/xPrice + xcumSumQty), ok
}

func (fr *FrBn) GetCacheImn() (cacheImnAsk, cacheImnBid) {
	return fr.cacheImnAsk, fr.cacheImnBid
}

// 溢价指数(P) = [Max(0, 冲击买价 - 价格指数) - Max(0, 价格指数 - 冲击卖价)] / 价格指数
