package adfuller

const (
	LEFT_TAIL  = "left_tail"
	RIGHT_TAIL = "right_tail"
)

// 残差为白噪采样方法
type whiteNoiseSampleMethod int

const (
	PARAMETIC_BOOTSTRAP    whiteNoiseSampleMethod = iota // "parametic"
	NONPARAMETIC_BOOTSTRAP                               // "nonparametic"
	WILD_BOOTSTRAP                                       // "wild"
)

type LagMode int

const (
	LAG_MODE_AIC   LagMode = iota // "AIC"
	LAG_MODE_BIC                  // "BIC"
	LAG_MODE_TSTAT                // "t-stat"
	LAG_MODE_ERROR                // "ERROR"
)

func (s LagMode) String() string {
	switch s {
	case LAG_MODE_AIC:
		return "AIC"
	case LAG_MODE_BIC:
		return "BIC"
	case LAG_MODE_TSTAT:
		return "t-stat"
	default:
		return "ERROR"
	}
}
func GetMyLagMode(s string) LagMode {
	switch s {
	case "AIC":
		return LAG_MODE_AIC
	case "BIC":
		return LAG_MODE_BIC
	case "t-stat":
		return LAG_MODE_TSTAT
	default:
		return LAG_MODE_ERROR
	}
}
