package fundingratemodel

import (
	"fmt"
	"os"
	"strings"
	"sync/atomic"

	"gopkg.in/yaml.v3"
)

type Config struct {
	MaxLeverage map[string]int `yaml:"maxleverage"`
}

// 用 atomic.Value 存当前配置，支持热更新时无锁读取
var cfgValue atomic.Value // stores *Config

func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read yaml: %w", err)
	}

	var c Config
	if err := yaml.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("unmarshal yaml: %w", err)
	}
	if c.MaxLeverage == nil {
		c.MaxLeverage = make(map[string]int)
	}

	// 规范化 key：全大写、去空格
	norm := make(map[string]int, len(c.MaxLeverage))
	for k, v := range c.MaxLeverage {
		sym := strings.ToUpper(strings.TrimSpace(k))
		if v <= 0 {
			return nil, fmt.Errorf("invalid leverage for %s: %d", sym, v)
		}
		norm[sym] = v
	}
	c.MaxLeverage = norm

	return &c, nil
}

func Init(path string) error {
	c, err := Load(path)
	if err != nil {
		return err
	}
	cfgValue.Store(c)
	return nil
}

// O(1) 查找：找不到就用 defaultLeverage
func GetMaxLeverage(symbol string, defaultLeverage int) int {
	cAny := cfgValue.Load()
	if cAny == nil {
		return defaultLeverage
	}
	c := cAny.(*Config)

	sym := strings.ToUpper(strings.TrimSpace(symbol))
	if v, ok := c.MaxLeverage[sym]; ok {
		return v
	}
	return defaultLeverage
}
