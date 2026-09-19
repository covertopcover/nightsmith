package main

import (
	"fmt"

	"github.com/BurntSushi/toml"
)

func tomlDecode(s string, v any) (toml.MetaData, error) { return toml.Decode(s, v) }

func fmtSscan(s string, v *int) (int, error) { return fmt.Sscan(s, v) }
