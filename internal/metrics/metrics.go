package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var CacheHit = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "cache_hit",
		Help: "Cache hit counts.",
	},
	[]string{"type"},
)
