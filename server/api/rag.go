package api

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"golang.org/x/sync/singleflight"

	"xcontrol/internal/metrics"
	"xcontrol/internal/rag"
	rconfig "xcontrol/internal/rag/config"
	"xcontrol/internal/rag/store"
	"xcontrol/server/proxy"
)

type ragService interface {
	Upsert(ctx context.Context, rows []store.DocRow) (int, error)
	Query(ctx context.Context, question string, limit int) ([]rag.Document, error)
}

var (
	ragSvc ragService = initRAG()
	rdb    *redis.Client
	sf     singleflight.Group
)

const pipelineVersion = "1"

func initRAG() ragService {
	cfg, err := rconfig.LoadServer()
	if err != nil {
		return nil
	}
	proxy.Set(cfg.Proxy)
	if addr := cfg.Redis.Addr; addr != "" {
		client := redis.NewClient(&redis.Options{Addr: addr, Password: cfg.Redis.Password})
		if err := client.Ping(context.Background()).Err(); err == nil {
			rdb = client
		}
	}
	return rag.New(cfg.ToConfig(), rdb)
}

func registerRAGRoutes(r *gin.RouterGroup) {
	r.POST("/rag/upsert", func(c *gin.Context) {
		if ragSvc == nil {
			c.JSON(http.StatusOK, gin.H{"rows": 0})
			return
		}
		var req struct {
			Docs []store.DocRow `json:"docs"`
		}
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		n, err := ragSvc.Upsert(c.Request.Context(), req.Docs)
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"rows": 0, "error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"rows": n})
	})

	r.POST("/rag/query", func(c *gin.Context) {
		var req struct {
			Question string          `json:"question"`
			History  json.RawMessage `json:"history"`
		}
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if ragSvc == nil {
			c.JSON(http.StatusOK, gin.H{"chunks": nil})
			return
		}
		qNorm := normalize(req.Question)
		hNorm := normalize(string(req.History))
		answerKey := "rag:q:" + hashString(qNorm+hNorm) + ":v" + pipelineVersion
		if rdb != nil {
			if data, err := rdb.Get(c, answerKey).Bytes(); err == nil {
				metrics.CacheHit.WithLabelValues("answer").Inc()
				c.Data(http.StatusOK, "application/json", data)
				return
			}
		}
		ctx := c.Request.Context()
		val, err, _ := sf.Do(answerKey, func() (any, error) {
			if rdb != nil {
				lockKey := "lock:q:" + hashString(qNorm)
				if ok, _ := rdb.SetNX(ctx, lockKey, "1", 5*time.Second).Result(); ok {
					defer rdb.Del(ctx, lockKey)
				} else {
					for i := 0; i < 50; i++ {
						time.Sleep(100 * time.Millisecond)
						if data, err := rdb.Get(ctx, answerKey).Bytes(); err == nil {
							metrics.CacheHit.WithLabelValues("answer").Inc()
							return data, nil
						}
					}
				}
			}
			docs, err := ragSvc.Query(ctx, req.Question, 5)
			if err != nil {
				return nil, err
			}
			resp := map[string]any{"chunks": docs}
			data, _ := json.Marshal(resp)
			if rdb != nil {
				rdb.Set(ctx, answerKey, data, 10*time.Minute)
			}
			return data, nil
		})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if b, ok := val.([]byte); ok {
			c.Data(http.StatusOK, "application/json", b)
			return
		}
		c.JSON(http.StatusOK, gin.H{"chunks": val})
	})
}

func normalize(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func hashString(s string) string {
	h := sha1.Sum([]byte(s))
	return hex.EncodeToString(h[:])
}
