package api

import (
	"context"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"

	"xcontrol/server/rag"
	rconfig "xcontrol/server/rag/config"
	"xcontrol/server/rag/embed"
	"xcontrol/server/rag/store"
)

// ragSvc provides repository sync and retrieval operations.
var ragSvc = initRAG()

// initRAG attempts to construct a RAG service from server configuration.
func initRAG() *rag.Service {
	cfg, err := rconfig.LoadServer()
	if err != nil {
		return nil
	}
	dsn := cfg.VectorDB.DSN()
	if dsn == "" {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	st, err := store.New(ctx, dsn)
	if err != nil {
		return nil
	}
	svcCfg := cfg.ToConfig()
	var emb embed.Embedder
	switch svcCfg.Embedder {
	case "bge":
		if ep := os.Getenv("BGE_ENDPOINT"); ep != "" {
			emb = embed.NewBGE(ep)
		}
	case "openai":
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			emb = embed.NewOpenAI("text-embedding-3-small", key)
		}
	default:
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			emb = embed.NewOpenAI("text-embedding-3-small", key)
		} else if ep := os.Getenv("BGE_ENDPOINT"); ep != "" {
			emb = embed.NewBGE(ep)
		}
	}
	svc := rag.New(svcCfg, st, emb)
	go svc.Sync(context.Background())
	go svc.Watch(context.Background())
	return svc
}

// registerRAGRoutes wires the /api/rag endpoints.
func registerRAGRoutes(r *gin.RouterGroup) {
	r.POST("/rag/sync", func(c *gin.Context) {
		if ragSvc == nil {
			c.JSON(http.StatusOK, gin.H{"status": "ok"})
			return
		}
		if err := ragSvc.Sync(c.Request.Context()); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	r.POST("/rag/query", func(c *gin.Context) {
		var req struct {
			Question string `json:"question"`
		}
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if ragSvc == nil {
			c.JSON(http.StatusOK, gin.H{"chunks": nil})
			return
		}
		docs, err := ragSvc.Query(c.Request.Context(), req.Question, 5)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"chunks": docs})
	})
}
