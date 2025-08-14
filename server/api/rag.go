package api

import (
	"context"
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"

	"xcontrol/internal/rag"
	rconfig "xcontrol/internal/rag/config"
	"xcontrol/internal/rag/store"
	"xcontrol/server/proxy"
)

// ragService defines methods used by the RAG API. It allows tests to supply a
// mock implementation without touching the real vector database or embedding
// service.
type ragService interface {
	Upsert(ctx context.Context, rows []store.DocRow) (int, error)
	Query(ctx context.Context, question string, limit int) ([]rag.Document, []float64, error)
}

// ragSvc handles RAG document storage and retrieval.
var ragSvc ragService = initRAG()

// initRAG attempts to construct a RAG service from server configuration.
func initRAG() ragService {
	cfg, err := rconfig.LoadServer()
	if err != nil {
		return nil
	}
	proxy.Set(cfg.Proxy)
	return rag.New(cfg.ToConfig())
}

// registerRAGRoutes wires the /api/rag upsert and query endpoints.
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
			Question string `json:"question"`
			AB       bool   `json:"ab"`
		}
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if ragSvc == nil {
			c.JSON(http.StatusOK, gin.H{"chunks": nil, "confidence": 0, "top_scores": nil})
			return
		}
		docs, scores, err := ragSvc.Query(c.Request.Context(), req.Question, 5)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		confidence := 0.0
		if len(scores) > 0 {
			confidence = scores[0]
		}
		const tau1 = 0.5
		const tau2 = 0.6
		if req.AB && (len(scores) == 0 || confidence < tau1) {
			if hypo, err := askFn(req.Question); err == nil {
				slog.Info("rag query fallback", "reason", "fallback_hyde", "question", req.Question)
				docs, scores, err = ragSvc.Query(c.Request.Context(), hypo, 5)
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
					return
				}
				if len(scores) > 0 {
					confidence = scores[0]
				} else {
					confidence = 0
				}
			}
		}
		if req.AB && (len(scores) == 0 || confidence < tau2) {
			note := "未检索到权威来源，请以通用知识作答并标注\"无内部来源\"。\n\n" + req.Question
			ans, err := askFn(note)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			slog.Info("rag query fallback", "reason", "fallback_askai", "question", req.Question)
			c.JSON(http.StatusOK, gin.H{
				"answer":     ans,
				"chunks":     docs,
				"confidence": confidence,
				"top_scores": scores,
			})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"chunks":     docs,
			"confidence": confidence,
			"top_scores": scores,
		})
	})
}
