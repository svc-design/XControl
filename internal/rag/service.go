package rag

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	pgvector "github.com/pgvector/pgvector-go"
	"github.com/redis/go-redis/v9"

	"xcontrol/internal/metrics"
	"xcontrol/internal/rag/config"
	"xcontrol/internal/rag/embed"
	"xcontrol/internal/rag/rerank"
	"xcontrol/internal/rag/store"
)

type Service struct {
	cfg *config.Config
	rdb *redis.Client
}

func New(cfg *config.Config, rdb *redis.Client) *Service {
	return &Service{cfg: cfg, rdb: rdb}
}

// Upsert stores pre-embedded documents into the vector database.
func (s *Service) Upsert(ctx context.Context, rows []store.DocRow) (int, error) {
	if s == nil || s.cfg == nil || len(rows) == 0 {
		return 0, nil
	}
	dsn := s.cfg.Global.VectorDB.DSN()
	if dsn == "" {
		return 0, nil
	}
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return 0, err
	}
	defer conn.Close(ctx)

	dim := len(rows[0].Embedding)
	// Allow schema migration so the embedding dimension can be updated
	if err := store.EnsureSchema(ctx, conn, dim, true); err != nil {
		return 0, err
	}
	return store.UpsertDocuments(ctx, conn, rows)
}

type Document struct {
	Repo     string         `json:"repo"`
	Path     string         `json:"path"`
	ChunkID  int            `json:"chunk_id"`
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata"`
}

func (s *Service) Query(ctx context.Context, question string, limit int) ([]Document, error) {
	if s == nil || s.cfg == nil {
		return nil, nil
	}
	embCfg := s.cfg.ResolveEmbedding()
	if embCfg.Endpoint == "" {
		return nil, nil
	}
	var emb embed.Embedder
	switch embCfg.Provider {
	case "ollama":
		emb = embed.NewOllama(embCfg.Endpoint, embCfg.Model, embCfg.Dimension)
	case "chutes":
		emb = embed.NewOpenAI(embCfg.Endpoint, embCfg.APIKey, embCfg.Model, embCfg.Dimension)
	default:
		if embCfg.Model != "" {
			emb = embed.NewOpenAI(embCfg.Endpoint, embCfg.APIKey, embCfg.Model, embCfg.Dimension)
		} else {
			emb = embed.NewBGE(embCfg.Endpoint, embCfg.APIKey, embCfg.Dimension)
		}
	}
	vecs, _, err := emb.Embed(ctx, []string{question})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, nil
	}
	dsn := s.cfg.Global.VectorDB.DSN()
	if dsn == "" {
		return nil, nil
	}
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return nil, err
	}
	defer conn.Close(ctx)

	alpha := s.cfg.Retrieval.Alpha
	if alpha < 0 || alpha > 1 {
		alpha = 0.5
	}
	cand := s.cfg.Retrieval.Candidates
	if cand <= 0 {
		cand = 50
	}

	qNorm := strings.ToLower(strings.TrimSpace(question))
	qHash := hashString(qNorm)

	type scored struct {
		Document
		VScore float64 `json:"vscore"`
		TScore float64 `json:"tscore"`
		Score  float64 `json:"score"`
	}

	var candidates []*scored
	retrKey := ""
	if s.rdb != nil {
		retrKey = fmt.Sprintf("retr:hybrid:%s:alpha:%g:k:%d", qHash, alpha, cand)
		if data, err := s.rdb.Get(ctx, retrKey).Bytes(); err == nil {
			if err := json.Unmarshal(data, &candidates); err == nil {
				metrics.CacheHit.WithLabelValues("retr").Inc()
			} else {
				candidates = nil
			}
		}
	}

	if candidates == nil {
		docsMap := map[string]*scored{}

		vrows, err := conn.Query(ctx, `SELECT repo,path,chunk_id,content,metadata, embedding <#> $1 AS dist FROM documents WHERE embedding IS NOT NULL ORDER BY embedding <#> $1 LIMIT $2`, pgvector.NewVector(vecs[0]), cand)
		if err != nil {
			return nil, err
		}
		for vrows.Next() {
			var d scored
			var metaBytes []byte
			var dist float64
			if err := vrows.Scan(&d.Repo, &d.Path, &d.ChunkID, &d.Content, &metaBytes, &dist); err != nil {
				vrows.Close()
				return nil, err
			}
			if len(metaBytes) > 0 {
				_ = json.Unmarshal(metaBytes, &d.Metadata)
			}
			d.VScore = -dist
			key := fmt.Sprintf("%s|%s|%d", d.Repo, d.Path, d.ChunkID)
			docsMap[key] = &d
		}
		vrows.Close()

		trows, err := conn.Query(ctx, `SELECT repo,path,chunk_id,content,metadata, ts_rank_cd(content_tsv, websearch_to_tsquery('zhcn_search', $1)) AS rank FROM documents WHERE content_tsv @@ websearch_to_tsquery('zhcn_search', $1) ORDER BY rank DESC LIMIT $2`, question, cand)
		if err != nil {
			return nil, err
		}
		for trows.Next() {
			var metaBytes []byte
			var rank float64
			key := ""
			d := scored{}
			if err := trows.Scan(&d.Repo, &d.Path, &d.ChunkID, &d.Content, &metaBytes, &rank); err != nil {
				trows.Close()
				return nil, err
			}
			if len(metaBytes) > 0 {
				_ = json.Unmarshal(metaBytes, &d.Metadata)
			}
			d.TScore = rank
			key = fmt.Sprintf("%s|%s|%d", d.Repo, d.Path, d.ChunkID)
			if exist, ok := docsMap[key]; ok {
				exist.TScore = d.TScore
			} else {
				docsMap[key] = &d
			}
		}
		trows.Close()

		candidates = make([]*scored, 0, len(docsMap))
		for _, d := range docsMap {
			d.Score = alpha*d.VScore + (1-alpha)*d.TScore
			candidates = append(candidates, d)
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].Score > candidates[j].Score })
		if len(candidates) > cand {
			candidates = candidates[:cand]
		}
		if s.rdb != nil && retrKey != "" {
			if data, err := json.Marshal(candidates); err == nil {
				_ = s.rdb.Set(ctx, retrKey, data, 20*time.Minute).Err()
			}
		}
	}

	candBytes, _ := json.Marshal(candidates)

	var rr rerank.Reranker
	rCfg := s.cfg.Models.Reranker
	if rCfg.Endpoint != "" {
		rr = rerank.NewBGE(rCfg.Endpoint, rCfg.Token)
	}
	rm := ""
	if len(rCfg.Models) > 0 {
		rm = rCfg.Models[0]
	}
	if rr != nil && len(candidates) > 0 {
		rerankKey := ""
		if s.rdb != nil {
			candHash := hashString(string(candBytes))
			rerankKey = fmt.Sprintf("rerank:%s:on:%s:model:%s", qHash, candHash, rm)
			if data, err := s.rdb.Get(ctx, rerankKey).Bytes(); err == nil {
				var cached []*scored
				if err := json.Unmarshal(data, &cached); err == nil {
					metrics.CacheHit.WithLabelValues("rerank").Inc()
					candidates = cached
					goto SELECT
				}
			}
		}

		docs := make([]string, len(candidates))
		for i, c := range candidates {
			docs[i] = c.Content
		}
		if scores, err := rr.Rerank(ctx, question, docs); err == nil && len(scores) == len(candidates) {
			for i := range candidates {
				candidates[i].Score = float64(scores[i])
			}
			sort.Slice(candidates, func(i, j int) bool { return candidates[i].Score > candidates[j].Score })
			if s.rdb != nil && rerankKey != "" {
				if data, err := json.Marshal(candidates); err == nil {
					_ = s.rdb.Set(ctx, rerankKey, data, 60*time.Minute).Err()
				}
			}
		}
	}

SELECT:
	if limit > len(candidates) {
		limit = len(candidates)
	}
	out := make([]Document, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, candidates[i].Document)
	}
	return out, nil
}

func hashString(s string) string {
	h := sha1.Sum([]byte(s))
	return hex.EncodeToString(h[:])
}
