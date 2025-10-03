package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	pgvector "github.com/pgvector/pgvector-go"
	"golang.org/x/sync/errgroup"

	"xcontrol/internal/rag/config"
	"xcontrol/internal/rag/embed"
	"xcontrol/internal/rag/rerank"
	"xcontrol/internal/rag/store"
)

type Service struct {
	cfg *config.Config
}

func New(cfg *config.Config) *Service {
	return &Service{cfg: cfg}
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

type scored struct {
	Document
	vscore float64
	tscore float64
	score  float64
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
		emb = embed.NewChutes(embCfg.Endpoint, embCfg.APIKey, embCfg.Dimension)
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
	minHits := s.cfg.Retrieval.MinResults
	if minHits <= 0 {
		minHits = 5
	}

	docsMap := map[string]*scored{}
	var mu sync.Mutex
	var eg errgroup.Group

	eg.Go(func() error {
		vrows, err := conn.Query(ctx, `SELECT repo,path,chunk_id,content,metadata, embedding <#> $1 AS dist FROM documents WHERE embedding IS NOT NULL ORDER BY embedding <#> $1 LIMIT $2`,
			pgvector.NewVector(vecs[0]), cand)
		if err != nil {
			return err
		}
		defer vrows.Close()
		for vrows.Next() {
			var d scored
			var metaBytes []byte
			var dist float64
			if err := vrows.Scan(&d.Repo, &d.Path, &d.ChunkID, &d.Content, &metaBytes, &dist); err != nil {
				return err
			}
			if len(metaBytes) > 0 {
				_ = json.Unmarshal(metaBytes, &d.Metadata)
			}
			d.vscore = -dist
			key := fmt.Sprintf("%s|%s|%d", d.Repo, d.Path, d.ChunkID)
			mu.Lock()
			docsMap[key] = &d
			mu.Unlock()
		}
		return nil
	})

	eg.Go(func() error {
		trows, err := conn.Query(ctx, `SELECT repo,path,chunk_id,content,metadata, ts_rank_cd(content_tsv, websearch_to_tsquery('zhcn_search', $1)) AS rank FROM documents WHERE content_tsv @@ websearch_to_tsquery('zhcn_search', $1) ORDER BY rank DESC LIMIT $2`,
			question, cand)
		if err != nil {
			return err
		}
		defer trows.Close()
		for trows.Next() {
			var metaBytes []byte
			var rank float64
			d := scored{}
			if err := trows.Scan(&d.Repo, &d.Path, &d.ChunkID, &d.Content, &metaBytes, &rank); err != nil {
				return err
			}
			if len(metaBytes) > 0 {
				_ = json.Unmarshal(metaBytes, &d.Metadata)
			}
			d.tscore = rank
			key := fmt.Sprintf("%s|%s|%d", d.Repo, d.Path, d.ChunkID)
			mu.Lock()
			if exist, ok := docsMap[key]; ok {
				exist.tscore = d.tscore
			} else {
				docsMap[key] = &d
			}
			mu.Unlock()
		}
		return nil
	})

	if err := eg.Wait(); err != nil {
		return nil, err
	}

	candidates := make([]*scored, 0, len(docsMap))
	for _, d := range docsMap {
		d.score = alpha*d.vscore + (1-alpha)*d.tscore
		candidates = append(candidates, d)
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
	if len(candidates) > cand {
		candidates = candidates[:cand]
	}

	// optional reranker
	var rr rerank.Reranker
	rCfg := s.cfg.Models.Reranker
	if rCfg.Endpoint != "" {
		rr = rerank.NewBGE(rCfg.Endpoint, rCfg.Token)
	}

	toDocs := func(cands []*scored, lim int) []Document {
		if lim > len(cands) {
			lim = len(cands)
		}
		out := make([]Document, 0, lim)
		for i := 0; i < lim; i++ {
			out = append(out, cands[i].Document)
		}
		return out
	}

	applyRerank := func(ctx context.Context, cands []*scored) []*scored {
		if rr == nil || len(cands) < 16 {
			return cands
		}
		rctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		docs := make([]string, len(cands))
		for i, c := range cands {
			docs[i] = c.Content
		}
		if scores, err := rr.Rerank(rctx, question, docs); err == nil && len(scores) == len(cands) {
			for i := range cands {
				cands[i].score = float64(scores[i])
			}
			sort.Slice(cands, func(i, j int) bool { return cands[i].score > cands[j].score })
		}
		return cands
	}

	if len(candidates) < minHits {
		resCh := make(chan []Document, 2)
		ctx2, cancel := context.WithCancel(ctx)
		go func() {
			resCh <- toDocs(applyRerank(ctx2, candidates), limit)
		}()
		go func() {
			hydeRes := s.hydeSearch(ctx2, emb, conn, question, cand)
			resCh <- toDocs(applyRerank(ctx2, hydeRes), limit)
		}()
		docs := <-resCh
		cancel()
		return docs, nil
	}

	candidates = applyRerank(ctx, candidates)
	return toDocs(candidates, limit), nil
}

func (s *Service) hydeSearch(ctx context.Context, emb embed.Embedder, conn *pgx.Conn, question string, cand int) []*scored {
	hypo := fmt.Sprintf("Answer: %s", question)
	vecs, _, err := emb.Embed(ctx, []string{hypo})
	if err != nil || len(vecs) == 0 {
		return nil
	}
	rows, err := conn.Query(ctx, `SELECT repo,path,chunk_id,content,metadata, embedding <#> $1 AS dist FROM documents WHERE embedding IS NOT NULL ORDER BY embedding <#> $1 LIMIT $2`,
		pgvector.NewVector(vecs[0]), cand)
	if err != nil {
		return nil
	}
	defer rows.Close()
	out := make([]*scored, 0)
	for rows.Next() {
		var d scored
		var metaBytes []byte
		var dist float64
		if err := rows.Scan(&d.Repo, &d.Path, &d.ChunkID, &d.Content, &metaBytes, &dist); err != nil {
			return nil
		}
		if len(metaBytes) > 0 {
			_ = json.Unmarshal(metaBytes, &d.Metadata)
		}
		d.vscore = -dist
		d.score = d.vscore
		out = append(out, &d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].score > out[j].score })
	if len(out) > cand {
		out = out[:cand]
	}
	return out
}
