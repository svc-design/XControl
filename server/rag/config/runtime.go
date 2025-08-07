package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Runtime holds runtime configuration for RAG features.
type Datasource struct {
	Name string `yaml:"name"`
	Repo string `yaml:"repo"`
	Path string `yaml:"path"`
}

type Runtime struct {
	Redis struct {
		Addr     string `yaml:"addr"`
		Password string `yaml:"password"`
	} `yaml:"redis"`
	Module      string       `yaml:"module"`
	Embedder    string       `yaml:"embedder"`
	VectorDB    VectorDB     `yaml:"vectordb"`
	Datasources []Datasource `yaml:"datasources"`
}

// VectorDB holds configuration for the PostgreSQL vector store.
type VectorDB struct {
	PGURL      string `yaml:"pgurl"`
	PGHost     string `yaml:"pg_host"`
	PGPort     int    `yaml:"pg_port"`
	PGUser     string `yaml:"pg_user"`
	PGPassword string `yaml:"pg_password"`
	PGDBName   string `yaml:"pg_db_name"`
	PGSSLMode  string `yaml:"pg_sslmode"`
}

// DSN returns the connection string for the database.
// If PGURL is provided it is used, otherwise a DSN is constructed
// from individual fields. When insufficient fields are provided it
// returns an empty string.
func (v VectorDB) DSN() string {
	if v.PGURL != "" {
		return v.PGURL
	}
	if v.PGHost == "" || v.PGUser == "" || v.PGDBName == "" {
		return ""
	}
	port := v.PGPort
	if port == 0 {
		port = 5432
	}
	ssl := v.PGSSLMode
	if ssl == "" {
		ssl = "require"
	}
	return fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=%s", v.PGUser, v.PGPassword, v.PGHost, port, v.PGDBName, ssl)
}

// LoadServer loads RAG configuration from server/config/server.yaml.
func LoadServer() (*Runtime, error) {
	path := filepath.Join("server", "config", "server.yaml")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg struct {
		RAG Runtime `yaml:"RAG"`
	}
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		return nil, err
	}
	return &cfg.RAG, nil
}

// ToConfig converts runtime configuration into service configuration.
func (rt *Runtime) ToConfig() *Config {
	if rt == nil {
		return nil
	}
	var c Config
	for _, ds := range rt.Datasources {
		c.Repos = append(c.Repos, Repo{
			URL:   ds.Repo,
			Paths: []string{ds.Path},
			Local: filepath.Join("server", "rag", ds.Name),
		})
	}
	c.Embedder = rt.Embedder
	return &c
}
