package main

import (
	"context"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	"github.com/spf13/cobra"

	rconfig "xcontrol/internal/rag/config"
	"xcontrol/server"
	"xcontrol/server/api"
	"xcontrol/server/config"
	"xcontrol/server/proxy"
	"xcontrol/ui"
)

var (
	configPath string
	logLevel   string
)

var rootCmd = &cobra.Command{
	Use:   "xcontrol-server",
	Short: "Start the xcontrol server",
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := config.Load(configPath)
		if err != nil {
			slog.Warn("load config", "err", err)
			cfg = &config.Config{}
		}
		if logLevel != "" {
			cfg.Log.Level = logLevel
		}
		if configPath != "" {
			api.ConfigPath = configPath
			rconfig.ServerConfigPath = configPath
		}
		proxy.Set(cfg.Global.Proxy)

		level := slog.LevelInfo
		switch strings.ToLower(cfg.Log.Level) {
		case "debug":
			level = slog.LevelDebug
		case "warn", "warning":
			level = slog.LevelWarn
		case "error":
			level = slog.LevelError
		}
		logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: level}))
		slog.SetDefault(logger)

		var conn *pgx.Conn
		if dsn := cfg.Global.VectorDB.DSN(); dsn != "" {
			logger.Debug("connecting to postgres", "dsn", dsn)
			conn, err = pgx.Connect(context.Background(), dsn)
			if err != nil {
				logger.Error("postgres connect error", "err", err)
			} else {
				logger.Info("postgres connected")
			}
		} else {
			logger.Warn("postgres dsn not provided")
		}

		if addr := cfg.Global.Redis.Addr; addr != "" {
			logger.Debug("connecting to redis", "addr", addr)
			rdb := redis.NewClient(&redis.Options{
				Addr:     addr,
				Password: cfg.Global.Redis.Password,
			})
			if err := rdb.Ping(context.Background()).Err(); err != nil {
				logger.Error("redis connect error", "err", err)
			} else {
				logger.Info("redis connected")
			}
		} else {
			logger.Warn("redis addr not provided")
		}

		uiFS, err := fs.Sub(ui.Assets, "dist")
		if err != nil {
			logger.Error("ui assets", "err", err)
			return
		}

		r := server.New(
			api.RegisterRoutes(conn, cfg.Sync.Repo.Proxy),
			func(r *gin.Engine) {
				fileServer := http.FileServer(http.FS(uiFS))
				r.NoRoute(func(c *gin.Context) {
					if strings.HasPrefix(c.Request.URL.Path, "/api") {
						c.AbortWithStatus(http.StatusNotFound)
						return
					}
					fileServer.ServeHTTP(c.Writer, c.Request)
				})
			},
		)

		r.GET("/metrics", gin.WrapH(promhttp.Handler()))

		r.Run() // listen and serve on 0.0.0.0:8080
	},
}

func init() {
	rootCmd.Flags().StringVar(&configPath, "config", "", "path to server configuration file")
	rootCmd.Flags().StringVar(&logLevel, "log-level", "", "log level (debug, info, warn, error)")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
