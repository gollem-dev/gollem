package main

import (
	"context"
	"io/fs"
	"log/slog"
	"net"
	"net/http"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/gollem-dev/gollem/cmd/gollem/frontend"
	"github.com/m-mizutani/goerr/v2"
)

type serverOption func(*server)

func withAddr(addr string) serverOption {
	return func(s *server) {
		s.addr = addr
	}
}

func withSource(src traceSource) serverOption {
	return func(s *server) {
		s.source = src
	}
}

func withNoBrowser() serverOption {
	return func(s *server) {
		s.noBrowser = true
	}
}

type server struct {
	addr      string
	source    traceSource
	noBrowser bool
	mux       *http.ServeMux
}

func newServer(opts ...serverOption) *server {
	s := &server{
		addr: ":18900",
		mux:  http.NewServeMux(),
	}
	for _, opt := range opts {
		opt(s)
	}
	s.setupRoutes()
	return s
}

func (s *server) setupRoutes() {
	// API routes
	s.mux.HandleFunc("GET /api/health", s.handleHealth)
	s.mux.HandleFunc("GET /api/traces", s.handleListTraces)
	s.mux.HandleFunc("GET /api/traces/{path...}", s.handleGetTrace)

	// Static files (SPA fallback)
	s.mux.Handle("/", s.spaHandler())
}

func (s *server) spaHandler() http.Handler {
	distFS, err := fs.Sub(frontend.StaticFiles, "dist")
	if err != nil {
		slog.Error("failed to create sub filesystem for dist", slog.Any("error", err))
		return http.NotFoundHandler()
	}

	fileServer := http.FileServer(http.FS(distFS))

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Try to serve the file directly
		path := r.URL.Path
		if path == "/" {
			path = "/index.html"
		}

		// Check if the file exists in the embedded FS
		f, err := distFS.Open(strings.TrimPrefix(path, "/"))
		if err == nil {
			_ = f.Close()
			fileServer.ServeHTTP(w, r)
			return
		}

		// SPA fallback: serve index.html for all other routes
		r.URL.Path = "/"
		fileServer.ServeHTTP(w, r)
	})
}

func (s *server) handler() http.Handler {
	return s.mux
}

func (s *server) start(ctx context.Context) error {
	listener, err := net.Listen("tcp", s.addr)
	if err != nil {
		return goerr.Wrap(err, "failed to listen", goerr.Value("addr", s.addr))
	}

	addr := listener.Addr().String()
	url := "http://" + addr
	slog.Info("starting trace viewer server", slog.String("addr", addr), slog.String("url", url))

	if !s.noBrowser {
		openBrowser(url)
	}

	srv := &http.Server{
		Handler:           s.mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		<-ctx.Done()
		_ = srv.Close()
	}()

	if err := srv.Serve(listener); err != nil && err != http.ErrServerClosed {
		return goerr.Wrap(err, "server error")
	}

	return nil
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "linux":
		cmd = exec.Command("xdg-open", url)
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", url)
	default:
		return
	}
	if err := cmd.Start(); err != nil {
		slog.Warn("failed to open browser", slog.Any("error", err))
	}
}
