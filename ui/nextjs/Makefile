# Makefile for Next.js on macOS

SHELL := /bin/bash
NODE_VERSION := $(shell node -v 2>/dev/null)
YARN := $(shell command -v yarn 2>/dev/null)

.PHONY: init build run export clean info

init:
	@echo "🔧 Installing dependencies..."
	@if [ -z "$(YARN)" ]; then \
		echo "⚠️  Yarn not found. Installing via Homebrew..."; \
		brew install yarn; \
	fi
	yarn install

build:
	@echo "🔨 Building Next.js project..."
	yarn build

run:
	@echo "🚀 Starting local dev server..."
	yarn dev

export:
	@echo "📦 Exporting static site to ./out ..."
	yarn build && yarn export

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf .next out

info:
	@echo "🧾 Current Node.js version: $(NODE_VERSION)"
