.PHONY: help install install-backend install-frontend dev dev-docker stop test test-backend test-frontend lint lint-backend lint-frontend format build clean evals

# Default target
help:
	@echo "Causal Analysis Agent - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install all dependencies (backend + frontend)"
	@echo "  make install-backend  Install backend dependencies only"
	@echo "  make install-frontend Install frontend dependencies only"
	@echo ""
	@echo "Development:"
	@echo "  make dev              Start local development environment (Docker)"
	@echo "  make dev-local        Start services locally (without Docker)"
	@echo "  make stop             Stop all Docker services"
	@echo "  make logs             View Docker logs"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-backend     Run backend tests only"
	@echo "  make test-frontend    Run frontend tests only"
	@echo "  make evals            Run agentic evaluations (quick)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters for backend and frontend"
	@echo "  make lint-backend     Run backend linters (ruff, black)"
	@echo "  make lint-frontend    Run frontend linters (eslint, prettier)"
	@echo "  make format           Format code (backend + frontend)"
	@echo ""
	@echo "Build:"
	@echo "  make build            Build Docker images"
	@echo "  make clean            Clean build artifacts and caches"

# Installation
install: install-backend install-frontend
	@echo "All dependencies installed successfully!"

install-backend:
	@echo "Installing backend dependencies..."
	cd backend && pip install poetry && poetry install

install-frontend:
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# Development
dev:
	@echo "Starting development environment..."
	docker compose up --build

dev-docker:
	@echo "Starting development environment in detached mode..."
	docker compose up --build -d

dev-local:
	@echo "Starting local development (requires manual service startup)..."
	@echo "Run the following in separate terminals:"
	@echo "  1. cd backend && poetry run uvicorn app.main:app --reload --port 8000"
	@echo "  2. cd frontend && npm run dev"
	@echo "  3. docker compose up postgres redis"

stop:
	@echo "Stopping Docker services..."
	docker compose down

logs:
	docker compose logs -f

# Testing
test: test-backend test-frontend
	@echo "All tests completed!"

test-backend:
	@echo "Running backend tests..."
	cd backend && poetry run pytest -v

test-frontend:
	@echo "Running frontend tests..."
	cd frontend && npm run lint

evals:
	@echo "Running agentic evaluations..."
	cd backend && poetry run python -m evals.run --quick

# Code Quality
lint: lint-backend lint-frontend
	@echo "All linting completed!"

lint-backend:
	@echo "Linting backend code..."
	cd backend && poetry run ruff check . && poetry run black --check .

lint-frontend:
	@echo "Linting frontend code..."
	cd frontend && npm run lint && npm run format:check

format:
	@echo "Formatting code..."
	cd backend && poetry run ruff check --fix . && poetry run black .
	cd frontend && npm run format

format-backend:
	@echo "Formatting backend code..."
	cd backend && poetry run ruff check --fix . && poetry run black .

format-frontend:
	@echo "Formatting frontend code..."
	cd frontend && npm run format

# Build
build:
	@echo "Building Docker images..."
	docker compose build

build-backend:
	docker compose build backend celery-worker celery-beat

build-frontend:
	docker compose build frontend

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	cd frontend && rm -rf .next node_modules/.cache 2>/dev/null || true
	@echo "Cleanup completed!"

# Database
db-migrate:
	@echo "Running database migrations..."
	cd backend && poetry run alembic upgrade head

db-reset:
	@echo "Resetting database..."
	docker compose down -v
	docker compose up postgres -d
	@sleep 3
	cd backend && poetry run alembic upgrade head
