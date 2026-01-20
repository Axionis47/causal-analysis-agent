# Causal Analysis Agent

A production-grade multi-agent system for end-to-end causal inference on Kaggle datasets.

## Overview

This project implements a multi-agent causal analysis system that:
- Downloads and analyzes datasets from Kaggle
- Performs exploratory data analysis (EDA)
- Discovers causal relationships using multiple algorithms
- Estimates treatment effects with various methods
- Validates findings through refutation tests
- Generates comprehensive reports

## Tech Stack

- **Backend**: Python 3.11, FastAPI, Celery
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **Database**: PostgreSQL 15, Redis 7
- **Agent Framework**: LangGraph, LangSmith
- **Causal Libraries**: DoWhy, causal-learn, EconML

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 20+
- Poetry (for Python dependency management)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd causal-analysis-agent

# Copy environment variables
cp .env.example .env

# Install all dependencies
make install
```

### 2. Start Development Environment

```bash
# Start all services with Docker Compose
make dev

# Or start in detached mode
make dev-docker
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Development Commands

```bash
# Install dependencies
make install              # Install all (backend + frontend)
make install-backend      # Backend only
make install-frontend     # Frontend only

# Development
make dev                  # Start with Docker (attached)
make dev-docker          # Start with Docker (detached)
make stop                # Stop all services
make logs                # View logs

# Testing
make test                # Run all tests
make test-backend        # Backend tests only
make test-frontend       # Frontend tests only

# Code Quality
make lint                # Run all linters
make lint-backend        # Backend linting (Ruff, Black)
make lint-frontend       # Frontend linting (ESLint, Prettier)
make format              # Format all code

# Build
make build               # Build Docker images
make clean               # Clean build artifacts
```

## Project Structure

```
.
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # Application entry point
│   │   ├── core/           # Configuration and utilities
│   │   ├── tasks/          # Celery background tasks
│   │   └── worker.py       # Celery worker configuration
│   ├── tests/              # Backend tests
│   └── pyproject.toml      # Python dependencies
├── frontend/               # Next.js frontend
│   ├── src/
│   │   └── app/           # Next.js App Router
│   └── package.json       # Node.js dependencies
├── infrastructure/         # Infrastructure configurations
├── shared/                # Shared utilities and types
├── docker-compose.yml     # Local development services
├── Makefile              # Development commands
└── .env.example          # Environment variables template
```

## Configuration

Copy `.env.example` to `.env` and configure the required variables:

- **Database**: PostgreSQL connection string
- **Redis**: Redis connection URL
- **LLM Providers**: API keys for Vertex AI, OpenAI, Anthropic
- **LangSmith**: API key for observability
- **Kaggle**: API credentials for dataset access

## Documentation

- `docs/GCP_SETUP.md` - GCP project and resource setup
- `docs/GITHUB_SETUP.md` - GitHub repository configuration
- `docs/PRODUCTION_READINESS.md` - Deployment validation checklist
- `docs/ANALYSIS_TYPES.md` - Supported causal analysis types

## Agentic Evaluations

```bash
cd backend
poetry run python -m evals.run --quick
```

## License

MIT License
