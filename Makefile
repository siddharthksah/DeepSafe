.PHONY: install start stop clean test lint help

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies (requires Docker)"
	@echo "  make start     - Start the application"
	@echo "  make stop      - Stop the application"
	@echo "  make clean     - Remove containers and artifacts"
	@echo "  make test      - Run system tests"
	@echo "  make lint      - Run linting (placeholder)"

install:
	@echo "Building Docker images..."
	docker-compose build

start:
	@echo "Starting DeepSafe..."
	docker-compose up -d
	@echo "DeepSafe is running at http://localhost:80"

stop:
	@echo "Stopping DeepSafe..."
	docker-compose down

clean:
	@echo "Cleaning up..."
	docker-compose down -v
	rm -rf __pycache__
	rm -rf .pytest_cache

test:
	@echo "Running system tests..."
	# Ensure API is running before testing
	docker-compose up -d api
	docker cp test_system.py deepsafe-api:/app/
	docker cp test_samples deepsafe-api:/app/
	docker exec deepsafe-api python test_system.py

lint:
	@echo "Linting code..."
	# Placeholder for linting command
	@echo "No linter configured yet. Run 'pip install black flake8' and configure."
