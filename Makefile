.PHONY: setup test run backup clean docker-build docker-up docker-down docker-logs docker-dev-build docker-dev-up docker-dev-down docker-clean

setup:
	python scripts/setup.py

test: setup
	pytest tests/

run: setup
	python frontend/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -r {} +

reset: clean
	conda remove -n Container_OCR --all
	rm -rf frontend/static/images/*
	rm -rf frontend/static/evidence/*
	rm -rf logs/*
	python scripts/setup.py

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-dev-build:
	docker-compose -f docker-compose.dev.yml build

docker-dev-up:
	docker-compose -f docker-compose.dev.yml up -d

docker-dev-down:
	docker-compose -f docker-compose.dev.yml down

docker-clean:
	docker-compose down -v
	docker-compose -f docker-compose.dev.yml down -v 