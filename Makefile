.PHONY: docker-build docker-run docker-logs docker-stop

docker-build:
	bash scripts/deploy.sh

docker-run:
	cd docker && docker-compose up -d

docker-logs:
	cd docker && docker-compose logs -f hrm-ocr-api

docker-stop:
	cd docker && docker-compose down
