# variables
NAME = fastapi
PORT = 8000:8000
IMAGE_NAME = fast:v1

# run container
run:
	docker run --rm -p $(PORT) --name $(NAME) -d $(IMAGE_NAME)

# build image
build:
	docker build --rm -t $(IMAGE_NAME) .

stop:
	docker stop ${NAME}

compose-build:
	docker compose build
compose-up:
	docker compose up -d

compose-log:
	docker compose log