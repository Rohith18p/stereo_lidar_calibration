# Makefile — Docker dev helper for a single container bound to this folder

# ---------- Config (edit if you like) ----------
SHELL := /usr/bin/env bash

# Image / container naming for your calib setup
IMAGE_NAME ?= calib_image
TAG        ?= latest
IMAGE      := $(IMAGE_NAME):$(TAG)

NAME       ?= calib_container

# Host <-> container mount
# Host workspace: ./calib_ws
# Container workspace: /ros_ws
HOST_DIR   := $(abspath calib_ws)
CONT_MOUNT ?= /ros_ws
CONT_WS    ?= $(CONT_MOUNT)   # default working dir in the container

# Enable GPU (requires NVIDIA runtime). Set to 0 to disable.
USE_GPU    ?= 1

# Optional extra volume mounts (space-separated "host:container[:opts]")
EXTRA_VOLUMES ?=

# Dockerfile and build context (your Dockerfile is in calib_dock/)
DOCKER_CONTEXT ?= calib_dock
DOCKERFILE     ?= $(DOCKER_CONTEXT)/Dockerfile

# ---------- Derived settings ----------
RUN_OPTS = --name $(NAME) --hostname $(NAME) --privileged --net=host \
	-e DISPLAY=$$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v "$(HOST_DIR):$(CONT_MOUNT):rw" \
	$(foreach vol,$(EXTRA_VOLUMES),-v $(vol))

ifeq ($(USE_GPU),1)
RUN_OPTS += --gpus all
endif

# Helpers to test current state
container_exists  = $(shell docker ps -a --filter "name=^/$(NAME)$$" --format '{{.Names}}')
container_running = $(shell docker ps    --filter "name=^/$(NAME)$$" --format '{{.Names}}')
image_exists      = $(shell docker image inspect "$(IMAGE)" >/dev/null 2>&1 && echo yes || echo no)

# ---------- Targets ----------
.PHONY: help
help:
	@echo "Targets:"
	@echo "  make up        - Build if needed, create container if missing, start if stopped"
	@echo "  make up-shell  - Same as 'up' then open an interactive shell"
	@echo "  make shell     - Open another interactive shell in the running container"
	@echo "  make build     - docker build (uses $(DOCKERFILE) with context $(DOCKER_CONTEXT))"
	@echo "  make rebuild   - docker build --no-cache"
	@echo "  make stop      - Stop the container"
	@echo "  make down      - Stop and remove the container"
	@echo "  make nuke      - Remove container and image (full erase)"
	@echo "  make status    - Show container status"
	@echo "  make logs      - Tail container logs"
	@echo "  make x11-allow - Allow X11 from local docker (Linux GUI)"
	@echo "Variables you can override: IMAGE_NAME, TAG, NAME, CONT_MOUNT, CONT_WS, USE_GPU, EXTRA_VOLUMES, DOCKER_CONTEXT, DOCKERFILE"

.PHONY: build
build:
	docker build -t "$(IMAGE)" -f "$(DOCKERFILE)" "$(DOCKER_CONTEXT)"

.PHONY: rebuild
rebuild:
	docker build --no-cache -t "$(IMAGE)" -f "$(DOCKERFILE)" "$(DOCKER_CONTEXT)"

.PHONY: up
up:
	@if [ "$(image_exists)" != "yes" ]; then \
		echo "Image $(IMAGE) not found. Building..."; \
		docker build -t "$(IMAGE)" -f "$(DOCKERFILE)" "$(DOCKER_CONTEXT)" || exit $$?; \
	fi; \
	if [ -z "$(container_exists)" ]; then \
		echo "Creating and starting container $(NAME)..."; \
		docker run -it -d $(RUN_OPTS) -w "$(CONT_WS)" "$(IMAGE)" bash; \
	elif [ -z "$(container_running)" ]; then \
		echo "Starting existing container $(NAME)..."; \
		docker start "$(NAME)"; \
	else \
		echo "Container $(NAME) is already running."; \
	fi

.PHONY: up-shell
up-shell: up
	@$(MAKE) shell

.PHONY: shell exec
shell exec:
	@if [ -z "$(container_running)" ]; then \
		echo "Container $(NAME) isn't running. Run 'make up' first."; \
		exit 1; \
	fi; \
	docker exec -it "$(NAME)" bash

.PHONY: stop
stop:
	@docker stop "$(NAME)" 2>/dev/null || true

.PHONY: down
down: stop
	@docker rm -f "$(NAME)" 2>/dev/null || true

.PHONY: rmi
rmi:
	@docker rmi -f "$(IMAGE)" 2>/dev/null || true

.PHONY: nuke
nuke: down rmi
	@echo "All cleared (container and image removed)."

.PHONY: status
status:
	@docker ps -a --filter "name=^/$(NAME)$$"

.PHONY: logs
logs:
	@docker logs -f "$(NAME)"

.PHONY: x11-allow
x11-allow:
	@xhost +local:docker || true
