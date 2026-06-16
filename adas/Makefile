.PHONY: build run jetson-build jetson-run dev-shell clean

ROS_DISTRO ?= jazzy

build:
	docker compose build adas-pipeline

run:
	docker compose run --rm adas-pipeline

jetson-build:
	docker compose build adas-pipeline-jetson

jetson-run:
	docker compose run --rm adas-pipeline-jetson

dev-shell:
	docker compose run --rm adas-pipeline /bin/bash

jetson-shell:
	docker compose run --rm adas-pipeline-jetson /bin/bash

clean:
	rm -rf adas_ws/install adas_ws/build adas_ws/log
