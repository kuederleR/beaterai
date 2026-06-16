.PHONY: build run jetson-build jetson-run dev-shell jetson-shell diag jetson-diag clean

ROS_DISTRO ?= humble

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

diag:
	docker compose run --rm adas-pipeline python3 /workspace/test_diagnostic.py

jetson-diag:
	docker compose run --rm adas-pipeline-jetson python3 /workspace/test_diagnostic.py

clean:
	rm -rf adas_ws/install adas_ws/build adas_ws/log
