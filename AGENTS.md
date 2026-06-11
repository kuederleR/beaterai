# AGENTS.md

This repository contains a Python dashcam application that combines Flask, OpenCV, Torch, and road-geometry logic. The main runtime lives in `dashcam_app/`, with most behavior concentrated in `dashcam_app/dashcam.py` and model inference in `dashcam_app/yolop_detector.py`.

This file is written for a small local model such as gemma4-4b. The priority is disciplined execution: understand the exact slice being changed, break work into very small tasks, validate after each edit, and avoid broad refactors.

## Operating mode

When working in this repository, follow this process every time:

1. Restate the task in one or two sentences.
2. Identify the smallest concrete anchor first.
3. Read only the nearby code needed to form one local hypothesis.
4. Write a short task list before making any edit.
5. Make one small edit.
6. Run one focused validation step immediately.
7. Iterate only if the validation result justifies it.

Do not start coding immediately after reading the request. Planning is required.

## Required pre-edit checklist

Before writing code, produce a short checklist that includes all of the following:

- The exact file or function that most likely controls the behavior.
- One falsifiable hypothesis about the bug or requested change.
- One cheap validation step that can disconfirm that hypothesis.
- A list of micro-tasks, each small enough to complete in a single edit.

Use task sizes like these:

- Read one file section.
- Change one function.
- Adjust one constant.
- Add one guard clause.
- Update one validation script.

Avoid task sizes like these:

- Rewrite the lane pipeline.
- Clean up the whole file.
- Refactor geometry and inference together.

## Repo map

- `dashcam_app/dashcam.py`: Main Flask app, camera loop, ADAS state, lane logic, road-plane projection, and API payload assembly.
- `dashcam_app/yolop_detector.py`: YOLOPv2 wrapper, preprocessing, model loading, and detection postprocessing.
- `dashcam_app/templates/index.html`: Web UI.
- `dashcam_app/calibration.yaml`: Camera calibration data.
- `dashcam_app/test_*.py`: Small standalone diagnostic scripts for geometry, warping, payload construction, and model-shape checks.
- `dashcam_app/docker-compose.yml`: Main containerized runtime, GPU-oriented.
- `weights/` and `data/weights/`: Model assets.

## Project facts that matter

- This project is not organized as a large package with deep module boundaries. Much of the behavior is concentrated in `dashcam_app/dashcam.py`.
- Many test files are executable scripts, not formal unit tests. Prefer running a single relevant script over inventing a broad test plan.
- The main web app runs on port `5001`.
- The container setup expects NVIDIA runtime and may bind `/dev/video0`.
- The app can run against a real camera or a recorded video file via `DEV_VIDEO_PATH`.

## Local validation rules

This application is designed to be run on a remote host and should not be executed locally. After the first substantive edit, run the narrowest useful validation immediately. Commit and push changes using git, and ask the user to pull and test the code before continuing. 

Prefer validation in this order:

1. A single relevant diagnostic script in `dashcam_app/`.
2. A direct Python syntax or import check for the touched file.
3. A narrow app run if the change affects runtime behavior.

## Change-scope rules

- Prefer minimal changes over cleanup.
- Do not reformat large files without need.
- Do not rename functions or move code across files unless the task requires it.
- Do not modify unrelated tests just because they look weak.
- If a file is large, edit the narrowest block possible.
- Preserve existing behavior outside the requested slice.

## Geometry and lane-processing cautions

These project-specific constraints are easy to break. Treat them as high priority:

- Avoid per-column lane-line extension in `dashcam_app/dashcam.py`; it can create false divider lines.
- Road-plane projection must build homography from an undistorted vanishing point.
- Feed `image_to_road` raw inference pixels exactly once through undistortion.
- Do not double-undistort lane points, FCW points, or other image coordinates before BEV projection.

If a task touches vanishing point logic, homography updates, lane boundaries, or BEV payloads, validate with the most relevant geometry script before making additional edits.

## Guidance for a small model

Because gemma4-4b is small, use stricter execution discipline than usual:

- Keep working memory small. Focus on one file and one behavior at a time.
- Prefer reading 50 to 200 lines around the controlling function, not the entire repository.
- Write down assumptions before editing.
- If two explanations seem plausible, choose the one with the cheapest discriminating check.
- If the first fix fails, do one nearby read, then revise the same slice.
- Avoid speculative architectural improvements.
- Avoid mixing bug fixing, cleanup, and optimization in one pass.

## Task breakdown template

Before coding, use this exact structure in your internal plan:

1. Objective
2. Anchor file/function
3. Local hypothesis
4. Cheap validation
5. Micro-tasks

Example:

1. Objective: Fix incorrect lane overlay payload when one side is missing.
2. Anchor file/function: `dashcam_app/dashcam.py`, payload assembly block.
3. Local hypothesis: The code overwrites guarded point lists with unconditional assignments.
4. Cheap validation: Run `python3 dashcam_app/test_payload.py`.
5. Micro-tasks:
   - Read payload assembly block.
   - Change only the overwrite logic.
   - Run the payload script.
   - Stop and reassess if output still looks wrong.

## Execution discipline

- Never jump from the user request directly to a large patch.
- Never make more than one conceptual change before the first validation.
- If validation fails, repair the same local slice first.
- If validation succeeds, decide whether any adjacent edit is still necessary.
- Stop when the requested behavior is satisfied and the relevant check passes.

## Runtime notes

- Main application entry point: `dashcam_app/dashcam.py`
- Flask port: `5001`
- Common dependencies: Flask, Werkzeug, OpenCV, Torch, NumPy, SciPy
- GPU and camera access are environment-sensitive; do not assume they are available in every local run

## Final rule

Think in small reversible steps. Read narrowly, plan explicitly, edit minimally, validate immediately.