import cv2
import numpy as np
import os

from trt_runner import TrtRunner, TRT_AVAILABLE


class TwinLiteDetector:
    def __init__(self, engine_path="models/twinlite.engine",
                 crop_y=28, crop_h=360):
        self.input_width = 640
        self.input_height = 360
        self.crop_y = crop_y
        self.crop_h = crop_h
        self.trt_runner = None

        if TRT_AVAILABLE and os.path.exists(engine_path):
            try:
                print(f"[TwinLite] Loading TensorRT engine from {engine_path}", flush=True)
                self.trt_runner = TrtRunner(engine_path)
                print(f"[TwinLite] TensorRT engine ready.", flush=True)
            except Exception as e:
                print(f"[TwinLite] Failed to load TRT engine: {e}", flush=True)
        else:
            print(f"[TwinLite] Engine {engine_path} not found. Auto-building...", flush=True)
            import subprocess
            subprocess.run(["python3", "build_engines.py"])
            if os.path.exists(engine_path):
                self.trt_runner = TrtRunner(engine_path)
                print(f"[TwinLite] TensorRT engine built and ready.", flush=True)

    def _preprocess(self, img):
        h, w = img.shape[:2]
        crop_y = min(self.crop_y, max(0, h - self.crop_h))
        roi = img[crop_y:crop_y + self.crop_h, :]
        roi_resized = cv2.resize(roi, (self.input_width, self.input_height),
                                 interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        normed = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normed, (2, 0, 1))[np.newaxis, ...]
        return chw.astype(np.float32)

    def detect(self, img):
        if self.trt_runner is None:
            return None, None

        h, w = img.shape[:2]
        inp = self._preprocess(img)
        outputs = self.trt_runner.infer(inp)
        # Outputs from TensorRT may not preserve ONNX output_names order.
        # Auto-detect: lane lines are sparse (~1-5% of pixels), drivable area is dense.
        # Compare sparsity to decide which is which.
        def _sparsity(logits):
            if logits.ndim == 4:
                cls = np.argmax(logits, axis=1)[0]
            elif logits.ndim == 3:
                cls = np.argmax(logits, axis=0)
            else:
                return 0.5
            return float(np.mean(cls > 0))

        s0, s1 = _sparsity(outputs[0]), _sparsity(outputs[1])
        if s0 < s1:
            ll_raw, da_raw = outputs[0], outputs[1]
        else:
            da_raw, ll_raw = outputs[0], outputs[1]

        def _argmax_mask(logits):
            if logits.ndim == 4:
                return np.argmax(logits, axis=1)[0].astype(np.uint8)
            elif logits.ndim == 3:
                return np.argmax(logits, axis=0).astype(np.uint8)
            return np.zeros((self.input_height, self.input_width), dtype=np.uint8)

        da_mask_small = _argmax_mask(da_raw)
        ll_mask_small = _argmax_mask(ll_raw)

        crop_y = min(self.crop_y, max(0, h - self.crop_h))
        da_full = np.zeros((h, w), dtype=np.uint8)
        ll_full = np.zeros((h, w), dtype=np.uint8)

        da_resized = cv2.resize(da_mask_small, (w, self.crop_h),
                                interpolation=cv2.INTER_NEAREST)
        ll_resized = cv2.resize(ll_mask_small, (w, self.crop_h),
                                interpolation=cv2.INTER_NEAREST)

        da_full[crop_y:crop_y + self.crop_h, :] = da_resized
        ll_full[crop_y:crop_y + self.crop_h, :] = ll_resized

        return ll_full, da_full

    @staticmethod
    def lanes_from_mask(ll_mask, car_center_x, min_points=4, max_gap=15):
        h, w = ll_mask.shape[:2]
        bot_row = h - 1
        top_row = h // 3

        class _Track:
            def __init__(self, x, y):
                self.pts = [(float(x), float(y))]
                self.gap = 0
                self._vx = 0.0
            def add(self, x, y):
                prev = self.pts[-1]
                dy = prev[1] - y
                if dy > 0:
                    self._vx = 0.3 * self._vx + 0.7 * (x - prev[0]) / dy
                self.pts.append((float(x), float(y)))
                self.gap = 0
            def predict_x(self, row):
                if not self.pts:
                    return None
                dy = self.pts[-1][1] - row
                return self.pts[-1][0] + self._vx * dy
            def result(self):
                return np.array(self.pts, dtype=np.float32) if len(self.pts) >= min_points else None

        tracks = []
        finished = []

        for row in range(bot_row, top_row, -1):
            row_data = ll_mask[row, :].astype(np.float32)

            nonzero = np.where(row_data > 0)[0]
            centroids = []
            if len(nonzero) > 0:
                seg_gaps = np.diff(nonzero) > 3
                segments = np.split(nonzero, np.where(seg_gaps)[0] + 1)
                centroids = [float(np.mean(seg)) for seg in segments if len(seg) >= 1]

            max_dist = 15 + 25 * (row - top_row) / max(bot_row - top_row, 1)

            track_list = [(i, t) for i, t in enumerate(tracks)]
            track_list = [(i, t) for i, t in track_list if t.predict_x(row) is not None]
            track_list.sort(key=lambda x: x[1].predict_x(row))
            centroids_sorted = sorted(enumerate(centroids), key=lambda x: x[1])

            matched_tracks = set()
            matched_centroids = set()

            if track_list and centroids_sorted:
                n, m = len(track_list), len(centroids_sorted)
                cost = np.full((n, m), np.inf)
                for ti in range(n):
                    pred = track_list[ti][1].predict_x(row)
                    for ci in range(m):
                        d = abs(centroids_sorted[ci][1] - pred)
                        if d < max_dist:
                            cost[ti, ci] = d - max_dist

                dp = np.full((n + 1, m + 1), np.inf)
                dp[0, :] = 0
                dp[:, 0] = 0
                for ti in range(1, n + 1):
                    for ci in range(1, m + 1):
                        dp[ti, ci] = min(dp[ti - 1, ci], dp[ti, ci - 1])
                        if cost[ti - 1, ci - 1] < np.inf:
                            dp[ti, ci] = min(dp[ti, ci], dp[ti - 1, ci - 1] + cost[ti - 1, ci - 1])

                ti, ci = n, m
                while ti > 0 and ci > 0:
                    if abs(dp[ti, ci] - dp[ti - 1, ci]) < 1e-9:
                        ti -= 1
                    elif abs(dp[ti, ci] - dp[ti, ci - 1]) < 1e-9:
                        ci -= 1
                    else:
                        ti -= 1
                        ci -= 1
                        if cost[ti, ci] < np.inf:
                            matched_tracks.add(track_list[ti][0])
                            matched_centroids.add(centroids_sorted[ci][0])
                            track_list[ti][1].add(centroids_sorted[ci][1], row)

            for i, t in enumerate(tracks):
                if i not in matched_tracks:
                    t.gap += 1

            for i, cx in enumerate(centroids):
                if i not in matched_centroids:
                    tracks.append(_Track(cx, row))

            dead = [t for t in tracks if t.gap > max_gap]
            for t in dead:
                r = t.result()
                if r is not None:
                    finished.append(r)
            tracks = [t for t in tracks if t.gap <= max_gap]

        for t in tracks:
            r = t.result()
            if r is not None:
                finished.append(r)

        return finished

    @staticmethod
    def trace_lane_edges(ll_mask, car_center_x, start_y, max_gap=15):
        h, w = ll_mask.shape[:2]
        start_y = int(np.clip(start_y, 0, h - 1))
        row_data = ll_mask[start_y, :].astype(np.float32)
        nonzero = np.where(row_data > 0)[0]
        if len(nonzero) == 0:
            return None, None

        seg_gaps = np.diff(nonzero) > 3
        segments = np.split(nonzero, np.where(seg_gaps)[0] + 1)

        left_start = None
        right_start = None
        for seg in segments:
            sl, sr = int(seg[0]), int(seg[-1])
            if sr < car_center_x:
                left_start = sr
            elif sl > car_center_x and right_start is None:
                right_start = sl

        def _trace(x_start, side):
            if x_start is None:
                return None
            pts = [(float(x_start), float(start_y))]
            last_x = x_start
            gc = 0
            for row in range(start_y - 1, h // 3, -1):
                row_data = ll_mask[row, :].astype(np.float32)
                if np.max(row_data) == 0:
                    gc += 1
                    if gc > max_gap:
                        break
                    continue
                max_dist = 15 + 25 * (row - h // 3) / max(h - 1 - h // 3, 1)
                wr = int(max_dist)
                x0 = max(0, int(last_x) - wr)
                x1 = min(w, int(last_x) + wr)
                win = row_data[x0:x1]
                nz = np.where(win > 0)[0]
                if len(nz) == 0:
                    gc += 1
                    if gc > max_gap:
                        break
                    continue
                ex = x0 + (nz[-1] if side == 'right' else nz[0])
                pts.append((float(ex), float(row)))
                last_x = ex
                gc = 0
            return np.array(pts, dtype=np.float32) if len(pts) >= 4 else None

        return _trace(left_start, 'right'), _trace(right_start, 'left')
