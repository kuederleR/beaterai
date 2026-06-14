import os
import time
import threading
import numpy as np

DEEPSREAM_AVAILABLE = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    Gst.init(None)
    DEEPSREAM_AVAILABLE = True
except (ImportError, ValueError):
    pass


def _gst_pipeline_string(source, width=1280, height=720, fps=30):
    if source.startswith("/dev/video"):
        return (
            f"nvarguscamerasrc sensor-id={source[-1]} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"framerate={fps}/1 ! "
            f"nvvidconv ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink name=appsink0"
        )
    return (
        f"filesrc location={source} ! "
        f"qtdemux ! "
        f"h264parse ! "
        f"nvv4l2decoder ! "
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink name=appsink0"
    )


class DeepStreamPipeline:
    def __init__(self, video_source="/dev/video0", width=1280, height=720, fps=30):
        self.video_source = video_source
        self.width = width
        self.height = height
        self.fps = fps
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None
        self._pipeline = None

    def _run_gstreamer(self):
        pipeline_str = _gst_pipeline_string(
            self.video_source, self.width, self.height, self.fps
        )
        print(f"[DeepStream] Starting GStreamer pipeline: {pipeline_str}", flush=True)

        self._pipeline = Gst.parse_launch(pipeline_str)
        appsink = self._pipeline.get_by_name("appsink0")
        appsink.set_property("emit-signals", True)
        appsink.set_property("max-buffers", 2)
        appsink.set_property("drop", True)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()

        def on_sample(sink):
            sample = sink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK
            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            w = structure.get_value("width")
            h = structure.get_value("height")
            result, map_info = buf.map(Gst.MapFlags.READ)
            if result:
                arr = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, 3)
                with self._lock:
                    self._frame = arr.copy()
                buf.unmap(map_info)
            return Gst.FlowReturn.OK

        appsink.connect("pull-sample", on_sample)

        self._pipeline.set_state(Gst.State.PLAYING)
        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._pipeline.set_state(Gst.State.NULL)
            loop.quit()

    def start(self):
        if not DEEPSREAM_AVAILABLE:
            print("[DeepStream] GStreamer not available, use OpenCV fallback", flush=True)
            return False
        self._running = True
        self._thread = threading.Thread(target=self._run_gstreamer, daemon=True)
        self._thread.start()
        return True

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._running = False
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        print("[DeepStream] Pipeline released", flush=True)


class OpenCVCapture:
    def __init__(self, video_source="/dev/video0", width=1280, height=720, fps=30, dev_video_path=None):
        self.video_source = video_source
        self.width = width
        self.height = height
        self.fps = fps
        self.dev_video_path = dev_video_path
        self._cap = None

    def start(self):
        import cv2
        if self.dev_video_path and os.path.exists(self.dev_video_path):
            print(f"[OpenCV] Looping video from {self.dev_video_path}", flush=True)
            self._cap = cv2.VideoCapture(self.dev_video_path)
        else:
            print(f"[OpenCV] Opening camera {self.video_source}", flush=True)
            self._cap = cv2.VideoCapture(self.video_source)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        import cv2
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret and self.dev_video_path:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self._cap.read()
        if ret:
            return True, frame
        return False, None

    def release(self):
        if self._cap is not None:
            self._cap.release()


def create_pipeline(video_source="/dev/video0", width=1280, height=720,
                    fps=30, dev_video_path=None):
    if dev_video_path and os.path.exists(dev_video_path):
        print(f"[Pipeline] Dev video path set, using OpenCV fallback for {dev_video_path}", flush=True)
        return OpenCVCapture(video_source, width, height, fps, dev_video_path)
    if DEEPSREAM_AVAILABLE:
        pipe = DeepStreamPipeline(video_source, width, height, fps)
        if pipe.start():
            return pipe
        print("[Pipeline] DeepStream failed, falling back to OpenCV", flush=True)
    return OpenCVCapture(video_source, width, height, fps, dev_video_path)
