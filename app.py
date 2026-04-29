"""
app.py — Optimised Flask + SocketIO backend for PSD (Platform for Sign Dialogue).

Performance architecture (3-thread pipeline):
──────────────────────────────────────────────
  Thread 1  camera_thread   : Reads frames from webcam as fast as the hardware allows.
                               Overlays the *last known* bounding boxes on every frame.
                               Encodes to JPEG and writes to latest_frame_bytes.
                               Camera FPS is NEVER blocked by inference speed.

  Thread 2  inference_thread: Waits for a new raw frame via a threading.Event.
                               Runs DETR and updates detection_state (boxes + sign).
                               Works as fast as the model + CPU allow, independently.

  MJPEG gen                 : Reads latest_frame_bytes via a threading.Event.
                               Always delivers the freshest annotated JPEG without
                               blocking on a queue.

Additional optimisations applied:
  • Camera forced to 640×480 @ 30 fps — smaller frames, less data to process.
  • cv2.CAP_PROP_BUFFERSIZE = 1 — eliminates the kernel's 3-frame stale buffer.
  • torch.inference_mode() instead of no_grad() — lower overhead, faster autograd skip.
  • JPEG quality 60 — roughly 2× faster encode/transmit vs quality 90.
  • Stale-overlay pattern — annotated video is always live; detection latency is invisible.

Run from the project root:
    python app.py
"""

import sys
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch
import albumentations as A
from deep_translator import GoogleTranslator
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit

# ---------------------------------------------------------------------------
# sys.path — make src/ importable from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import DETR                          # noqa: E402
from utils.boxes import rescale_bboxes          # noqa: E402
from utils.setup import get_classes, get_colors # noqa: E402

# ---------------------------------------------------------------------------
# Flask + SocketIO
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "Frontend", "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "Frontend", "static"),
)
app.config["SECRET_KEY"] = "psd-sign-dialogue-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.80   # detection confidence gate
HOLD_THRESHOLD = 15     # inference frames a sign must be held to commit to sentence
CAMERA_W       = 640    # force camera resolution (reduces per-frame data volume)
CAMERA_H       = 480
JPEG_QUALITY   = 60     # lower = faster encode/transmit, acceptable visual quality
CHECKPOINT     = os.path.join(PROJECT_ROOT, "pretrained", "4426_model.pt")

# ---------------------------------------------------------------------------
# DETR model — loaded once at startup
# ---------------------------------------------------------------------------
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])

CLASSES = get_classes()
COLORS  = get_colors()

model = DETR(num_classes=len(CLASSES))
model.eval()
model.load_pretrained(CHECKPOINT)

# ---------------------------------------------------------------------------
# Shared state — protected by state_lock
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
shared = {
    "sentence":     [],   # words committed so far
    "current_sign": "",   # sign currently shown in overlay
    "language":     "en", # selected UI language
    "last_sign":    "",   # last sign that started accumulating
    "hold_count":   0,    # consecutive inference frames with the same sign
}

# ---------------------------------------------------------------------------
# Detection state — written by inference_thread, read by camera_thread
# Boxes are in pixel-space coordinates of the CURRENT camera frame.
# ---------------------------------------------------------------------------
detect_lock = threading.Lock()
detection_state: dict = {
    "boxes": [],   # list of (x1, y1, x2, y2, cls_idx, prob_val)
    "sign":  "",
}

# ---------------------------------------------------------------------------
# Latest annotated JPEG — written by camera_thread, read by MJPEG generator
# ---------------------------------------------------------------------------
frame_lock          = threading.Lock()
latest_frame_bytes: bytes = b""
frame_ready         = threading.Event()   # set when a new JPEG is available

# ---------------------------------------------------------------------------
# Latest raw frame — written by camera_thread, read by inference_thread
# Uses an Event instead of a Queue so inference always works on the NEWEST frame.
# ---------------------------------------------------------------------------
raw_lock   = threading.Lock()
latest_raw_frame: Optional[np.ndarray] = None
raw_ready  = threading.Event()            # set when a new raw frame is available

# ---------------------------------------------------------------------------
# Thread 1 — Camera capture + annotation
# ---------------------------------------------------------------------------

def camera_thread() -> None:
    """
    Reads frames from the webcam as fast as possible.
    Overlays the last known detection boxes (stale is fine — they update ~5–15 fps)
    then JPEG-encodes the result and stores it in latest_frame_bytes.
    """
    global latest_raw_frame, latest_frame_bytes

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS,          30)
    # Flush the driver's internal buffer so we always get the current frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # ── Share raw frame with inference thread ────────────────────────────
        with raw_lock:
            latest_raw_frame = frame  # no copy needed; inference thread copies it
        raw_ready.set()

        # ── Annotate with the latest detection boxes ─────────────────────────
        # We draw on a copy so the inference thread's raw frame is never mutated.
        annotated = frame.copy()
        with detect_lock:
            boxes = list(detection_state["boxes"])  # snapshot

        for (x1, y1, x2, y2, cls_idx, prob_val) in boxes:
            color = tuple(COLORS[cls_idx])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASSES[cls_idx]} {prob_val:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx, ly = x1, max(y1 - 8, 0)
            cv2.rectangle(annotated, (lx, ly - th - 4), (lx + tw, ly + 4), color, -1)
            cv2.putText(annotated, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # ── JPEG encode & publish ────────────────────────────────────────────
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with frame_lock:
                latest_frame_bytes = buf.tobytes()
            frame_ready.set()

    cap.release()


# ---------------------------------------------------------------------------
# Thread 2 — DETR inference
# ---------------------------------------------------------------------------

def inference_thread() -> None:
    """
    Runs DETR on the latest raw frame whenever camera_thread delivers one.
    Updates detection_state and shared sentence state.
    """
    global latest_raw_frame

    while True:
        # Block until camera_thread signals a new frame
        raw_ready.wait()
        raw_ready.clear()

        # Grab the latest frame (may have been updated multiple times since last inference)
        with raw_lock:
            frame = latest_raw_frame
        if frame is None:
            continue

        h, w = frame.shape[:2]

        # ── DETR inference ───────────────────────────────────────────────────
        with torch.inference_mode():
            transformed = transforms(image=frame)
            tensor = torch.unsqueeze(transformed["image"], dim=0)
            result = model(tensor)

        probs = result["pred_logits"].softmax(-1)[:, :, :-1]
        max_probs, max_classes = probs.max(-1)
        keep_mask = max_probs > CONF_THRESHOLD

        batch_idx, query_idx = torch.where(keep_mask)
        bboxes  = rescale_bboxes(result["pred_boxes"][batch_idx, query_idx, :], (w, h))
        classes = max_classes[batch_idx, query_idx]
        probas  = max_probs[batch_idx, query_idx]

        # ── Build pixel-space box list + dominant sign ───────────────────────
        boxes_out: list = []
        top_sign  = ""
        best_conf = -1.0

        for bclass, bprob, bbox in zip(classes, probas, bboxes):
            cls_idx  = int(bclass.item())
            prob_val = float(bprob.item())
            x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            boxes_out.append((x1, y1, x2, y2, cls_idx, prob_val))
            if prob_val > best_conf:
                best_conf = prob_val
                top_sign  = CLASSES[cls_idx]

        # ── Update detection state for camera thread ─────────────────────────
        with detect_lock:
            detection_state["boxes"] = boxes_out
            detection_state["sign"]  = top_sign

        # ── Sign stabilisation → sentence building ───────────────────────────
        with state_lock:
            shared["current_sign"] = top_sign

            if top_sign and top_sign == shared["last_sign"]:
                shared["hold_count"] += 1
            else:
                shared["hold_count"] = 1
                shared["last_sign"]  = top_sign

            if top_sign and shared["hold_count"] == HOLD_THRESHOLD:
                shared["sentence"].append(top_sign)
                _emit_status()


# ---------------------------------------------------------------------------
# Status broadcast loop — keeps the UI sign overlay fresh at ~10 fps
# ---------------------------------------------------------------------------

def status_broadcast_loop() -> None:
    while True:
        time.sleep(0.1)
        with state_lock:
            _emit_status()


def _emit_status() -> None:
    """Emit update_status. Must be called with state_lock held."""
    socketio.emit(
        "update_status",
        {
            "sign":     shared["current_sign"],
            "sentence": " ".join(shared["sentence"]),
        },
    )


# ---------------------------------------------------------------------------
# Start background threads
# ---------------------------------------------------------------------------
threading.Thread(target=camera_thread,       daemon=True, name="camera").start()
threading.Thread(target=inference_thread,    daemon=True, name="inference").start()
threading.Thread(target=status_broadcast_loop, daemon=True, name="status").start()

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


def _mjpeg_generator():
    """Yield MJPEG frames as fast as camera_thread produces them."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        frame_ready.wait(timeout=1.0)   # at most 1 s stall if camera dies
        frame_ready.clear()
        with frame_lock:
            jpg = latest_frame_bytes
        if jpg:
            yield boundary + jpg + b"\r\n"


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

# ---------------------------------------------------------------------------
# Socket.IO event handlers
# ---------------------------------------------------------------------------

@socketio.on("connect")
def on_connect():
    with state_lock:
        emit("update_status", {
            "sign":     shared["current_sign"],
            "sentence": " ".join(shared["sentence"]),
        })


@socketio.on("command")
def on_command(data):
    """backspace — remove last word;  clear — wipe everything."""
    action = data.get("action", "")
    with state_lock:
        if action == "backspace" and shared["sentence"]:
            shared["sentence"].pop()
        elif action == "clear":
            shared["sentence"].clear()
            shared["current_sign"] = ""
            shared["last_sign"]    = ""
            shared["hold_count"]   = 0
        _emit_status()


@socketio.on("set_language")
def on_set_language(data):
    lang = data.get("language", "en")
    with state_lock:
        shared["language"] = lang
    print(f"[SocketIO] Language set to: {lang}")


@socketio.on("translate_now")
def on_translate_now(data):
    """Translate text and emit stt_translation to the requesting client."""
    text   = data.get("text", "")
    target = data.get("target", "hi")
    if not text:
        emit("stt_translation", {"translated": ""})
        return
    try:
        translated = GoogleTranslator(source="auto", target=target).translate(text)
    except Exception as exc:
        print(f"[WARN] Translation failed: {exc}")
        translated = text
    emit("stt_translation", {"translated": translated})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting PSD — Platform for Sign Dialogue")
    print("Open http://localhost:5000 in your browser.")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
