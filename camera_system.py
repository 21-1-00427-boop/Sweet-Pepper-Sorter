"""
camera_system.py
─────────────────
Manages both Picamera2 instances, YOLO inference, and colour analysis.

Two separate detection paths
────────────────────────────
  1. LIVE boxes (visual only) — runs continuously at LIVE_DETECT_FPS in the
     camera pipeline. Draws bounding boxes on the stream so the operator can
     see the pepper being tracked. NO colour analysis, NO decisions.

  2. DECISION MATRIX (the real logic) — runs ONLY during the stepper pause
     window. Up to MAX_INFERENCES passes, locks on CONSENSUS_NEEDED matching
     votes, runs colour analysis, pushes a SortDecision. This is what drives
     accept/reject and the colour mask.

Decision pipeline (cam 1 → cam 0)
──────────────────────────────────
  • CAM 1 sees the pepper first → decision pushed; if GOOD, colour carried.
  • CAM 0 sees the same pepper after the belt advances → decision pushed.
  • CAM 1 colour label displayed alongside CAM 0 result on the dashboard.
"""

from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import queue
import time

from stepper_controller import SortDecision

try:
    from telemetry_publisher import TelemetryPublisher
except Exception:
    TelemetryPublisher = None

# ── Imaging configuration ──────────────────────────────────────────────────
MODEL_PATH   = "/home/scrapcode/Downloads/Pepper_v3/fine_tune_v3_ncnn_model"
IMGSZ        = 640
CONF         = 0.4
CAPTURE_SIZE = (1080, 1080)
DISPLAY_SIZE = (640, 480)
TARGET_FPS   = 30

# How often the LIVE (visual-only) detector runs, in frames-per-second.
# Keep this low (2-4) so it does not overload the Pi while the motors run.
LIVE_DETECT_FPS = 3

CLAHE_CLIP   = 2.0
CLAHE_GRID   = (8, 8)

# ── Decision window settings ───────────────────────────────────────────────
# During the pause the model detects the pepper continuously. Every detection
# contributes its colour reading to a running total; at the end of the window
# the readings are AVERAGED into one final result that counts as 1 inference.
DETECT_INTERVAL = 0.15   # seconds between detections during the pause window

# ── Sort acceptance rules ──────────────────────────────────────────────────
# The TRAINED YOLO model decides good vs bad directly via its class:
#   {0: 'Good_Pepper', 1: 'Bad_Pepper'}
# Colour analysis is used ONLY to report the dominant colour, NOT to judge
# acceptance. A pepper the model calls "Good_Pepper" is accepted even if green.
GOOD_CLASS_NAME = "Good_Pepper"   # any other class is treated as bad/reject

STAGE_COLORS = {
    "Red":    ( 60,  60, 220),
    "Orange": (  0, 140, 255),
    "Yellow": (  0, 220, 220),
    "Green":  ( 50, 200,  50),
    "Unknown":(180, 180, 180),
}

# ── Shared CV objects ──────────────────────────────────────────────────────
_clahe        = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
_kernel       = np.ones((3, 3),   np.uint8)
_large_kernel = np.ones((15, 15), np.uint8)


# ══════════════════════════════════════════════════════════════════════════
# ── Vision helpers ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _enhance(frame):
    lab     = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([_clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def _analyse_colors(crop_rgb):
    total = crop_rgb.shape[0] * crop_rgb.shape[1]
    if total == 0:
        return {n: {"count": 0, "pct": 0.0}
                for n in ("Red", "Orange", "Yellow", "Green")}, np.zeros((1, 1), np.uint8)

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    fg_gate   = (s_ch >= 55) & (v_ch >= 40)
    init_mask = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
    init_mask[fg_gate] = 255

    contours, _ = cv2.findContours(init_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_fg    = np.zeros_like(init_mask, dtype=bool)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            pm = np.zeros_like(init_mask)
            cv2.drawContours(pm, [largest], -1, 255, -1)
            pm       = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, _large_kernel)
            clean_fg = (pm == 255)

    masks = {
        "Red":    clean_fg & ((h_ch <= 10) | (h_ch >= 160)),
        "Orange": clean_fg & (h_ch >= 8)  & (h_ch <= 16),
        "Yellow": clean_fg & (h_ch >= 16) & (h_ch <= 23),
        "Green":  clean_fg & (h_ch >= 24) & (h_ch <= 59),
    }
    combined = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
    for m in masks.values():
        combined[m] = 255
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  _kernel)

    color_data = {
        name: {"count": int(mask.sum()), "pct": mask.sum() / total * 100}
        for name, mask in masks.items()
    }
    return color_data, combined


def _infer_maturity(cd):
    pcts  = {n: d["pct"] for n, d in cd.items()}
    total = sum(pcts.values())
    if total < 2.0:
        return "Unknown", "No Object"
    dom, dom_pct = max(pcts.items(), key=lambda x: x[1])
    if dom == "Green" and dom_pct > total * 0.90:
        return "Green", "Immature"
    if pcts["Green"] > 5.0 and any(pcts[c] > 5.0 for c in ("Red", "Orange", "Yellow")):
        target = max(("Red", "Orange", "Yellow"), key=lambda k: pcts[k])
        return f"Breaker ({target})", "Turning"
    if dom in ("Red", "Orange", "Yellow"):
        return dom, "Ripe"
    return dom, "Indeterminate"


def _is_good(yolo_class_name):
    """Accept based on the trained model's class, NOT colour."""
    return yolo_class_name == GOOD_CLASS_NAME


# ── 4-category colour sorting (only for peppers that PASS CAM 0) ────────────
CATEGORY_DOMINANCE = 60.0   # a colour must reach this % (normalised) to be pure
CAM1_WEIGHT        = 0.6    # intake camera weighted more
CAM0_WEIGHT        = 0.4    # final camera

# ── Size measurement (measured at CAM 1 only, from the colour mask) ─────────
# Pepper area = number of foreground pixels in the isolation mask, averaged
# over the pause window. Calibrate SIZE_LARGE_THRESHOLD by watching the
# "[CAM 1] measured area" line printed in the console for known peppers.
SIZE_LARGE_THRESHOLD = 75000   # px area >= this → "Large", else "Medium"

# ── FIFO transit / desync sanity checks ────────────────────────────────────
# A pepper takes this many stepper cycles to travel from CAM 1 to CAM 0.
# Each queued pepper records its enqueue cycle; at CAM 0 we verify it arrived
# about this many cycles later. A mismatch flags a likely missed/false
# detection that could desync the queue.
CAM1_TO_CAM0_CYCLES = 2
ARRIVAL_TOLERANCE   = 1     # allow +/- this many cycles before flagging

# ── Reject paddle (Ch 6) ────────────────────────────────────────────────────
# A pepper rejected at CAM 1 reaches the Ch 6 paddle station this many cycles
# later (fired during that cycle's pause).
CAM1_TO_PADDLE_CYCLES = 1

# ── Carousel bins ───────────────────────────────────────────────────────────
# Bins sit one cycle apart along the conveyor after CAM 0. A trapdoor releases
# the pepper when it has travelled the matching number of cycles. The value is
# the number of cycles AFTER CAM 0 at which that bin's trapdoor sits.
BIN_CYCLE_OFFSET = {
    "Reject":        1,
    "Green Large":   2,
    "Green Medium":  3,
    "Red Large":     4,
    "Red Medium":    5,
    "Orange Large":  6,
    "Orange Medium": 7,
    "Mix Large":     8,
    "Mix Medium":    9,
}

# The 9 physical bins (order matches the conveyor layout)
BIN_NAMES = list(BIN_CYCLE_OFFSET.keys())

def _classify_category(cam1_pct, cam0_pct):
    """
    Combine CAM 1 and CAM 0 colour distributions (weighted) and decide the
    final category: 'Red', 'Green', 'Orange', or 'Mix'.

    cam1_pct / cam0_pct : dict like {'Red':%, 'Orange':%, 'Yellow':%, 'Green':%}
                          (either may be None if that camera had no reading)

    Rules:
      • Distributions are weighted (CAM1 0.6 / CAM0 0.4) then normalised so
        Red+Orange+Yellow+Green = 100%.
      • If Red / Green / Orange individually reach CATEGORY_DOMINANCE → that
        category. Yellow can NEVER form a pure category (it pushes to Mix).
      • Otherwise → 'Mix'.
    """
    colors = ("Red", "Orange", "Yellow", "Green")

    # Handle a missing camera reading gracefully
    if cam1_pct is None and cam0_pct is None:
        return "Mix", {c: 0.0 for c in colors}
    if cam1_pct is None:
        w1, w0 = 0.0, 1.0
        cam1_pct = {c: 0.0 for c in colors}
    elif cam0_pct is None:
        w1, w0 = 1.0, 0.0
        cam0_pct = {c: 0.0 for c in colors}
    else:
        w1, w0 = CAM1_WEIGHT, CAM0_WEIGHT

    # Weighted combine
    combined = {c: w1 * cam1_pct.get(c, 0.0) + w0 * cam0_pct.get(c, 0.0)
                for c in colors}

    # Normalise so the four colours sum to 100%
    total = sum(combined.values())
    if total <= 0:
        return "Mix", {c: 0.0 for c in colors}
    norm = {c: combined[c] / total * 100.0 for c in colors}

    # Pure categories: Red / Green / Orange only
    for cat in ("Red", "Green", "Orange"):
        if norm[cat] >= CATEGORY_DOMINANCE:
            return cat, norm

    return "Mix", norm


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _merge_boxes(detections):
    if not detections:
        return []
    boxes  = [{"cls": int(b.cls[0]), "conf": float(b.conf[0]),
                "xyxy": list(map(float, b.xyxy[0]))} for b in detections]
    n      = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if boxes[i]["cls"] == boxes[j]["cls"] and \
               _iou(boxes[i]["xyxy"], boxes[j]["xyxy"]) >= 0.4:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for members in groups.values():
        coords = [boxes[k]["xyxy"] for k in members]
        merged.append({
            "cls":  boxes[members[0]]["cls"],
            "conf": sum(boxes[k]["conf"] for k in members) / len(members),
            "xyxy": [min(c[0] for c in coords), min(c[1] for c in coords),
                     max(c[2] for c in coords), max(c[3] for c in coords)],
        })
    return merged


def _draw_box(frame, yolo_label, conf, stage, maturity, x1, y1, x2, y2, color):
    blen = max(min((x2 - x1), (y2 - y1)) // 4, 8)
    for pts in [
        [(x1, y1+blen),(x1, y1),(x1+blen, y1)],
        [(x2-blen, y1),(x2, y1),(x2, y1+blen)],
        [(x1, y2-blen),(x1, y2),(x1+blen, y2)],
        [(x2-blen, y2),(x2, y2),(x2, y2-blen)],
    ]:
        cv2.polylines(frame, [np.array(pts)], False, color, 2, cv2.LINE_AA)
    font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    for i, (txt, col) in enumerate(reversed([
        (f"{yolo_label} {conf:.0%}", (200, 255, 180)),
        (f"{stage} {maturity}", color),
    ])):
        (_, th_px), _ = cv2.getTextSize(txt, font, sc, th)
        ty = max(y1-4, 24) - i*(th_px+3)
        cv2.putText(frame, txt, (x1+1, ty+1), font, sc, (0,0,0), th+1, cv2.LINE_AA)
        cv2.putText(frame, txt, (x1,   ty),   font, sc, col,     th,   cv2.LINE_AA)


def _draw_color_bar(frame, color_data, x1, y1, x2, y2):
    bx1, bx2 = x1+4, x2-4
    by, bh   = y2-10, 6
    bw       = max(bx2-bx1, 1)
    total    = sum(d["pct"] for d in color_data.values())
    if total == 0:
        return
    cx = bx1
    for name, col in STAGE_COLORS.items():
        if name == "Unknown" or name not in color_data:
            continue
        sw = int(bw * color_data[name]["pct"] / max(total, 1))
        if sw > 0:
            cv2.rectangle(frame, (cx, by), (cx+sw, by+bh), col, -1)
            cx += sw


# ══════════════════════════════════════════════════════════════════════════
# ── Per-camera state ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

class _CameraState:
    def __init__(self, cam_id):
        self.cam_id            = cam_id
        self.lock              = threading.Lock()
        self.frame_for_det     = None          # latest raw frame (updated by pipeline)
        self.output_main_frame = None          # annotated display frame
        self.output_mask_frame = np.zeros((120, 200, 3), dtype=np.uint8)
        self.presence_state    = {"present": False, "count": 0}

        # LIVE boxes (visual only) — list of (x1,y1,x2,y2,conf) in CAPTURE coords
        self.live_boxes        = []

        self.telemetry         = {
            "cam":       f"CAM_{cam_id}",
            "id":        "None",
            "infer":     "STANDBY",
            "stage":     "Conveyor Empty",
            "class":     "Idle / Listening",
            "breakdown": "R:0.0% O:0.0% Y:0.0% G:0.0%",
            "verdict":   "—",
            "action":    "—",
            "votes":     "[]",
            "cam1_color":"—",
        }


# ══════════════════════════════════════════════════════════════════════════
# ── Main class ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

class CameraSystem:
    def __init__(self, result_queue: queue.Queue, stop_flag: threading.Event, stepper):
        self._result_queue = result_queue
        self._stop_flag    = stop_flag
        self._stepper      = stepper

        self.cam_states = [_CameraState(0), _CameraState(1)]

        self._pending_color_lock  = threading.Lock()
        self._pending_color_stage = "Unknown"
        self._pending_maturity    = "Unknown"
        # FIFO queue of peppers that passed CAM 1 and are travelling to CAM 0.
        # Each entry: {"color_dist": {...}|None, "size": "Large"/"Medium",
        #              "dom_color": str}. Strict first-in-first-out: the Nth good
        # pepper at CAM 1 is the Nth pepper CAM 0 evaluates.
        self._transit_queue = []

        # FIFO queue from CAM 0 → bin trapdoors. Each entry:
        # {"bin": str, "target_cycle": int}. Released when current cycle
        # reaches target_cycle (bin's fixed offset after CAM 0).
        self._drop_queue = []

        # FIFO for the Ch 6 reject paddle. When CAM 1 rejects a pepper, an
        # entry {"target_cycle": int} is queued for 1 cycle later (when that
        # pepper reaches the paddle station during the pause).
        self._reject_paddle_queue = []

        self._stats_lock  = threading.Lock()
        self.sorter_stats = {
            "total_accepted":  0,      # GOOD confirmed at CAM 0 (final sorted count)
            "total_rejected":  0,      # BAD at CAM 1 OR CAM 0
            "awaiting_cam0":   False,  # any GOOD peppers from CAM 1 in transit?
            "awaiting_color":  "—",    # next pepper's colour (front of queue)
            "awaiting_size":   "—",    # next pepper's size (front of queue)
            "transit_count":   0,      # how many peppers between CAM 1 and CAM 0
            "desync_warnings": 0,      # count of detection/timing anomalies flagged
            "last_warning":    "—",    # most recent anomaly message
            # Carousel drop tracking: peppers actually deposited into each bin
            "bins": {name: 0 for name in BIN_NAMES},
            "drop_pending":    0,      # results between CAM 0 and the drop point
            "last_drop":       "—",    # most recent bin a pepper dropped into
            "cam1_last_color": "—",
            "cam0_last_color": "—",
            # Final category counts = colour × size (peppers that PASS CAM 0)
            "categories": {
                "Red Large":    0, "Red Medium":    0,
                "Green Large":  0, "Green Medium":  0,
                "Orange Large": 0, "Orange Medium": 0,
                "Mix Large":    0, "Mix Medium":    0,
            },
            "last_category":   "—",    # most recent final "Colour Size" label
        }

        self._active_cam_id   = 0
        self._active_cam_lock = threading.Lock()
        self._model           = None

        # MQTT tally publisher (best-effort; None if paho-mqtt unavailable)
        self._telemetry = TelemetryPublisher() if TelemetryPublisher else None

    def _publish_tally(self):
        """Publish the current tally over MQTT (no-op if change-free/disabled)."""
        if self._telemetry is not None:
            # Pass a shallow copy of the relevant stats
            with self._stats_lock:
                snapshot = {
                    "total_accepted": self.sorter_stats["total_accepted"],
                    "total_rejected": self.sorter_stats["total_rejected"],
                    "bins":           dict(self.sorter_stats["bins"]),
                }
            self._telemetry.publish_tally(snapshot)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load_model(self):
        print("Loading YOLO model...")
        self._model = YOLO(MODEL_PATH, task="detect")
        self._model(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), imgsz=IMGSZ, verbose=False)
        print("Model warm-up complete.")

    def start(self):
        if self._model is None:
            raise RuntimeError("Call load_model() before start().")
        for cam_idx in range(2):
            t = threading.Thread(target=self._camera_pipeline, args=(cam_idx,), daemon=True)
            t.start()

    # ── Active camera helpers ──────────────────────────────────────────────

    def get_active_cam_id(self):
        with self._active_cam_lock:
            return self._active_cam_id

    def switch_camera(self):
        with self._active_cam_lock:
            self._active_cam_id = 1 - self._active_cam_id
            return self._active_cam_id

    def get_active_frames(self):
        cid = self.get_active_cam_id()
        cs  = self.cam_states[cid]
        with cs.lock:
            return cs.output_main_frame, cs.output_mask_frame

    def get_telemetry(self):
        cid = self.get_active_cam_id()
        cs  = self.cam_states[cid]
        with cs.lock:
            tele     = dict(cs.telemetry)
            presence = dict(cs.presence_state)
        with self._stats_lock:
            stats = dict(self.sorter_stats)
        return {"active_cam": cid, "presence": presence, "telemetry": tele, "stats": stats}

    # ── Decision push ──────────────────────────────────────────────────────

    def _current_cycle(self):
        """Read the stepper's current cycle count (0 if unavailable)."""
        try:
            return self._stepper.get_stats().get("cycle_count", 0)
        except Exception:
            return 0

    def _flag_warning(self, msg):
        """Record a desync/detection anomaly (caller must hold _stats_lock)."""
        self.sorter_stats["desync_warnings"] += 1
        self.sorter_stats["last_warning"] = msg
        print(f"[SANITY] {msg} "
              f"(total warnings: {self.sorter_stats['desync_warnings']})")

    def _push_decision(self, cam_id, pepper_id, dom_color, yolo_class, good,
                       detected=True, color_dist=None, size_label="—"):
        """
        Counting flow:
          CAM 1 (intake):
            • BAD  → rejected += 1  (ejected, never reaches CAM 0)
            • GOOD → carried forward (colour distribution + SIZE); awaits CAM 0
          CAM 0 (final evaluation):
            • BAD  → rejected += 1
            • GOOD → accepted += 1 AND categorised into "<Colour> <Size>"
                     (e.g. "Red Large") using combined CAM1+CAM0 colour and the
                     size measured at CAM 1.

        `size_label` : "Large"/"Medium" measured at CAM 1 (ignored for CAM 0).
        """
        if not detected:
            decision = SortDecision(pepper_id=pepper_id, reject=False,
                                    color_stage="Unknown", maturity=yolo_class)
            self._result_queue.put(decision)
            print(f"[CAM {cam_id}] No pepper detected — no count change.")
            return

        def _pct_only(cd):
            if not cd:
                return None
            return {c: cd[c]["pct"] for c in ("Red", "Orange", "Yellow", "Green")}

        this_pct = _pct_only(color_dist)
        now_cycle = self._current_cycle()

        with self._stats_lock:
            if cam_id == 1:
                # ── INTAKE camera ──────────────────────────────────────────
                self.sorter_stats["cam1_last_color"] = dom_color
                if good:
                    # Passed intake → ENQUEUE for CAM 0 (FIFO), tag with cycle
                    with self._pending_color_lock:
                        self._transit_queue.append({
                            "color_dist":   this_pct,
                            "size":         size_label,
                            "dom_color":    dom_color,
                            "enqueue_cycle": now_cycle,
                        })
                        qlen  = len(self._transit_queue)
                        front = self._transit_queue[0]
                    self.sorter_stats["awaiting_cam0"]  = True
                    self.sorter_stats["transit_count"]  = qlen
                    self.sorter_stats["awaiting_color"] = front["dom_color"]
                    self.sorter_stats["awaiting_size"]  = front["size"]
                    print(f"[CAM 1] GOOD pepper passed intake "
                          f"({dom_color}, {size_label}) @ cycle {now_cycle} → "
                          f"enqueued (now {qlen} in transit)")

                    # Sanity: with 2-cycle travel, at most ~2 peppers should be
                    # in transit. More than that signals CAM 0 is missing peppers.
                    if qlen > CAM1_TO_CAM0_CYCLES + 1:
                        self._flag_warning(
                            f"Queue backlog = {qlen} (expected ≤ "
                            f"{CAM1_TO_CAM0_CYCLES + 1}); CAM 0 may be missing peppers")
                else:
                    # Bad at intake → rejected next cycle, never reaches CAM 0
                    self.sorter_stats["total_rejected"] += 1
                    # Queue the Ch 6 paddle to fire when this pepper reaches the
                    # station (CAM1_TO_PADDLE_CYCLES later, during that pause).
                    with self._pending_color_lock:
                        self._reject_paddle_queue.append({
                            "target_cycle": now_cycle + CAM1_TO_PADDLE_CYCLES,
                        })
                    print(f"[CAM 1] BAD pepper at intake → REJECTED, paddle queued "
                          f"for cycle {now_cycle + CAM1_TO_PADDLE_CYCLES} "
                          f"(total rejected: {self.sorter_stats['total_rejected']})")

            else:
                # ── OUTPUT camera (FINAL evaluation) ───────────────────────
                self.sorter_stats["cam0_last_color"] = dom_color

                # Pop the matching CAM 1 entry (front of FIFO queue)
                with self._pending_color_lock:
                    if self._transit_queue:
                        entry = self._transit_queue.pop(0)
                    else:
                        entry = None
                    qlen = len(self._transit_queue)

                if entry is None:
                    self._flag_warning(
                        "CAM 0 detected a pepper but transit queue was EMPTY — "
                        "missed CAM 1 detection or false CAM 0 detection; "
                        "categorising with CAM 0 data only")
                    cam1_pct  = None
                    cam1_size = "—"
                else:
                    cam1_pct  = entry["color_dist"]
                    cam1_size = entry["size"]

                    # Sanity: did it arrive about CAM1_TO_CAM0_CYCLES later?
                    travelled = now_cycle - entry.get("enqueue_cycle", now_cycle)
                    if abs(travelled - CAM1_TO_CAM0_CYCLES) > ARRIVAL_TOLERANCE:
                        self._flag_warning(
                            f"Arrival timing off: pepper travelled {travelled} "
                            f"cycles (expected ~{CAM1_TO_CAM0_CYCLES}); possible "
                            f"queue desync — match may be wrong")

                if good:
                    self.sorter_stats["total_accepted"] += 1

                    category, norm = _classify_category(cam1_pct, this_pct)
                    size  = cam1_size if cam1_size in ("Large", "Medium") else "Medium"
                    label = f"{category} {size}"

                    if label in self.sorter_stats["categories"]:
                        self.sorter_stats["categories"][label] += 1
                    self.sorter_stats["last_category"] = label
                    bin_label = label   # this pepper heads to the <label> bin

                    print(f"[CAM 0] GOOD → ACCEPTED/SORTED. Final = {label} "
                          f"(R:{norm['Red']:.0f}% O:{norm['Orange']:.0f}% "
                          f"Y:{norm['Yellow']:.0f}% G:{norm['Green']:.0f}%) "
                          f"| accepted: {self.sorter_stats['total_accepted']}")
                else:
                    # CAM 0 has the final say → reject next cycle
                    self.sorter_stats["total_rejected"] += 1
                    bin_label = "Reject"   # heads to the reject bin
                    print(f"[CAM 0] BAD at final eval (CAM 0 overrides) → REJECTED "
                          f"(total rejected: {self.sorter_stats['total_rejected']})")

                # ── Enqueue this result for its bin's trapdoor ─────────────
                # The bin sits BIN_CYCLE_OFFSET[bin] cycles after CAM 0; the
                # pepper is released when it reaches that cycle (FIFO preserved
                # because all peppers move together and offsets are fixed).
                offset = BIN_CYCLE_OFFSET.get(bin_label, 1)
                target_cycle = now_cycle + offset
                with self._pending_color_lock:
                    self._drop_queue.append({
                        "bin":          bin_label,
                        "target_cycle": target_cycle,
                    })
                    self.sorter_stats["drop_pending"] = len(self._drop_queue)
                print(f"[CAM 0] '{bin_label}' enqueued → drops at cycle "
                      f"{target_cycle} ({offset} cycles away)")

                # Update transit display from the new front of the queue
                with self._pending_color_lock:
                    front = self._transit_queue[0] if self._transit_queue else None
                self.sorter_stats["transit_count"] = qlen
                self.sorter_stats["awaiting_cam0"] = qlen > 0
                self.sorter_stats["awaiting_color"] = front["dom_color"] if front else "—"
                self.sorter_stats["awaiting_size"]  = front["size"]      if front else "—"

        decision = SortDecision(pepper_id=pepper_id, reject=not good,
                                color_stage=dom_color, maturity=yolo_class)
        self._result_queue.put(decision)
        print(f"[CAM {cam_id}] Decision pushed: {decision}")

        # Publish updated tally over MQTT (only sends if a count changed)
        self._publish_tally()

    def should_fire_reject_paddle(self):
        """
        Called by the StepperController once per pause. Returns True if a
        CAM 1-rejected pepper has reached the Ch 6 paddle station this cycle
        (i.e. an entry's target_cycle has arrived). Consumes the entry.
        """
        now_cycle = self._current_cycle()
        fire = False
        with self._pending_color_lock:
            still_waiting = []
            for entry in self._reject_paddle_queue:
                if now_cycle >= entry["target_cycle"]:
                    fire = True   # at least one paddle event is due
                else:
                    still_waiting.append(entry)
            self._reject_paddle_queue = still_waiting
        if fire:
            print(f"[PADDLE] Reject paddle due at cycle {now_cycle}")
        return fire

    def get_bins_to_fire(self):
        """
        Called by the StepperController once per pause. Scans the drop queue,
        tallies any peppers that have reached their bin's cycle, and returns a
        list of bin labels whose gate servos should fire this pause. Mirrors
        should_fire_reject_paddle(). Consumes the matched entries.
        """
        now_cycle = self._current_cycle()
        fired = []
        with self._stats_lock:
            with self._pending_color_lock:
                still_waiting = []
                for entry in self._drop_queue:
                    if now_cycle >= entry["target_cycle"]:
                        bin_name = entry["bin"]
                        if bin_name in self.sorter_stats["bins"]:
                            self.sorter_stats["bins"][bin_name] += 1
                        self.sorter_stats["last_drop"] = bin_name
                        fired.append(bin_name)
                        print(f"[DROP] '{bin_name}' bin reached (cycle {now_cycle})")
                    else:
                        still_waiting.append(entry)
                self._drop_queue = still_waiting
                self.sorter_stats["drop_pending"] = len(self._drop_queue)
        if fired:
            self._publish_tally()   # bin counts changed → publish
        return fired

    # ── Live detection thread (visual only, throttled) ─────────────────────

    def _live_detect_thread(self, cs: _CameraState):
        """
        Runs YOLO at LIVE_DETECT_FPS for REALTIME visual feedback:
          • bounding box with the model class (Good/Bad) + confidence
          • dominant colour label
          • a live-updating colour isolation mask
        This is visuals only — the averaged accept/reject decision and the
        sort statistics are still owned by the pause-window decision matrix.
        """
        interval = 1.0 / max(LIVE_DETECT_FPS, 1)
        while not self._stop_flag.is_set():
            t0 = time.time()

            # AI runs ONLY during the pause window. While the motor is moving,
            # the model is disabled and the live boxes/mask are cleared.
            if not self._stepper.inference_allowed.is_set():
                with cs.lock:
                    cs.live_boxes = []
                    blank = np.zeros((120, 200, 3), dtype=np.uint8)
                    cv2.putText(blank, "AI paused (moving)", (18, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
                    cs.output_mask_frame = blank
                time.sleep(0.1)
                continue

            with cs.lock:
                f = cs.frame_for_det.copy() if cs.frame_for_det is not None else None
            if f is None:
                time.sleep(interval)
                continue

            results = self._model(f, imgsz=IMGSZ, conf=CONF, verbose=False)
            merged  = _merge_boxes(results[0].boxes)

            fh, fw = f.shape[:2]
            boxes  = []
            best_mask = None   # realtime mask from the largest detection

            for bi, b in enumerate(merged):
                x1, y1, x2, y2 = map(int, b["xyxy"])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, fw), min(y2, fh)
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_name = self._model.names[b["cls"]]
                good     = _is_good(cls_name)

                # Colour analysis for dominant colour + mask
                crop      = f[y1:y2, x1:x2].copy()
                cd, gray  = _analyse_colors(crop)
                dom_color = max(cd.items(), key=lambda x: x[1]["pct"])[0] if cd else "Unknown"

                boxes.append((x1, y1, x2, y2, b["conf"], cls_name, good, dom_color))

                # Build the realtime mask from the largest (first) box —
                # clean white silhouette on black (matches reference look)
                if bi == 0:
                    white_mask = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    verdict = "GOOD" if good else "BAD"
                    cv2.putText(white_mask, f"{cls_name} ({verdict})", (4, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                    best_mask = cv2.resize(white_mask, (200, 120))

            if best_mask is None:
                best_mask = np.zeros((120, 200, 3), dtype=np.uint8)
                cv2.putText(best_mask, "No target", (40, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

            with cs.lock:
                cs.live_boxes        = boxes
                cs.output_mask_frame = best_mask   # realtime mask

            sleep_t = interval - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── Decision matrix thread (pause window only) ─────────────────────────

    def _inference_window(self, cs: _CameraState):
        """
        Waits for the stepper pause window, then detects the pepper
        CONTINUOUSLY for the whole pause. Every successful detection adds its
        colour reading (R/O/Y/G percentages) to a running total. When the
        window closes, all readings are AVERAGED into ONE final result — this
        counts as a single inference for the cycle.
        """
        cycle_num = 0
        while not self._stop_flag.is_set():

            if not self._stepper.inference_allowed.wait(timeout=0.5):
                continue

            cycle_num += 1
            print(f"[CAM {cs.cam_id}] Inference window opened (cycle ~{cycle_num})")

            # Running totals for averaging
            sum_pct   = {"Red": 0.0, "Orange": 0.0, "Yellow": 0.0, "Green": 0.0}
            class_votes = {}         # {class_name: count} from the YOLO model
            sum_conf    = 0.0        # sum of detection confidences
            sum_area    = 0          # sum of mask foreground pixel areas (CAM 1 size)
            num_hits  = 0            # frames where a pepper was actually detected
            num_polls = 0            # total detection attempts this window
            last_box  = None

            # ── Detect continuously until the window closes ───────────────
            while self._stepper.inference_allowed.is_set() and not self._stop_flag.is_set():
                num_polls += 1

                with cs.lock:
                    f = cs.frame_for_det.copy() if cs.frame_for_det is not None else None
                if f is None:
                    time.sleep(DETECT_INTERVAL)
                    continue

                # Use the RAW frame (same as the live detector) so the decision
                # matches what the model sees live. CLAHE enhancement was causing
                # the model to flip Good↔Bad on the same pepper.
                results  = self._model(f, imgsz=IMGSZ, conf=CONF, verbose=False)
                merged   = _merge_boxes(results[0].boxes)

                if not merged:
                    time.sleep(DETECT_INTERVAL)
                    continue

                box            = merged[0]
                fh, fw         = f.shape[:2]
                x1, y1, x2, y2 = map(int, box["xyxy"])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, fw), min(y2, fh)
                if x2 <= x1 or y2 <= y1:
                    time.sleep(DETECT_INTERVAL)
                    continue

                last_box = (x1, y1, x2, y2)
                crop     = f[y1:y2, x1:x2].copy()
                cd, mask = _analyse_colors(crop)

                # Accumulate colour reading (for dominant-colour display)
                for c in sum_pct:
                    sum_pct[c] += cd[c]["pct"]

                # Accumulate pepper area (foreground pixels in the mask) for size
                sum_area += int(cv2.countNonZero(mask))

                # Accumulate the model's CLASS vote (this drives accept/reject)
                cls_name = self._model.names[box["cls"]]
                class_votes[cls_name] = class_votes.get(cls_name, 0) + 1
                sum_conf += box["conf"]

                num_hits += 1
                time.sleep(DETECT_INTERVAL)

            # ── Window closed: resolve ONE result ─────────────────────────
            print(f"[CAM {cs.cam_id}] Window closed. "
                  f"Detected pepper in {num_hits}/{num_polls} polls. "
                  f"Class votes: {class_votes}")

            if num_hits > 0:
                # Averaged colour percentages → dominant colour (DISPLAY ONLY)
                avg_cd = {
                    c: {"pct": sum_pct[c] / num_hits, "count": 0}
                    for c in sum_pct
                }
                final_cd = avg_cd
                dom_color = max(avg_cd.items(), key=lambda x: x[1]["pct"])[0]

                # Majority class vote from the trained model → drives verdict
                final_class = max(class_votes.items(), key=lambda x: x[1])[0]
                avg_conf    = sum_conf / num_hits

                # Averaged pepper area → size label (measured at CAM 1)
                avg_area  = sum_area / num_hits
                size_label = "Large" if avg_area >= SIZE_LARGE_THRESHOLD else "Medium"
                if cs.cam_id == 1:
                    print(f"[CAM 1] measured area = {avg_area:.0f} px "
                          f"→ {size_label} (threshold {SIZE_LARGE_THRESHOLD})")
            else:
                final_cd    = {}
                dom_color   = "Unknown"
                final_class = "None"
                avg_conf    = 0.0
                avg_area    = 0
                size_label  = "—"

            good    = _is_good(final_class)
            verdict = "GOOD" if (num_hits > 0 and good) else ("BAD" if num_hits > 0 else "—")
            action  = "ACCEPT" if (num_hits > 0 and good) else ("REJECT" if num_hits > 0 else "—")

            # ── Build breakdown string from the averaged colour data ──────
            if final_cd:
                breakdown_str = (
                    f"R:{final_cd['Red']['pct']:.1f}%  "
                    f"O:{final_cd['Orange']['pct']:.1f}%  "
                    f"Y:{final_cd['Yellow']['pct']:.1f}%  "
                    f"G:{final_cd['Green']['pct']:.1f}%  "
                    f"(avg of {num_hits})"
                )
            else:
                breakdown_str = "R:0.0% O:0.0% Y:0.0% G:0.0%"

            print(f"[CAM {cs.cam_id}] Final: class={final_class} ({avg_conf:.1%}) "
                  f"dom_color={dom_color} → {verdict}")

            cam1_carried = "—"
            if cs.cam_id == 0:
                with self._pending_color_lock:
                    cam1_carried = self._pending_color_stage

            with cs.lock:
                cs.telemetry.update({
                    "infer":      dom_color,                       # dominant colour
                    "stage":      final_class,                     # model class label
                    "breakdown":  breakdown_str,
                    "verdict":    verdict,
                    "action":     action,
                    "votes":      f"{num_hits} detections | {class_votes}",
                    "cam1_color": cam1_carried,
                    "id":         f"CAM{cs.cam_id}_CYC_{cycle_num}",
                    "class":      f"{final_class.upper()} ({avg_conf:.1%})",
                })
                cs.presence_state["present"] = num_hits > 0
                cs.presence_state["count"]   = 1 if num_hits > 0 else 0

            # One averaged result = one inference / one decision.
            # Pass the dominant colour + verdict + colour distribution + size.
            self._push_decision(cs.cam_id, f"CAM{cs.cam_id}_CYC_{cycle_num}",
                                dom_color, final_class, good if num_hits > 0 else False,
                                detected=(num_hits > 0), color_dist=final_cd,
                                size_label=size_label)

            print(f"[CAM {cs.cam_id}] Inference window closed")


    # ── Camera pipeline thread (one per camera) ────────────────────────────

    def _camera_pipeline(self, cam_index: int):
        """
        Captures frames, keeps cs.frame_for_det fresh, draws LIVE boxes (thin,
        from the throttled live detector) plus the LOCKED decision box (bold,
        with colour bar) when a decision exists.
        """
        cs = self.cam_states[cam_index]

        try:
            picam2 = Picamera2(cam_index)
            config = picam2.create_preview_configuration(
                main={"size": CAPTURE_SIZE, "format": "RGB888"},
                controls={"FrameRate": TARGET_FPS},
                display=None, buffer_count=2,
            )
            picam2.configure(config)
            picam2.start()
            print(f"[CAM {cam_index}] Active.")
        except Exception as exc:
            print(f"[CAM {cam_index}] Hardware error: {exc}")
            return

        # Start the two detection threads for this camera
        threading.Thread(target=self._live_detect_thread, args=(cs,), daemon=True).start()
        threading.Thread(target=self._inference_window,   args=(cs,), daemon=True).start()

        sx = DISPLAY_SIZE[0] / CAPTURE_SIZE[0]
        sy = DISPLAY_SIZE[1] / CAPTURE_SIZE[1]

        while not self._stop_flag.is_set():
            loop_start = time.time()

            req       = picam2.capture_request()
            raw_array = req.make_array("main")
            req.release()

            frame_rgb = np.ascontiguousarray(raw_array)

            with cs.lock:
                cs.frame_for_det = frame_rgb.copy()

            display_frame = cv2.resize(frame_rgb.copy(), DISPLAY_SIZE)

            with cs.lock:
                live = list(cs.live_boxes)

            # ── Draw REALTIME boxes (class + colour, coloured by verdict) ──
            for (lx1, ly1, lx2, ly2, lconf, cls_name, good, dom_color) in live:
                dx1 = int(lx1 * sx); dy1 = int(ly1 * sy)
                dx2 = int(lx2 * sx); dy2 = int(ly2 * sy)
                if dx2 > dx1 and dy2 > dy1:
                    # Box colour: green if Good, red if Bad
                    box_color = (60, 200, 60) if good else (60, 60, 220)
                    verdict   = "GOOD" if good else "BAD"
                    _draw_box(display_frame,
                              f"{cls_name}", lconf,
                              dom_color, verdict,
                              dx1, dy1, dx2, dy2, box_color)

            with cs.lock:
                cs.output_main_frame = display_frame

            sleep_t = (1 / TARGET_FPS) - (time.time() - loop_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

        picam2.stop()
