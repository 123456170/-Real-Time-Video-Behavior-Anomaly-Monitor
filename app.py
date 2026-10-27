import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
import math
from collections import deque
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Real-Time Video Behavior Anomaly Monitor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

W, H = 640, 360
FPS = 10

DEMO_SEC = 24
BASELINE_SEC = 6
RING_SEC = 3
LIVE_REC_SEC = 12

BASELINE_FRAMES = int(BASELINE_SEC * FPS)

FEATURE_COLS = [
    "speed",
    "accel",
    "dir_change",
    "dwell",
    "area",
    "density",
    "frame_motion",
    "spread",
    "zone_dwell",
]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def encode_jpeg(frame, quality=82):
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return enc.tobytes() if ok else b""


def evidence_to_str(evidence):
    if not evidence:
        return ""
    return "; ".join([f"{k} z={float(v):.2f}" for k, v in evidence])


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def angle_between(vx1, vy1, vx2, vy2):
    n1 = math.hypot(vx1, vy1)
    n2 = math.hypot(vx2, vy2)
    if n1 < 1e-4 or n2 < 1e-4:
        return 0.0
    c = (vx1 * vx2 + vy1 * vy2) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c))


def create_video_writer():
    attempts = [
        (".mp4", ["avc1", "H264", "x264", "mp4v"]),
        (".avi", ["XVID", "MJPG"]),
    ]

    for suffix, codecs in attempts:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        for cc in codecs:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
            if writer.isOpened():
                fmt = "video/mp4" if suffix == ".mp4" else "video/x-msvideo"
                return writer, path, fmt

        try:
            os.unlink(path)
        except OSError:
            pass

    return None, None, None


def finalize_video(writer, path, fmt):
    if writer is not None:
        writer.release()

    if path and os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        try:
            os.unlink(path)
        except OSError:
            pass
        return data, fmt

    return None, None


# ----------------------------------------------------------------------------
# Synthetic video simulator
# ----------------------------------------------------------------------------

class Simulator:
    def __init__(self, seed=7, mode="demo"):
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.frame = 0
        self.t = 0.0
        self.agents = []
        self.next_id = 1
        self.events = []
        self.background = self._make_background()

        if self.mode == "demo":
            self._schedule_demo()

    def _make_background(self):
        bg = np.zeros((H, W, 3), dtype=np.uint8)

        for y in range(H):
            v = int(28 + 26 * (y / max(1, H - 1)))
            bg[y, :, 0] = v
            bg[y, :, 1] = min(255, v + 4)
            bg[y, :, 2] = min(255, v + 8)

        cv2.rectangle(bg, (0, 110), (W, 165), (58, 58, 58), -1)
        cv2.rectangle(bg, (0, 210), (W, 265), (58, 58, 58), -1)

        for x in range(0, W, 80):
            cv2.line(bg, (x, 185), (x + 38, 185), (95, 95, 95), 1)

        cv2.putText(
            bg,
            "ZONE A",
            (14, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            bg,
            "CENTRAL WALKWAY",
            (14, 198),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            bg,
            "ZONE B",
            (14, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )

        return bg

    def _schedule_demo(self):
        raw = [
            {
                "id": 1,
                "type": "sprint",
                "start_sec": 8.0,
                "end_sec": 10.5,
                "harmful": True,
                "description": "Sudden sprint across the corridor",
            },
            {
                "id": 2,
                "type": "loiter",
                "start_sec": 12.0,
                "end_sec": 16.0,
                "harmful": True,
                "description": "Extended loitering in central walkway",
            },
            {
                "id": 3,
                "type": "harmless",
                "start_sec": 14.0,
                "end_sec": 17.0,
                "harmful": False,
                "description": "Service robot zigzag crossing: unusual but harmless",
            },
            {
                "id": 4,
                "type": "reverse",
                "start_sec": 18.0,
                "end_sec": 20.5,
                "harmful": True,
                "description": "Movement against expected flow",
            },
            {
                "id": 5,
                "type": "abandoned",
                "start_sec": 21.0,
                "end_sec": 24.0,
                "harmful": True,
                "description": "Unattended object appears in walkway",
            },
        ]

        for ev in raw:
            ev["start_frame"] = int(ev["start_sec"] * FPS)
            ev["end_frame"] = int(ev["end_sec"] * FPS)
            ev["agent_id"] = None
            ev["applied"] = False

        self.events = raw

    def _new_agent(
        self,
        kind,
        x,
        y,
        vx,
        vy,
        radius,
        behavior,
        stand_until=None,
        event_id=None,
        base_y=None,
    ):
        agent = {
            "id": self.next_id,
            "kind": kind,
            "x": float(x),
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
            "radius": int(radius),
            "behavior": behavior,
            "stand_until": stand_until,
            "event_id": event_id,
            "anomaly_kind": None,
            "speed_mult": 1.0,
            "loiter_center": None,
            "base_y": float(base_y if base_y is not None else y),
            "phase": float(self.rng.uniform(0, 2 * math.pi)),
            "hist": deque(maxlen=6),
            "dead": False,
        }
        self.next_id += 1
        self.agents.append(agent)
        return agent

    def _spawn_normal(self):
        alive_persons = [a for a in self.agents if a["kind"] == "person" and not a["dead"]]

        if len(alive_persons) < 5 and self.frame % 12 == 0:
            side_left = self.rng.random() < 0.5
            x = -25.0 if side_left else float(W + 25)
            y = float(self.rng.integers(95, H - 95))
            speed = float(self.rng.uniform(32, 58))
            vx = speed if side_left else -speed
            vy = float(self.rng.normal(0, 4))

            stand = self.rng.random() < 0.12
            stand_until = self.t + float(self.rng.uniform(1.5, 4.0)) if stand else None

            self._new_agent(
                kind="person",
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                radius=int(self.rng.integers(10, 15)),
                behavior="stand" if stand else "walk",
                stand_until=stand_until,
            )

    def _spawn_person_inside(self):
        return self._new_agent(
            kind="person",
            x=float(W * 0.35),
            y=float(H * 0.52),
            vx=45.0,
            vy=float(self.rng.normal(0, 3)),
            radius=12,
            behavior="walk",
        )

    def _event_by_id(self, event_id):
        for ev in self.events:
            if ev["id"] == event_id:
                return ev
        return None

    def _start_event(self, ev):
        kind = ev["type"]

        if kind in ("sprint", "loiter", "reverse"):
            candidates = [
                a
                for a in self.agents
                if a["kind"] == "person" and not a["dead"] and a["event_id"] is None
            ]

            if candidates:
                idx = int(self.rng.integers(0, len(candidates)))
                a = candidates[idx]
            else:
                a = self._spawn_person_inside()

            ev["agent_id"] = a["id"]
            a["event_id"] = ev["id"]
            a["anomaly_kind"] = kind

            if kind == "sprint":
                a["behavior"] = "walk"
                a["speed_mult"] = 3.2
            elif kind == "loiter":
                a["behavior"] = "loiter"
                a["loiter_center"] = (float(W * 0.52), float(H * 0.52))
                a["speed_mult"] = 1.0
            elif kind == "reverse":
                a["vx"] = -a["vx"]
                a["vy"] = float(self.rng.normal(0, 3))
                a["speed_mult"] = 1.25

        elif kind == "abandoned":
            a = self._new_agent(
                kind="object",
                x=float(W * 0.58),
                y=float(H * 0.56),
                vx=0.0,
                vy=0.0,
                radius=14,
                behavior="static",
                event_id=ev["id"],
            )
            ev["agent_id"] = a["id"]

        elif kind == "harmless":
            a = self._new_agent(
                kind="harmless",
                x=-20.0,
                y=float(H * 0.45),
                vx=68.0,
                vy=0.0,
                radius=7,
                behavior="zigzag",
                event_id=ev["id"],
                base_y=float(H * 0.45),
            )
            ev["agent_id"] = a["id"]

    def _apply_scheduled_events(self):
        for ev in self.events:
            if not ev.get("applied", False) and self.frame >= ev["start_frame"]:
                self._start_event(ev)
                ev["applied"] = True

    def _update_agents(self, dt):
        for a in self.agents:
            if a["dead"]:
                continue

            if a["event_id"] is not None:
                ev = self._event_by_id(a["event_id"])
                if ev and self.frame > ev["end_frame"]:
                    a["event_id"] = None
                    a["anomaly_kind"] = None
                    a["speed_mult"] = 1.0

                    if a["kind"] in ("object", "harmless"):
                        a["dead"] = True
                    else:
                        a["behavior"] = "walk"
                        a["vx"] = 45.0 if a["x"] < W * 0.5 else -45.0
                        a["vy"] = float(self.rng.normal(0, 3))

            if a["kind"] == "object":
                pass

            elif a["behavior"] == "stand":
                if a["stand_until"] is not None and self.t >= a["stand_until"]:
                    a["behavior"] = "walk"
                    a["vx"] = 42.0 if a["x"] < W * 0.5 else -42.0
                    a["vy"] = float(self.rng.normal(0, 3))

                a["x"] += float(self.rng.normal(0, 0.25))
                a["y"] += float(self.rng.normal(0, 0.25))

            elif a["behavior"] == "loiter":
                cx, cy = a.get("loiter_center") or (W * 0.52, H * 0.52)
                ang = self.t * 1.2 + a["phase"]
                tx = cx + 28 * math.cos(ang)
                ty = cy + 18 * math.sin(ang)

                a["vx"] = (tx - a["x"]) * 2.5
                a["vy"] = (ty - a["y"]) * 2.5

                a["x"] += a["vx"] * dt
                a["y"] += a["vy"] * dt

            elif a["behavior"] == "zigzag":
                a["x"] += a["vx"] * a.get("speed_mult", 1.0) * dt
                a["y"] = a["base_y"] + 38 * math.sin(self.t * 4.5 + a["phase"])

            else:
                mult = a.get("speed_mult", 1.0)
                a["x"] += a["vx"] * mult * dt
                a["y"] += a["vy"] * mult * dt + 1.5 * math.sin(self.t * 2 + a["phase"]) * dt

            a["hist"].append((a["x"], a["y"], self.frame))

            if a["kind"] == "person" and (
                a["x"] < -70 or a["x"] > W + 70 or a["y"] < -70 or a["y"] > H + 70
            ):
                a["dead"] = True

        self.agents = [a for a in self.agents if not a["dead"]]

    def _render(self):
        frame = self.background.copy()

        for a in self.agents:
            if a["dead"]:
                continue

            x = int(np.clip(a["x"], 0, W - 1))
            y = int(np.clip(a["y"], 0, H - 1))

            if a["kind"] == "object":
                color = (60, 80, 180)
                axes = (16, 12)
            elif a["kind"] == "harmless":
                color = (255, 220, 90)
                axes = (7, 7)
            else:
                anomaly = a.get("anomaly_kind")
                if anomaly == "sprint":
                    color = (80, 80, 220)
                elif anomaly == "loiter":
                    color = (200, 100, 180)
                elif anomaly == "reverse":
                    color = (90, 180, 180)
                else:
                    color = (180, 140, 90)

                r = int(a["radius"])
                axes = (r, int(r * 1.35))

            cv2.ellipse(frame, (x, y), axes, 0, 0, 360, color, -1)
            cv2.ellipse(frame, (x, y), axes, 0, 0, 360, (20, 20, 20), 1)

        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame

    def step(self):
        dt = 1.0 / FPS
        self.frame += 1
        self.t += dt

        self._spawn_normal()
        self._apply_scheduled_events()
        self._update_agents(dt)

        frame = self._render()
        return frame, self.agents

    def is_in_event(self, frame_idx=None):
        f = self.frame if frame_idx is None else frame_idx
        return any(ev["start_frame"] <= f <= ev["end_frame"] for ev in self.events)

    def inject(self, kind):
        durations = {
            "sprint": 2.5,
            "loiter": 4.0,
            "reverse": 2.5,
            "abandoned": 4.0,
            "harmless": 3.5,
        }

        descriptions = {
            "sprint": "Injected sprint event",
            "loiter": "Injected loitering event",
            "reverse": "Injected reverse-flow event",
            "abandoned": "Injected abandoned-object event",
            "harmless": "Injected unusual-but-harmless robot event",
        }

        duration = durations.get(kind, 3.0)
        harmful = kind != "harmless"

        ev = {
            "id": len(self.events) + 1,
            "type": kind,
            "start_sec": self.t,
            "end_sec": self.t + duration,
            "start_frame": self.frame,
            "end_frame": self.frame + int(duration * FPS),
            "harmful": harmful,
            "description": descriptions.get(kind, "Injected event"),
            "agent_id": None,
            "applied": True,
        }

        self.events.append(ev)
        self._start_event(ev)


# ----------------------------------------------------------------------------
# Lightweight tracker
# ----------------------------------------------------------------------------

class Track:
    def __init__(self, tid, det, frame_idx):
        self.id = tid
        self.hist = deque(maxlen=8)
        self.hist.append((det["x"], det["y"], frame_idx))
        self.area = det["area"]
        self.bbox = det["bbox"]
        self.first = frame_idx
        self.last = frame_idx
        self.lost = 0

        self.vx = 0.0
        self.vy = 0.0
        self.prev_vx = 0.0
        self.prev_vy = 0.0

    def update(self, det, frame_idx, dt):
        self.prev_vx, self.prev_vy = self.vx, self.vy

        px, py, _ = self.hist[-1]
        self.vx = (det["x"] - px) / max(dt, 1e-6)
        self.vy = (det["y"] - py) / max(dt, 1e-6)

        self.hist.append((det["x"], det["y"], frame_idx))
        self.area = det["area"]
        self.bbox = det["bbox"]
        self.last = frame_idx
        self.lost = 0

    def predict(self, dt=0.1):
        x, y, _ = self.hist[-1]
        return x + self.vx * dt, y + self.vy * dt


class Tracker:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=24,
            detectShadows=False,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.tracks = []
        self.next_id = 1

    def process(self, frame, frame_idx):
        dt = 1.0 / FPS

        mask = self.bg.apply(frame)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        frame_motion = float(np.mean(mask) / 255.0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 100 or area > 20000:
                continue

            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue

            x = float(m["m10"] / m["m00"])
            y = float(m["m01"] / m["m00"])
            bbox = cv2.boundingRect(c)

            dets.append({"x": x, "y": y, "area": area, "bbox": bbox})

        candidates = [t for t in self.tracks if t.lost < 30]

        pairs = []
        for ti, t in enumerate(candidates):
            px, py = t.predict(dt)
            for di, d in enumerate(dets):
                dist = math.hypot(px - d["x"], py - d["y"])
                if dist > 75:
                    continue

                min_area = max(1.0, min(t.area, d["area"]))
                max_area = max(t.area, d["area"])
                if max_area / min_area > 4.0:
                    continue

                pairs.append((dist, ti, di))

        pairs.sort(key=lambda z: z[0])

        matched_t = set()
        matched_d = set()

        for _, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue

            candidates[ti].update(dets[di], frame_idx, dt)
            matched_t.add(ti)
            matched_d.add(di)

        for ti, t in enumerate(candidates):
            if ti not in matched_t:
                t.lost += 1

        for di, d in enumerate(dets):
            if di not in matched_d:
                t = Track(self.next_id, d, frame_idx)
                self.next_id += 1
                self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.lost < 30]
        active = [t for t in self.tracks if t.lost == 0]

        features = []

        for t in active:
            speed = math.hypot(t.vx, t.vy)
            accel = math.hypot(t.vx - t.prev_vx, t.vy - t.prev_vy) / max(dt, 1e-6)
            dir_change = angle_between(t.prev_vx, t.prev_vy, t.vx, t.vy)
            dwell = float(frame_idx - t.first) / FPS

            positions = np.array([(x, y) for x, y, _ in t.hist], dtype=float)
            if len(positions) > 1:
                mean = positions.mean(axis=0)
                spread = float(np.max(np.linalg.norm(positions - mean, axis=1)))
            else:
                spread = 0.0

            cx, cy, _ = t.hist[-1]
            zone_dwell = dwell if 0.45 * H < cy < 0.62 * H else 0.0

            density = 0
            for o in active:
                if o.id == t.id:
                    continue
                ox, oy, _ = o.hist[-1]
                if math.hypot(ox - cx, oy - cy) < 90:
                    density += 1

            feat = {
                "speed": float(speed),
                "accel": float(accel),
                "dir_change": float(dir_change),
                "dwell": float(dwell),
                "area": float(t.area),
                "density": float(density),
                "frame_motion": float(frame_motion),
                "spread": float(spread),
                "zone_dwell": float(zone_dwell),
            }

            features.append(
                {
                    "track_id": int(t.id),
                    "features": feat,
                    "bbox": t.bbox,
                }
            )

        return features


# ----------------------------------------------------------------------------
# Anomaly detector with evidence
# ----------------------------------------------------------------------------

class AnomalyDetector:
    def __init__(self, threshold=0.65, contamination=0.05):
        self.threshold = float(threshold)
        self.contamination = float(contamination)

        self.model = None
        self.scaler = None
        self.train_X = []

        self.trained = False
        self.score_lo = 0.0
        self.score_hi = 1.0

        self.median = None
        self.iqr = None

        self.benign_prototypes = []
        self.harmful_prototypes = []
        self.harmless_prototypes = []

        self.version = 1

    def _as_vector(self, feat):
        return np.array([float(feat.get(c, 0.0)) for c in FEATURE_COLS], dtype=float)

    def _scale(self, feat):
        x = self._as_vector(feat)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is not None:
            return self.scaler.transform(x.reshape(1, -1))[0]
        return x

    def add_training(self, feat):
        x = self._as_vector(feat)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.train_X.append(x.tolist())

    def train(self):
        if len(self.train_X) < 15:
            return

        X = np.array(self.train_X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.model = IsolationForest(
            n_estimators=120,
            contamination=self.contamination,
            random_state=0,
            n_jobs=1,
        ).fit(Xs)

        raw = -self.model.score_samples(Xs)
        self.score_lo = float(np.percentile(raw, 5))
        self.score_hi = float(np.percentile(raw, 95))

        if self.score_hi <= self.score_lo + 1e-6:
            self.score_hi = self.score_lo + 1e-6

        self.median = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        self.iqr = np.maximum(q75 - q25, 1e-3)

        self.trained = True
        self.version += 1

    def predict_label(self, xs):
        if not self.harmful_prototypes and not self.harmless_prototypes:
            return "unknown", 0.0

        best_label = "unknown"
        best_conf = 0.0

        for label, protos in (
            ("harmful", self.harmful_prototypes),
            ("unusual_harmless", self.harmless_prototypes),
        ):
            for p in protos[-50:]:
                sim = cosine_similarity(xs, p)
                if sim > best_conf:
                    best_conf = sim
                    best_label = label

        if best_conf < 0.78:
            return "unknown", float(best_conf)

        return best_label, float(best_conf)

    def score(self, feat):
        if not self.trained or self.model is None:
            return 0.0, [], "unknown", 0.0

        x = self._as_vector(feat)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        xs = self.scaler.transform(x.reshape(1, -1))[0]
        raw = float(-self.model.score_samples(xs.reshape(1, -1))[0])

        norm = float(np.clip((raw - self.score_lo) / (self.score_hi - self.score_lo), 0.0, 1.0))

        z = np.abs((x - self.median) / (self.iqr * 1.349 + 1e-6))
        z = np.clip(z, 0.0, 6.0)

        z_mean = float(np.mean(z))
        score = float(np.clip(0.75 * norm + 0.25 * min(1.0, z_mean / 3.0), 0.0, 1.0))

        if self.benign_prototypes:
            max_sim = 0.0
            for p in self.benign_prototypes[-50:]:
                sim = cosine_similarity(xs, p)
                if sim > max_sim:
                    max_sim = sim
            score *= max(0.35, 1.0 - 0.65 * max_sim)

        evidence = sorted(
            zip(FEATURE_COLS, z.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]

        label, conf = self.predict_label(xs)

        return float(score), evidence, label, float(conf)

    def add_benign(self, feat):
        self.benign_prototypes.append(self._scale(feat))

    def add_prototype(self, feat, label):
        xs = self._scale(feat)
        if label == "harmful":
            self.harmful_prototypes.append(xs)
        elif label == "unusual_harmless":
            self.harmless_prototypes.append(xs)

    def clear_learned(self):
        self.benign_prototypes = []
        self.harmful_prototypes = []
        self.harmless_prototypes = []
        self.version += 1


def clone_trained_detector(src, threshold, contamination):
    d = AnomalyDetector(threshold=threshold, contamination=contamination)
    d.trained = src.trained
    d.model = src.model
    d.scaler = src.scaler
    d.train_X = list(src.train_X)
    d.score_lo = src.score_lo
    d.score_hi = src.score_hi
    d.median = src.median
    d.iqr = src.iqr
    d.version = src.version
    return d


# ----------------------------------------------------------------------------
# Alert manager
# ----------------------------------------------------------------------------

class AlertManager:
    def __init__(self, threshold=0.65):
        self.threshold = float(threshold)
        self.streak = {}
        self.last_time = {}
        self.alerts = []
        self.next_id = 1

    def process(
        self,
        track_id,
        score,
        frame_idx,
        time_s,
        evidence,
        features,
        label,
        conf,
        ring,
        current_jpeg,
    ):
        if score >= self.threshold:
            self.streak[track_id] = self.streak.get(track_id, 0) + 1
            last = self.last_time.get(track_id, -10.0)

            if self.streak[track_id] >= 2 and (time_s - last) > 1.0:
                clip_items = list(ring)[-18:]
                clip = [item[2] for item in clip_items if item[2]]

                alert = {
                    "id": self.next_id,
                    "frame": int(frame_idx),
                    "time": float(time_s),
                    "track_id": int(track_id),
                    "score": float(score),
                    "evidence": evidence,
                    "features": dict(features),
                    "features_vec": np.array(
                        [features.get(c, 0.0) for c in FEATURE_COLS],
                        dtype=float,
                    ),
                    "predicted_label": label,
                    "predicted_conf": float(conf),
                    "review_label": None,
                    "clip": clip,
                }

                self.alerts.append(alert)
                self.next_id += 1
                self.last_time[track_id] = time_s

                if len(self.alerts) > 300:
                    self.alerts.pop(0)

                return alert
        else:
            self.streak[track_id] = 0

        return None


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def evaluate_alerts(alerts, events, duration_sec, tol_sec=1.0):
    duration_sec = max(1e-6, float(duration_sec))
    hours = max(1e-9, duration_sec / 3600.0)

    detected_ids = set()
    delays = []
    rows = []

    for ev in events:
        start = float(ev["start_sec"])
        end = float(ev["end_sec"])

        matches = [
            a
            for a in alerts
            if start - tol_sec <= float(a["time"]) <= end + tol_sec
        ]

        detected = len(matches) > 0
        delay = None

        if detected:
            first = min(matches, key=lambda a: a["time"])
            delay = max(0.0, float(first["time"]) - start)
            delays.append(delay)
            detected_ids.add(ev["id"])

        rows.append(
            {
                "Event ID": ev["id"],
                "Type": ev["type"],
                "Start (s)": round(start, 2),
                "End (s)": round(end, 2),
                "Class": "Harmful / review" if ev["harmful"] else "Unusual harmless",
                "Detected": "Yes" if detected else "No",
                "Delay (s)": round(delay, 2) if delay is not None else "",
            }
        )

    false_alerts = []
    for a in alerts:
        at = float(a["time"])
        belongs = any(
            float(ev["start_sec"]) - tol_sec <= at <= float(ev["end_sec"]) + tol_sec
            for ev in events
        )
        if not belongs:
            false_alerts.append(a)

    harmful_events = [e for e in events if e["harmful"]]
    harmless_events = [e for e in events if not e["harmful"]]

    det_harmful = len([e for e in harmful_events if e["id"] in detected_ids])
    det_harmless = len([e for e in harmless_events if e["id"] in detected_ids])

    metrics = {
        "duration_sec": duration_sec,
        "total_events": len(events),
        "detected_events": len(detected_ids),
        "recall_all": len(detected_ids) / max(1, len(events)),
        "recall_harmful": det_harmful / max(1, len(harmful_events)),
        "recall_harmless": det_harmless / max(1, len(harmless_events)),
        "total_alerts": len(alerts),
        "false_alarms": len(false_alerts),
        "false_alarms_per_hour": len(false_alerts) / hours,
        "mean_delay_sec": float(np.mean(delays)) if delays else None,
        "median_delay_sec": float(np.median(delays)) if delays else None,
    }

    columns = [
        "Event ID",
        "Type",
        "Start (s)",
        "End (s)",
        "Class",
        "Detected",
        "Delay (s)",
    ]

    return metrics, pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------------
# Demo generation
# ----------------------------------------------------------------------------

def generate_demo(threshold, contamination):
    sim = Simulator(seed=7, mode="demo")
    tracker = Tracker()
    detector = AnomalyDetector(threshold=threshold, contamination=contamination)
    alert_mgr = AlertManager(threshold=threshold)

    ring = deque(maxlen=int(RING_SEC * FPS))

    writer, path, fmt = create_video_writer()

    total_frames = int(DEMO_SEC * FPS)
    score_rows = []

    for _ in range(total_frames):
        frame, _agents = sim.step()

        if writer is not None:
            writer.write(frame)

        jpeg = encode_jpeg(frame)
        ring.append((sim.frame, sim.t, jpeg))

        track_feats = tracker.process(frame, sim.frame)
        in_event = sim.is_in_event(sim.frame)

        if not detector.trained and not in_event:
            for tf in track_feats:
                detector.add_training(tf["features"])

            if sim.frame >= BASELINE_FRAMES and len(detector.train_X) >= 15:
                detector.train()

        max_score = 0.0
        best_track = None

        for tf in track_feats:
            score, evidence, label, conf = detector.score(tf["features"])

            if score > max_score:
                max_score = score
                best_track = tf["track_id"]

            alert_mgr.process(
                track_id=tf["track_id"],
                score=score,
                frame_idx=sim.frame,
                time_s=sim.t,
                evidence=evidence,
                features=tf["features"],
                label=label,
                conf=conf,
                ring=ring,
                current_jpeg=jpeg,
            )

        score_rows.append(
            {
                "time": float(sim.t),
                "score": float(max_score),
                "event": 1 if in_event else 0,
                "track": best_track,
            }
        )

    video_bytes, video_fmt = finalize_video(writer, path, fmt)

    metrics, events_df = evaluate_alerts(
        alerts=alert_mgr.alerts,
        events=sim.events,
        duration_sec=DEMO_SEC,
    )

    score_df = pd.DataFrame(score_rows)

    return {
        "video_bytes": video_bytes,
        "video_format": video_fmt or "video/mp4",
        "metrics": metrics,
        "events_df": events_df,
        "score_df": score_df,
        "alerts": alert_mgr.alerts,
        "events": sim.events,
        "detector": detector,
    }


# ----------------------------------------------------------------------------
# Live state helpers
# ----------------------------------------------------------------------------

def start_live_recording(live, seconds):
    if live.get("writer") is not None:
        data, fmt = finalize_video(
            live["writer"],
            live["writer_path"],
            live["writer_fmt"],
        )
        live["clip_bytes"] = data
        live["clip_format"] = fmt
        live["writer"] = None
        live["writer_path"] = None
        live["writer_fmt"] = None
        live["clip_just_ready"] = True

    writer, path, fmt = create_video_writer()
    live["writer"] = writer
    live["writer_path"] = path
    live["writer_fmt"] = fmt
    live["record_remaining"] = int(seconds * FPS) if writer is not None else 0


def ensure_live():
    params = st.session_state.params

    sim = Simulator(seed=11, mode="live")
    tracker = Tracker()
    detector = clone_trained_detector(
        st.session_state.demo["detector"],
        threshold=params["threshold"],
        contamination=params["contamination"],
    )
    alert_mgr = AlertManager(threshold=params["threshold"])

    live = {
        "sim": sim,
        "tracker": tracker,
        "detector": detector,
        "alert_mgr": alert_mgr,
        "ring": deque(maxlen=int(RING_SEC * FPS)),
        "frame_idx": 0,
        "time": 0.0,
        "running": True,
        "history": deque(maxlen=180),
        "current_frame": None,
        "latest_alert": None,
        "writer": None,
        "writer_path": None,
        "writer_fmt": None,
        "record_remaining": 0,
        "clip_bytes": None,
        "clip_format": None,
        "clip_just_ready": False,
        "metrics": None,
        "events_df": None,
    }

    st.session_state.live = live

    start_live_recording(live, LIVE_REC_SEC)
    advance_live(1)


def advance_live(steps=1):
    live = st.session_state.get("live")
    if live is None:
        return

    sim = live["sim"]
    tracker = live["tracker"]
    detector = live["detector"]
    alert_mgr = live["alert_mgr"]
    ring = live["ring"]

    for _ in range(steps):
        frame, _agents = sim.step()

        live["current_frame"] = frame
        live["frame_idx"] = sim.frame
        live["time"] = sim.t

        if live.get("writer") is not None and live.get("record_remaining", 0) > 0:
            live["writer"].write(frame)
            live["record_remaining"] -= 1

            if live["record_remaining"] <= 0:
                data, fmt = finalize_video(
                    live["writer"],
                    live["writer_path"],
                    live["writer_fmt"],
                )
                live["clip_bytes"] = data
                live["clip_format"] = fmt
                live["writer"] = None
                live["writer_path"] = None
                live["writer_fmt"] = None
                live["clip_just_ready"] = True

        jpeg = encode_jpeg(frame)
        ring.append((sim.frame, sim.t, jpeg))

        track_feats = tracker.process(frame, sim.frame)
        in_event = sim.is_in_event(sim.frame)

        max_score = 0.0
        best_track = None

        for tf in track_feats:
            score, evidence, label, conf = detector.score(tf["features"])

            if score > max_score:
                max_score = score
                best_track = tf["track_id"]

            alert = alert_mgr.process(
                track_id=tf["track_id"],
                score=score,
                frame_idx=sim.frame,
                time_s=sim.t,
                evidence=evidence,
                features=tf["features"],
                label=label,
                conf=conf,
                ring=ring,
                current_jpeg=jpeg,
            )

            if alert is not None:
                live["latest_alert"] = alert

        live["history"].append(
            {
                "time": float(sim.t),
                "score": float(max_score),
                "event": 1 if in_event else 0,
                "track": best_track,
            }
        )


def apply_feedback(alert, label, detector, source):
    alert["review_label"] = label

    st.session_state.setdefault("feedback_log", []).append(
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "alert_id": alert["id"],
            "track": alert["track_id"],
            "score": round(float(alert["score"]), 3),
            "label": label,
        }
    )

    if not st.session_state.params.get("learn", True):
        return

    if detector is None:
        return

    if label == "false_alarm":
        detector.add_benign(alert["features"])
    elif label == "harmful":
        detector.add_prototype(alert["features"], "harmful")
    elif label == "unusual_harmless":
        detector.add_prototype(alert["features"], "unusual_harmless")


def compute_live_metrics():
    live = st.session_state.live
    duration = max(1e-6, float(live["time"]))

    metrics, events_df = evaluate_alerts(
        alerts=live["alert_mgr"].alerts,
        events=live["sim"].events,
        duration_sec=duration,
    )

    live["metrics"] = metrics
    live["events_df"] = events_df


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def plot_score_timeline(score_df, events, threshold):
    if score_df is None or score_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 3.2))

    ax.plot(score_df["time"], score_df["score"], color="#2b7de9", linewidth=1.8)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.0)

    for ev in events:
        color = "#d9534f" if ev["harmful"] else "#f0ad4e"
        ax.axvspan(ev["start_sec"], ev["end_sec"], alpha=0.18, color=color)

    ax.set_xlim(0, max(score_df["time"].max(), 1.0))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Unusual score")
    ax.set_title("Frame-level unusual-motion score with ground-truth event windows")

    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Live monitor fragment
# ----------------------------------------------------------------------------

def live_monitor_impl():
    live = st.session_state.get("live")
    if live is None:
        st.warning("Live monitor is not initialized.")
        return

    if live.get("running", False):
        advance_live(2)

    if live.get("clip_just_ready", False):
        live["clip_just_ready"] = False
        st.rerun()

    st.subheader("Live synthetic monitor")

    col_video, col_stats = st.columns([2.2, 1.0])

    with col_video:
        frame = live.get("current_frame")
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("Starting live simulation...")

    with col_stats:
        st.metric("Status", "Running" if live.get("running") else "Paused")
        st.metric("Frame", live.get("frame_idx", 0))
        st.metric("Sim time", f"{live.get('time', 0.0):.1f}s")

        score = live["history"][-1]["score"] if live["history"] else 0.0
        st.metric("Current unusual score", f"{score:.2f}")
        st.progress(float(np.clip(score, 0.0, 1.0)))

        rec = live.get("record_remaining", 0)
        if rec > 0:
            st.caption(f"Recording live clip: {rec} frames remaining")
        elif live.get("clip_bytes"):
            st.caption("Live clip ready in the section below.")
        else:
            st.caption("No live clip is currently recording.")

    alert = live.get("latest_alert")
    if alert is not None:
        st.markdown("**Latest live alert evidence**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alert", f"#{alert['id']}")
        c2.metric("Track", alert["track_id"])
        c3.metric("Predicted", alert["predicted_label"])
        c4.metric("Reviewer", alert["review_label"] or "pending")

        for feat, z in alert["evidence"]:
            st.progress(float(min(1.0, float(z) / 5.0)))
            st.caption(f"{feat}: robust z={float(z):.2f}")

        fb = st.columns(3)

        if fb[0].button("Harmful", key=f"live_fb_harmful_{alert['id']}"):
            apply_feedback(alert, "harmful", live["detector"], "live")

        if fb[1].button("Unusual harmless", key=f"live_fb_harmless_{alert['id']}"):
            apply_feedback(alert, "unusual_harmless", live["detector"], "live")

        if fb[2].button("False alarm", key=f"live_fb_false_{alert['id']}"):
            apply_feedback(alert, "false_alarm", live["detector"], "live")

    if live["history"]:
        hist_df = pd.DataFrame(list(live["history"]))
        st.line_chart(hist_df.set_index("time")[["score"]])

    ctrl = st.columns(5)

    if ctrl[0].button(
        "Pause" if live.get("running") else "Start",
        key="live_toggle_running",
    ):
        live["running"] = not live.get("running", False)

    if ctrl[1].button("Inject harmless", key="live_inject_harmless"):
        live["sim"].inject("harmless")

    if ctrl[2].button("Inject sprint", key="live_inject_sprint"):
        live["sim"].inject("sprint")

    if ctrl[3].button("Inject loiter", key="live_inject_loiter"):
        live["sim"].inject("loiter")

    if ctrl[4].button("Record 12s", key="live_record_clip"):
        start_live_recording(live, 12)

    st.caption(
        "The monitor flags unusual motion patterns. Reviewer feedback distinguishes "
        "harmful events, unusual-but-harmless events, and false alarms."
    )


try:
    live_monitor = st.fragment(live_monitor_impl, run_every=0.2)
except Exception:
    live_monitor = live_monitor_impl


# ----------------------------------------------------------------------------
# UI sections
# ----------------------------------------------------------------------------

def render_live_tab():
    demo = st.session_state.demo

    st.subheader("Automatically generated demo video")

    if demo.get("video_bytes"):
        st.video(demo["video_bytes"], format=demo.get("video_format", "video/mp4"))

        ext = "mp4" if "mp4" in demo.get("video_format", "video/mp4") else "avi"
        st.download_button(
            "Download demo video",
            data=demo["video_bytes"],
            file_name=f"video_behavior_anomaly_demo.{ext}",
            mime=demo.get("video_format", "video/mp4"),
            key="download_demo_video",
        )
    else:
        st.warning(
            "A video encoder was not available in this environment. "
            "The live frame simulation below still works."
        )

    st.caption(
        "The demo contains normal pedestrian-like motion plus sprinting, loitering, "
        "reverse flow, an abandoned object, and an unusual-but-harmless service robot."
    )

    st.divider()

    live_monitor()

    st.divider()

    live = st.session_state.live
    if live.get("clip_bytes"):
        st.subheader("Recorded live clip")

        fmt = live.get("clip_format", "video/mp4")
        ext = "mp4" if "mp4" in fmt else "avi"

        st.video(live["clip_bytes"], format=fmt)
        st.download_button(
            "Download live clip",
            data=live["clip_bytes"],
            file_name=f"live_behavior_clip.{ext}",
            mime=fmt,
            key="download_live_clip",
        )


def render_alerts_tab():
    st.subheader("Alert queue and evidence")

    alerts = st.session_state.demo["alerts"]

    if not alerts:
        st.success("No alerts were generated with the current settings.")
    else:
        rows = []
        for a in alerts:
            rows.append(
                {
                    "ID": a["id"],
                    "Time (s)": round(a["time"], 2),
                    "Track": a["track_id"],
                    "Score": round(a["score"], 3),
                    "Predicted": a["predicted_label"],
                    "Reviewer": a["review_label"] or "",
                    "Evidence": evidence_to_str(a["evidence"]),
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        labels = [
            f"#{a['id']} | {a['time']:.1f}s | track {a['track_id']} | score {a['score']:.2f}"
            for a in reversed(alerts)
        ]

        choice = st.selectbox("Choose an alert to inspect", labels, key="alert_selector")
        alert = alerts[len(alerts) - 1 - labels.index(choice)]

        st.markdown(
            f"**Alert #{alert['id']}** — predicted `{alert['predicted_label']}` "
            f"(confidence {alert['predicted_conf']:.2f}), "
            f"reviewer `{alert['review_label'] or 'pending'}`"
        )

        clip = alert.get("clip", [])
        if clip:
            st.write("Clip evidence around the alert:")
            picks = np.linspace(0, len(clip) - 1, min(6, len(clip)), dtype=int)
            cols = st.columns(len(picks))

            for col, idx in zip(cols, picks):
                buf = np.frombuffer(clip[int(idx)], dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is not None:
                    col.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("No clip frames were stored for this alert.")

        st.write("Top feature evidence:")
        ev_cols = st.columns(3)

        for i, (feat, z) in enumerate(alert["evidence"]):
            with ev_cols[i % 3]:
                st.metric(feat, f"z={float(z):.2f}")
                st.progress(float(min(1.0, float(z) / 5.0)))

        fb = st.columns(4)

        detector = (
            st.session_state.live["detector"]
            if "live" in st.session_state
            else st.session_state.demo["detector"]
        )

        if fb[0].button("Mark harmful", key=f"demo_mark_harmful_{alert['id']}"):
            apply_feedback(alert, "harmful", detector, "demo")

        if fb[1].button(
            "Mark unusual harmless",
            key=f"demo_mark_harmless_{alert['id']}",
        ):
            apply_feedback(alert, "unusual_harmless", detector, "demo")

        if fb[2].button(
            "Mark false alarm",
            key=f"demo_mark_false_{alert['id']}",
        ):
            apply_feedback(alert, "false_alarm", detector, "demo")

        if fb[3].button("Clear learned feedback", key="clear_learned_feedback"):
            if "live" in st.session_state:
                st.session_state.live["detector"].clear_learned()
            st.success("Learned feedback prototypes cleared for the live baseline.")

    st.subheader("Feedback log")

    if st.session_state.get("feedback_log"):
        st.dataframe(pd.DataFrame(st.session_state.feedback_log), use_container_width=True)
    else:
        st.caption("No reviewer feedback recorded yet.")


def render_evaluation_tab():
    demo = st.session_state.demo
    m = demo["metrics"]

    st.subheader("Demo evaluation")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Event recall", f"{m['recall_all'] * 100:.0f}%")
    c2.metric("Harmful recall", f"{m['recall_harmful'] * 100:.0f}%")
    c3.metric("False alarms / hour", f"{m['false_alarms_per_hour']:.1f}")

    if m["mean_delay_sec"] is None:
        c4.metric("Mean detection delay", "n/a")
    else:
        c4.metric("Mean detection delay", f"{m['mean_delay_sec']:.2f}s")

    st.caption(
        f"Demo duration: {m['duration_sec']:.0f}s | "
        f"ground-truth events: {m['total_events']} | "
        f"alert episodes: {m['total_alerts']} | "
        f"false alarm episodes: {m['false_alarms']}"
    )

    st.dataframe(demo["events_df"], use_container_width=True)

    plot_score_timeline(
        demo["score_df"],
        demo["events"],
        st.session_state.params["threshold"],
    )

    st.caption(
        "Red shaded windows are harmful/review events. Orange shaded windows are "
        "unusual-but-harmless events. The dashed line is the alert threshold."
    )

    st.divider()
    st.subheader("Live session evaluation")

    if st.button("Compute live metrics", key="compute_live_metrics"):
        compute_live_metrics()

    live = st.session_state.live

    if live.get("metrics"):
        lm = live["metrics"]

        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Live events", lm["total_events"])
        lc2.metric("Live alerts", lm["total_alerts"])
        lc3.metric("Live false alarms/hour", f"{lm['false_alarms_per_hour']:.1f}")

        if lm["mean_delay_sec"] is None:
            lc4.metric("Live mean delay", "n/a")
        else:
            lc4.metric("Live mean delay", f"{lm['mean_delay_sec']:.2f}s")

        st.dataframe(live["events_df"], use_container_width=True)
    else:
        st.caption("Run the live simulation, inject events, then compute live metrics.")


def render_docs_tab():
    st.subheader("What this demo implements")

    st.markdown(
        """
This app implements the **Real-Time Video Behavior Anomaly Monitor** mission:

- Learn normal motion patterns from an unlabeled baseline window.
- Identify unusual changes in tracked movement features.
- Provide evidence rather than only an unexplained score.
- Allow reviewers to label alerts as harmful, unusual-but-harmless, or false alarms.
- Generate a synthetic video file automatically so the demo is not blank.
- Run a live interactive simulation with injectable events.
        """
    )

    st.markdown(
        """
### Modules

1. **Synthetic video generator**
   - Creates a surveillance-like scene with moving objects.
   - Includes normal traffic and unusual events.
   - Writes an MP4/AVI demo file automatically.

2. **Tracker and feature extractor**
   - Uses OpenCV background subtraction.
   - Associates detections into short-lived tracks.
   - Computes speed, acceleration, direction change, dwell time, area,
     local density, frame motion energy, spatial spread, and central-zone dwell.

3. **Temporal baseline and anomaly model**
   - Trains an Isolation Forest on the early normal window.
   - Uses robust z-scores for interpretable feature evidence.
   - Combines model anomaly score with feature deviation.

4. **Alerting and clip evidence**
   - Requires persistence before raising an alert.
   - Stores a short JPEG frame clip around each alert.
   - Shows top contributing features for operator review.

5. **Reviewer feedback and online learning guardrails**
   - Feedback can mark harmful, unusual harmless, or false alarm.
   - False alarms can suppress similar future patterns.
   - Harmful/harmless prototypes help predict future event type.
   - Learned feedback can be cleared/rolled back.

6. **Evaluation**
   - Event recall.
   - Harmful-event recall.
   - False alarms per hour.
   - Detection delay.
   - Timeline plot with ground-truth event windows.
        """
    )

    st.markdown(
        """
### Suggested production stack

The demo uses a lightweight local stack:

- Streamlit for the operator dashboard.
- pandas for tabular evidence and metrics.
- scikit-learn for the baseline anomaly model.
- OpenCV for synthetic video, tracking, and clip evidence.
- Matplotlib for evaluation plots.

For a production deployment, the same architecture can be extended with:

- PyTorch or a pretrained video model for richer embeddings.
- River for true online learning.
- A time-series database for feature history.
- A message bus for alert routing.
- A review queue with auditable feedback controls.
        """
    )

    st.markdown(
        """
### Important limitation

This is a synthetic educational demo. It is not a security product and should not be
used to make high-stakes decisions without validation on real representative data,
privacy review, and human-centered operating procedures.
        """
    )


# ----------------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------------

def main():
    if "params" not in st.session_state:
        st.session_state.params = {
            "threshold": 0.65,
            "contamination": 0.05,
            "learn": True,
        }

    st.sidebar.title("Mission controls")

    threshold = st.sidebar.slider(
        "Alert threshold",
        min_value=0.30,
        max_value=0.95,
        value=float(st.session_state.params.get("threshold", 0.65)),
        step=0.01,
        key="threshold_slider",
    )

    contamination = st.sidebar.slider(
        "Baseline contamination",
        min_value=0.01,
        max_value=0.20,
        value=float(st.session_state.params.get("contamination", 0.05)),
        step=0.01,
        key="contamination_slider",
    )

    learn = st.sidebar.checkbox(
        "Learn from reviewer feedback",
        value=bool(st.session_state.params.get("learn", True)),
        key="learn_checkbox",
    )

    st.session_state.params.update(
        {
            "threshold": float(threshold),
            "contamination": float(contamination),
            "learn": bool(learn),
        }
    )

    if "live" in st.session_state:
        st.session_state.live["detector"].threshold = float(threshold)
        st.session_state.live["alert_mgr"].threshold = float(threshold)

    st.sidebar.markdown("---")

    if st.sidebar.button("Regenerate demo with current settings"):
        for k in ["demo", "live", "feedback_log"]:
            st.session_state.pop(k, None)
        st.rerun()

    if st.sidebar.button("Reset all state"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.info(
        "No API key required. The app generates its own synthetic video demo and "
        "starts a live simulation automatically."
    )

    if "demo" not in st.session_state:
        with st.spinner(
            "Generating realistic synthetic demo video, normal baseline, alerts, and metrics..."
        ):
            st.session_state.demo = generate_demo(
                threshold=st.session_state.params["threshold"],
                contamination=st.session_state.params["contamination"],
            )
        st.session_state.feedback_log = []

    if "live" not in st.session_state:
        ensure_live()

    st.title("🎥 Real-Time Video Behavior Anomaly Monitor")
    st.subheader("Evidence-first unusual motion monitoring with reviewer feedback")

    st.markdown(
        """
This demo automatically generates a synthetic surveillance-style video, learns a
normal motion baseline, flags unusual intervals, shows the features that contributed
to each alert, and lets operators review events without treating every unusual motion
as malicious.
        """
    )

    tab_live, tab_alerts, tab_eval, tab_docs = st.tabs(
        ["Live video demo", "Alert evidence", "Evaluation", "How it works"]
    )

    with tab_live:
        render_live_tab()

    with tab_alerts:
        render_alerts_tab()

    with tab_eval:
        render_evaluation_tab()

    with tab_docs:
        render_docs_tab()


if __name__ == "__main__":
    main()