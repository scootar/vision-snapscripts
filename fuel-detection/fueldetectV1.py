# Limelight 3 SnapScript — Coral TPU Neural Detector with Clump Detection
# -------------------------------------------------------------------------
# This script:
#   1. Runs a neural object-detection model on an attached Google Coral TPU
#      (Edge TPU USB Accelerator) every frame.
#   2. Clusters ("clumps") all recognised detections that are spatially close
#      into groups.  A clump must contain at least 3 detections to qualify.
#      Each clump is represented by the convex hull polygon of its member
#      bounding-box corners.
#   3. Selects the largest clump (by convex-hull area), computes its
#      approximate centroid, and returns a small contour centred on that
#      point so the Limelight's built-in targeting maths (tx / ty) work
#      directly.
#
# The eight-element llpython array published to NetworkTables contains:
#   [0] = 1.0 if a valid clump exists, else 0.0
#   [1] = centroid X of the largest clump (pixels)
#   [2] = centroid Y of the largest clump (pixels)
#   [3] = area (pixels²) of the largest clump convex hull
#   [4] = number of detections in the largest clump
#   [5] = total number of detections this frame
#   [6] = total number of qualifying clumps (>= 3 members)
#   [7] = inference time in milliseconds
#
# CONFIGURATION — adjust the four constants below before deploying.
# -------------------------------------------------------------------------

import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# ---------------------------------------------------------------------------
# User-tuneable constants
# ---------------------------------------------------------------------------
# Path to the Edge-TPU-compiled TFLite model on the Limelight's filesystem.
# Upload your model via the Limelight web UI or scp.
MODEL_PATH = "/home/pi/models/detect_edgetpu.tflite"

# Path to the Edge TPU runtime shared library (standard on Limelight 3).
EDGETPU_LIB = "libedgetpu.so.1"

# Minimum confidence score for a detection to be kept.
SCORE_THRESHOLD = 0.45

# Maximum pixel distance between two detection centres for them to be
# considered part of the same clump.  Tune this for your field-of-view and
# expected target spacing.
CLUMP_DISTANCE_PX = 120

# Minimum number of detections required to form a valid clump.
MIN_CLUMP_SIZE = 3

# ---------------------------------------------------------------------------
# One-time initialisation (runs once when the pipeline is loaded)
# ---------------------------------------------------------------------------
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[tflite.load_delegate(EDGETPU_LIB)],
)
interpreter.allocate_tensors()

# Cache input/output tensor details for fast per-frame access
_input_details = interpreter.get_input_details()
_output_details = interpreter.get_output_details()
_input_shape = _input_details[0]["shape"]          # e.g. [1, 300, 300, 3]
_model_h = _input_shape[1]
_model_w = _input_shape[2]


# ---------------------------------------------------------------------------
# Helper — Union-Find for clustering
# ---------------------------------------------------------------------------
class UnionFind:
    """Lightweight disjoint-set / union-find structure."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _cluster_detections(centers, max_dist):
    """Return a list of lists of indices — one list per cluster.

    Uses union-find with an O(n²) pairwise distance check.  For the
    expected detection counts on an FRC field (< ~50) this is instant.
    """
    n = len(centers)
    if n == 0:
        return []

    uf = UnionFind(n)
    max_dist_sq = max_dist * max_dist

    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if dx * dx + dy * dy <= max_dist_sq:
                uf.union(i, j)

    clusters = {}
    for i in range(n):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)

    return list(clusters.values())


# ---------------------------------------------------------------------------
# Main pipeline — called every frame by Limelight OS
# ---------------------------------------------------------------------------
def runPipeline(image, llrobot):
    """Process a single camera frame.

    Parameters
    ----------
    image : numpy.ndarray
        BGR image from the Limelight camera.
    llrobot : list[float]
        Eight-element array sent from robot code via NetworkTables
        (``llrobot`` key).  Available for two-way communication.

    Returns
    -------
    (largestContour, image, llpython)
        largestContour : numpy.ndarray — contour sent to Limelight targeting.
        image          : numpy.ndarray — annotated BGR image for the stream.
        llpython       : list[float]   — up to 8 values for the ``llpython``
                                          NetworkTables entry.
    """

    img_h, img_w = image.shape[:2]
    llpython = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    largest_contour = np.array([[]])

    # ------------------------------------------------------------------
    # 1.  Run the Coral TPU detector
    # ------------------------------------------------------------------
    # Resize the frame to the model's expected input dimensions.
    img_resized = cv2.resize(image, (_model_w, _model_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Copy into the input tensor (uint8 quantised models expect 0-255).
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
    interpreter.set_tensor(_input_details[0]["index"], input_data)

    t0 = time.perf_counter()
    interpreter.invoke()
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------
    #  Parse SSD-style output tensors
    #  Standard TFLite detection models output four tensors:
    #    0 — bounding boxes  [1, N, 4]  (ymin, xmin, ymax, xmax) normalised
    #    1 — class IDs       [1, N]
    #    2 — scores          [1, N]
    #    3 — detection count [1]
    # ------------------------------------------------------------------
    raw_boxes  = interpreter.get_tensor(_output_details[0]["index"])[0]   # [N,4]
    raw_ids    = interpreter.get_tensor(_output_details[1]["index"])[0]   # [N]
    raw_scores = interpreter.get_tensor(_output_details[2]["index"])[0]   # [N]
    raw_count  = int(interpreter.get_tensor(_output_details[3]["index"])[0])

    # Scale normalised coords back to the original image dimensions.
    sx, sy = float(img_w), float(img_h)

    objs = []  # list of (xmin, ymin, xmax, ymax) in image-pixel coords
    for i in range(raw_count):
        if raw_scores[i] < SCORE_THRESHOLD:
            continue
        ymin, xmin, ymax, xmax = raw_boxes[i]
        objs.append((
            max(0.0, xmin * sx),
            max(0.0, ymin * sy),
            min(sx,  xmax * sx),
            min(sy,  ymax * sy),
        ))

    num_detections = len(objs)
    llpython[5] = float(num_detections)
    llpython[7] = round(inference_ms, 2)

    if num_detections == 0:
        _draw_hud(image, [], None, 0, inference_ms)
        return largest_contour, image, llpython

    # ------------------------------------------------------------------
    # 2.  Collect bounding-box info & cluster into clumps
    # ------------------------------------------------------------------
    centers = []
    boxes = []  # (xmin, ymin, xmax, ymax) in image coords

    for (xmin, ymin, xmax, ymax) in objs:
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        centers.append((cx, cy))
        boxes.append((xmin, ymin, xmax, ymax))

        # Draw individual detection boxes (thin blue)
        cv2.rectangle(
            image,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            (255, 180, 0),
            1,
        )

    clusters = _cluster_detections(centers, CLUMP_DISTANCE_PX)

    # Keep only clumps with enough members
    valid_clumps = [c for c in clusters if len(c) >= MIN_CLUMP_SIZE]
    llpython[6] = float(len(valid_clumps))

    if not valid_clumps:
        _draw_hud(image, [], None, num_detections, inference_ms)
        return largest_contour, image, llpython

    # ------------------------------------------------------------------
    # 3.  Build convex hulls, find the largest, extract centroid
    # ------------------------------------------------------------------
    best_hull = None
    best_area = -1.0
    best_count = 0

    for clump_indices in valid_clumps:
        # Gather all four corners of every bbox in this clump
        pts = []
        for idx in clump_indices:
            xmin, ymin, xmax, ymax = boxes[idx]
            pts.extend([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ])

        pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts_np)
        area = cv2.contourArea(hull)

        # Draw every clump hull in green
        cv2.drawContours(image, [hull.astype(np.int32)], -1, (0, 255, 0), 2)

        if area > best_area:
            best_area = area
            best_hull = hull
            best_count = len(clump_indices)

    # Highlight the winning hull
    cv2.drawContours(
        image, [best_hull.astype(np.int32)], -1, (0, 255, 255), 3
    )

    # Centroid via image moments
    M = cv2.moments(best_hull)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        # Fallback: mean of hull points
        cx = float(np.mean(best_hull[:, 0, 0]))
        cy = float(np.mean(best_hull[:, 0, 1]))

    # Draw the centroid
    cv2.circle(image, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    cv2.circle(image, (int(cx), int(cy)), 12, (255, 255, 255), 2)

    # Build a small square contour around the centroid so Limelight can
    # compute tx / ty from it.
    half = 10
    largest_contour = np.array([
        [[int(cx - half), int(cy - half)]],
        [[int(cx + half), int(cy - half)]],
        [[int(cx + half), int(cy + half)]],
        [[int(cx - half), int(cy + half)]],
    ], dtype=np.int32)

    # ------------------------------------------------------------------
    # 4.  Populate llpython
    # ------------------------------------------------------------------
    llpython[0] = 1.0                # valid target flag
    llpython[1] = round(cx, 1)       # centroid X (px)
    llpython[2] = round(cy, 1)       # centroid Y (px)
    llpython[3] = round(best_area, 1)  # hull area (px²)
    llpython[4] = float(best_count)  # detections in largest clump

    _draw_hud(image, valid_clumps, (cx, cy), num_detections, inference_ms)

    return largest_contour, image, llpython


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------
def _draw_hud(image, clumps, centroid, num_det, inf_ms):
    """Draw a small heads-up display in the top-left corner."""
    h, w = image.shape[:2]
    lines = [
        f"Detections: {num_det}",
        f"Clumps (>={MIN_CLUMP_SIZE}): {len(clumps)}",
        f"Inference: {inf_ms:.1f} ms",
    ]
    if centroid is not None:
        lines.append(f"Target: ({centroid[0]:.0f}, {centroid[1]:.0f})")

    y0 = 25
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (10, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
