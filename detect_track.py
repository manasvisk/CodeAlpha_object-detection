#!/usr/bin/env python3
"""
Simple real-time video detection + tracking using a pre-trained YOLO model (Ultralytics)
and a lightweight centroid tracker. Overlays images on detected classes when available.

Requirements:
  pip install -r requirements.txt

Usage:
  python detect_track.py              # open default webcam (0)
  python detect_track.py --video file.mp4
  python detect_track.py --overlay-dir overlays/

Notes:
 - The script uses the Ultralytics package to load a YOLOv8 model (it will auto-download weights).
 - A simple centroid-based tracker assigns persistent IDs to detections.
 - Overlay images can be placed in an overlays directory and mapped by class name.
"""

import argparse
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO


class CentroidTracker:
    """Very small centroid tracker: assigns IDs to object centroids.
    Not as robust as SORT/DeepSORT but simple and dependency-free.
    """
    def __init__(self, maxDisappeared=30, maxDistance=50):
        self.nextObjectID = 0
        self.objects = dict()  # objectID -> centroid
        self.disappeared = dict()  # objectID -> frames disappeared
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # rects = list of bounding boxes [startX, startY, endX, endY]
        if len(rects) == 0:
            # increment disappeared for all
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute distance matrix between existing and new
            D = np.linalg.norm(np.array(objectCentroids)[:, None] - inputCentroids[None, :], axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                oid = objectIDs[row]
                self.objects[oid] = inputCentroids[col]
                self.disappeared[oid] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    oid = objectIDs[row]
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.maxDisappeared:
                        self.deregister(oid)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


def overlay_image(base_img, overlay_img, x, y, w, h):
    """Overlay overlay_img onto base_img at position (x,y) with size (w,h).
    overlay_img may have alpha channel; it will be resized to (w,h).
    """
    try:
        overlay = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        return base_img

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            base_img[y:y+h, x:x+w, c] = (alpha * overlay[:, :, c] + (1 - alpha) * base_img[y:y+h, x:x+w, c])
    else:
        base_img[y:y+h, :3] = overlay[:, :, :3]
    return base_img


def main():
    parser = argparse.ArgumentParser(description='Real-time YOLO detection + simple tracking + overlay')
    parser.add_argument('--video', '-v', help='Path to video file (omit to use webcam)', default=None)
    parser.add_argument('--overlay-dir', '-o', help='Directory with overlay PNGs named by class (e.g., person.png)', default='overlays')
    parser.add_argument('--model', '-m', help='YOLO model (ultralytics) spec, e.g. yolov8n.pt', default='yolov8n.pt')
    parser.add_argument('--conf', help='Confidence threshold', type=float, default=0.35)
    parser.add_argument('--max-distance', help='Tracker max centroid distance', type=int, default=80)
    args = parser.parse_args()

    print('[INFO] Loading model:', args.model)
    model = YOLO(args.model)

    # load overlays
    overlays = {}
    if os.path.isdir(args.overlay_dir):
        for fname in os.listdir(args.overlay_dir):
            name, ext = os.path.splitext(fname)
            if ext.lower() in ('.png', '.jpg', '.jpeg'):
                p = os.path.join(args.overlay_dir, fname)
                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    overlays[name.lower()] = img
    else:
        print(f'[WARN] Overlay dir "{args.overlay_dir}" not found — continuing without overlays')

    tracker = CentroidTracker(maxDisappeared=40, maxDistance=args.max_distance)

    if args.video:
        vs = cv2.VideoCapture(args.video)
    else:
        vs = cv2.VideoCapture(0)

    prev_time = time.time()

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break

        h, w = frame.shape[:2]

        # run model (Ultralytics) on the current frame
        results = model(frame, imgsz=640, conf=args.conf, verbose=False)
        boxes = []
        labels = []
        confidences = []

        # results is a list; take first
        if len(results) > 0:
            r = results[0]
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy()) if hasattr(box, 'conf') else 0.0
                    cls = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else -1
                    name = model.names[cls] if cls in model.names else str(cls)
                    boxes.append((x1, y1, x2, y2))
                    labels.append(name)
                    confidences.append(conf)

        objects = tracker.update(boxes)

        # draw detections and overlays
        for i, (bbox, lbl, conf) in enumerate(zip(boxes, labels, confidences)):
            x1, y1, x2, y2 = bbox
            # find matching object ID by nearest centroid
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            assigned_id = None
            minDist = float('inf')
            for oid, centroid in objects.items():
                d = np.linalg.norm(np.array(centroid) - np.array((cX, cY)))
                if d < minDist:
                    minDist = d; assigned_id = oid

            color = (70, 190, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{lbl} {conf:.2f} ID:{assigned_id if assigned_id is not None else "-"}'
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # overlay image if available
            key = lbl.lower()
            if key in overlays:
                ow = x2 - x1
                oh = int(ow * overlays[key].shape[0] / max(1, overlays[key].shape[1]))
                oy = max(0, y1 - oh - 6)
                ox = x1
                # keep inside frame
                if ox + ow > w: ow = w - ox
                if oy < 0: oy = 0
                try:
                    frame = overlay_image(frame, overlays[key], ox, oy, ow, oh)
                except Exception:
                    pass

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow('YOLO Detect + Track', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
