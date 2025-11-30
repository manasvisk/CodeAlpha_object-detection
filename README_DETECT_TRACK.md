Real-time YOLO detection + tracking (simple)
=========================================

Overview
--------
This example uses the Ultralytics YOLO package to run a pretrained model (yolov8n by default) on webcam or video input, applies a small centroid-based tracker to provide persistent IDs, and demonstrates how to overlay images for detected classes.

Setup
-----
1. Create a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Create an `overlays/` directory and place PNG/JPG files named by class, e.g. `person.png`, `dog.png`.

You can download a sample overlay like this:

```bash
mkdir -p overlays
curl -L -o overlays/person.png "https://images.unsplash.com/photo-1554151228-14d9def656e4?q=80&w=600&auto=format&fit=crop"
```

Running
-------
Run with webcam:

```bash
python detect_track.py
```

Run with a video file and overlay directory:

```bash
python detect_track.py --video sample.mp4 --overlay-dir overlays
```

Controls
--------
- Press `q` to quit the live window.

Notes & Next Steps
------------------
- The centroid tracker is intentionally simple; for production use consider integrating SORT or DeepSORT for better association.
- You can swap `--model yolov8n.pt` for `yolov8m.pt` or custom weights.
- For headless servers you can save frames to disk instead of showing a window.
