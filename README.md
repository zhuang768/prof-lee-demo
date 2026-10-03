# Python Webcam Vision Demo

A Python webcam demonstration combining object detection, human pose analysis, and a live analytics sidebar. I prepared this project to present to a professor. It later motivated me to explore ESP32 sensing and physical control through a separate self-balancing robot project.

## Implemented in the Source

- YOLOv8n object detection with class labels, confidence scores, bounding boxes, and per-frame object counts.
- MediaPipe pose landmarks and a virtual steering-wheel angle estimated from the relative positions of the wrists.
- An on-screen sidebar showing FPS, object-inference time, object counts, and steering status.
- Threaded webcam capture, with camera indices 0–2 checked at startup.
- Keyboard toggles for object detection and pose analysis, plus screenshot capture.

The current implementation calls `model.predict()`, not `model.track()`. It performs per-frame detection rather than persistent object-ID tracking, despite some older labels in the source using the word “tracking.”

## Repository Layout

```text
README.md
程式碼與專案檔案/
  main.py
  yolov8n.pt
即時Webcam物件追蹤與姿態辨識_開發紀錄.docx
```

The Word document is an earlier development record. The executable implementation is `main.py`.

## Environment and Running

You need Python, a webcam, a desktop environment that supports OpenCV windows, and these third-party packages:

- `opencv-python`
- `numpy`
- `ultralytics`
- `mediapipe` with the legacy `mp.solutions.pose` API

No dependency lockfile or verified cross-platform installation matrix is currently provided. Check that your MediaPipe version exposes `mp.solutions.pose`; versions without that API cannot run this source unchanged. Model loading or the first use of pose estimation may require downloading model assets, depending on the installed packages and cache.

Run from the model's directory so the relative `yolov8n.pt` path resolves correctly:

```sh
cd "程式碼與專案檔案"
python3 main.py
```

Grant camera access to the terminal or Python application if your operating system requests it. Run locally, not in a headless server environment.

## Controls

| Key | Action |
| --- | --- |
| `p` | Toggle pose analysis |
| `o` | Toggle object detection |
| `s` | Save the displayed dashboard under `screenshots/` in the working directory |
| `q` | Quit |

## Limitations and Validation

- Performance depends on the computer, camera, lighting, and scene. No fixed FPS or accuracy guarantee is claimed.
- The steering angle is a visual demonstration based on wrist landmarks, not a vehicle-control system or a validated biomechanical measurement.
- This project does not currently send commands to the ESP32 robot. Vision-to-robot integration is a future goal.
- The source sets `KMP_DUPLICATE_LIB_OK=TRUE` as an earlier macOS workaround. It does not establish that every native-library combination is compatible.
- A camera session and package compatibility were not re-tested as part of this documentation update.

## Privacy

The source processes camera frames in the local application and contains no explicit frame-upload feature. Screenshots are saved locally only when `s` is pressed. Avoid publishing identifiable images of other people without their consent. Third-party packages may download model assets or have their own network behavior; this project has not been audited for fully offline operation.

## Related Work

- [ESP32 self-balancing robot](https://github.com/zhuang768/esp32-self-balancing-robot) — a separate embedded-control prototype, still undergoing hardware integration and validation.
- [GitHub profile](https://github.com/zhuang768)

The professor presentation is the project's context, not a claim of academic endorsement or a formal research collaboration.
