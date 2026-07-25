<p align="center">
  <a href="./21653880.mp4">
    <img src="https://img.icons8.com/color/96/000000/play-button-circled.png" width="80" alt="Play">
    <br>
    <strong>▶ Watch Demo Video</strong>
  </a>
  <br>
  <sub>Click to view the UAV tracking demo</sub>
</p>

<h1 align="center">Real-Time UAV Tracking System</h1>

<p align="center">
  <strong>A Python-based real-time object detection and tracking system for UAVs (drones) using YOLOv8 and OpenCV.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00BBFF?style=flat-square&logo=ai" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
</p>

---

## Overview

This system processes input videos, detects UAVs frame-by-frame using **YOLOv8**, and outputs an annotated video with tracking overlays. It is designed for educational and research purposes in computer vision and drone tracking.

---

## Features

- Real-time UAV detection and tracking
- YOLOv8 (Ultralytics) for accurate object detection
- Annotated output video with bounding boxes and tracking info
- Supports Python 3.11–3.14
- Simple CLI interface

---

## Getting Started

### Prerequisites

- **Python 3.11, 3.12, 3.13, or 3.14**
- **git** (for cloning)

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/Linuxthecoder/real-time-uav-tracking-system.git
   cd real-time-uav-tracking-system
   ```

2. **Place your input video** (e.g., `video.mp4`) in the project folder.

3. **Run the tracker:**

   ```sh
   python main.py --source video.mp4
   ```

   The annotated output will be saved to `outputs/output.mp4`.

---

## Usage

```sh
python main.py --source <path-to-video>
```

| Argument     | Description                  |
|--------------|------------------------------|
| `--source`   | Path to the input video file |

---

## Tech Stack

- **Object Detection:** YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV
- **Language:** Python 3.11+

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Developer

<div align="center">
  <strong>Linuxthecoder</strong>
  <br>
  <a href="https://github.com/Linuxthecoder">GitHub</a>
</div>

---

<p align="center">
  <em>For educational and research purposes. Contributions and forks are welcome!</em>
</p>
