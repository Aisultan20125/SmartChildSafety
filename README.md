# SmartChildSafety
SmartChildSafety is an intelligent, real-time computer vision system designed to prevent domestic child injuries before they occur. While traditional baby monitors and security cameras passively record accidents after the fact, SmartChildSafety uses edge AI to actively detect hazards and notify parents in real time.

# 🛡️ SmartChildSafety — Local Edge AI Video Monitoring System

**SmartChildSafety** is an intelligent, real-time computer vision and Edge AI system designed to prevent domestic child injuries before they occur[cite: 1, 2]. 

Unlike traditional baby monitors that passively record incidents after the fact, SmartChildSafety uses edge AI to actively track hazards, monitor physical touch, and instantly alert parents[cite: 1, 2].

>  **Project Origin:** Developed in Almaty, Kazakhstan by **Almaty Youth STEM**.
> **Devolper: Aisultan**

---

## ✨ Key Features

* **Multi-Layered Detection Engine:** Combines HSV color masking, MediaPipe Hand Tracking, and YOLO vision models to monitor hazardous objects across the entire frame[cite: 2, 3].
* **Touch & Laser Boundary Tracking:** Calculates real-time distance vectors between hand landmarks and safety zones using virtual laser lines[cite: 3].
* **"Iron Grip" State Retention:** Maintains alert state (`TAKEN!`) even if the child partially occludes or hides the target object[cite: 3].
* **Instant Telegram Alerts:** Sends real-time photo evidence with annotated bounding boxes directly to caregivers[cite: 2, 3].
* **Two-Way Audio Interaction:**
  * **Voice Message Player:** Downloads voice messages sent via Telegram, converts OGG/Opus audio to WAV using `FFmpeg` & `pydub`, and plays them through local room speakers[cite: 3].
  * **Text-to-Speech (TTS):** Converts incoming Telegram text messages into spoken audio using `pyttsx3`[cite: 3].
* **Local Web Dashboard (Flask):** Live MJPEG video stream accessible from any browser on the local network (`:5000`)[cite: 3].
* **100% Edge Privacy:** All video frames are processed locally on the host machine without sending raw video streams to cloud servers[cite: 1, 3].

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Computer Vision & AI:** OpenCV, MediaPipe, Ultralytics YOLO, PyTorch (CUDA supported)[cite: 2, 3]
* **Web Engine:** Flask (MJPEG Live Stream & Web App Interface)[cite: 3]
* **Audio & Voice Processing:** Pygame Mixer, Pyttsx3, Pydub, FFmpeg[cite: 3]
* **Integration:** Telegram Bot API (Polling, Voice Downloads, Inline Keyboards)[cite: 2, 3]

---

## 🚀 Quick Start Guide

### 1. System Requirements
Make sure **FFmpeg** is installed on your system (required for Telegram voice message audio conversion)[cite: 3]:
* **macOS:** `brew install ffmpeg`[cite: 3]
* **Linux:** `sudo apt install ffmpeg`[cite: 3]
* **Windows:** Download from [gyan.dev/ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and add its `bin` folder to your System `PATH`[cite: 3].

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/SmartChildSafety.git](https://github.com/YOUR_USERNAME/SmartChildSafety.git)
   cd SmartChildSafety
