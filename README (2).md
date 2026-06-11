# 🎬 AI Video Content Moderation

An automated video moderation pipeline that detects **visually unsafe content** and **harmful audio** in videos using **CLIP**, **BLIP**, and **Whisper** — fully CPU-compatible, no cloud API required.

---

## 🧠 How It Works

The pipeline runs in two stages:

```
Video File
    │
    ├──► Frame Sampling (1 frame / 2 sec)
    │         │
    │         ├──► CLIP  → Visual risk scoring against unsafe prompts
    │         └──► BLIP  → Caption generation for context filtering
    │
    └──► Audio Extraction (only if visually SAFE)
              │
              └──► Whisper → Transcription + bad word detection
```

### Stage 1 — Visual Analysis
- Samples one frame every 2 seconds using OpenCV
- **BLIP** generates a natural-language caption per frame
- **CLIP** scores each frame against 10 risk categories (nudity, violence, weapons, drugs, etc.)
- Context filtering suppresses false positives (e.g. kitchen knives, medical nudity, shirtless athletes)
- Early exit if confidence > 0.90 to save compute time

### Stage 2 — Audio Analysis
- Only runs if the video passes visual analysis
- Extracts audio in-memory (no `.wav` file saved to disk)
- **Whisper (tiny)** transcribes speech
- Regex pattern matching detects unsafe words across categories: profanity, sexual content, violence, weapons, drugs

---

## 🚨 Detection Categories

| Category | Threshold | Examples |
|---|---|---|
| Nudity / Explicit Content | 0.85 | nudity, pornographic content |
| Sexual Activity | 0.65 | kissing, sexual activity |
| Graphic Violence | 0.78 | blood, gore |
| Weapons | 0.70 | guns, knives |
| Offensive Gestures | 0.75 | middle finger |
| Drug Use | 0.60 | drug paraphernalia |

---

## 🛠️ Tech Stack

| Model | Purpose | Source |
|---|---|---|
| **CLIP ViT-B/32** | Visual risk scoring | OpenAI |
| **BLIP Base** | Frame captioning & context | Salesforce |
| **Whisper tiny** | Speech transcription | OpenAI |
| **OpenCV** | Frame extraction | — |
| **MoviePy** | Audio extraction | — |

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/Chaitanya-kandagaddala-2006/ai-video-moderation
cd ai-video-moderation

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install openai-clip
pip install openai-whisper
pip install transformers pillow opencv-python moviepy tqdm
```

> **Note:** No GPU required. Tested on CPU (Windows). For faster inference, a CUDA-enabled GPU is automatically used if available.

---

## 🚀 Usage

1. Place your video file in the project folder.
2. Edit the video path in `main.py`:

```python
video_path = "your_video.mp4"   # line 24
```

3. Run:

```bash
python main.py
```

### Example Output

```
Video: test12.mp4
Device: cpu
FPS: 30  Frames: 4500

==============================
🚫 VIDEO VISUALLY FLAGGED AS UNSAFE

Main reason : graphic violence
Confidence  : 0.83
Time        : 14 sec
BLIP reason : a person holding a weapon in a dark room
==============================

==============================
🚫 FINAL RESULT : UNSAFE
==============================
```

---

## 📁 Project Structure

```
ai-video-moderation/
│
├── main.py              # Main pipeline script
├── README.md            # This file
├── requirements.txt     # Dependencies
└── test_videos/         # Sample test videos (not tracked by git)
```

---

## ⚙️ Configuration

You can tune thresholds in `main.py` to adjust sensitivity:

```python
TH = {
    "nudity": 0.85,           # raise to reduce false positives
    "kissing scene": 0.65,    # lower = more sensitive
    "weapon": 0.70,
    "drug use": 0.60,
    ...
}
```

And add/remove bad-word patterns in the `bad_patterns` list for custom audio rules.

---

## 🔍 Context Filtering (False Positive Reduction)

The pipeline automatically suppresses detections when BLIP captions suggest safe context:

| Detected | Safe Context (suppressed if caption contains) |
|---|---|
| Nudity / Explicit | `sign`, `poster`, `menu`, `logo` (e.g. billboard) |
| Violence / Gore | `kitchen`, `cooking`, `food`, `chef` |
| Nudity / Explicit | `doctor`, `hospital`, `medical` |
| Shirtless | `beach`, `pool`, `gym`, `workout`, `swimming` |

---

## 📊 Test Results

| Video | Visual Result | Audio Result | Final |
|---|---|---|---|
| test1.mp4 | ✅ Safe | ✅ Safe | ✅ SAFE |
| test2.mp4 | ✅ Safe | ✅ Safe | ✅ SAFE |
| test3.mp4 | 🚫 Flagged | — | 🚫 UNSAFE |
| test12.mp4 | 🚫 Flagged | — | 🚫 UNSAFE |

---

## 🤝 Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Salesforce BLIP](https://github.com/salesforce/BLIP)
- Built during **AI Engineer Internship at RayV** (2025–2026)

---

## 👤 Author

**K.L.V Chaitanya Kandagaddala**
[LinkedIn](https://linkedin.com/in/klvchaitanya) · [GitHub](https://github.com/Chaitanya-kandagaddala-2006) · klvchaitanya2006@gmail.com
