# ==========================================================
# Fast AI Video Content Moderation (VS Code Version)
# CLIP (visual risk) + BLIP (context) + Whisper (audio)
# Optimized for Windows CPU
# ==========================================================

import os
import cv2
import torch
import clip
import whisper
import re
from PIL import Image
from tqdm import tqdm
from moviepy import VideoFileClip
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np


# -------------------------
# VIDEO PATH (EDIT HERE)
# -------------------------
video_path = "test12.mp4"   # keep video in same folder
print("Video:", video_path)


# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# -------------------------
# Load CLIP
# -------------------------
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()


# -------------------------
# Load BLIP
# -------------------------
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

blip_model.eval()


# -------------------------
# Load Whisper (tiny for CPU)
# -------------------------
whisper_model = whisper.load_model("tiny")


# -------------------------
# BLIP caption helper
# -------------------------
@torch.no_grad()
def get_caption(pil_image):
    inputs = blip_processor(images=pil_image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_new_tokens=20)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption.lower()


# -------------------------
# CLIP prompts
# -------------------------
prompts = [
    "a safe scene",
    "nudity",
    "pornographic content",
    "explicit content",
    "kissing scene",
    "sexual activity",
    "graphic violence",
    "blood or gore",
    "weapon",
    "a person showing a middle finger",
    "drug use"
]

text_tokens = clip.tokenize(prompts).to(device)

TH = {
    "nudity": 0.85,
    "pornographic content": 0.85,
    "explicit content": 0.85,
    "kissing scene": 0.65,
    "sexual activity": 0.65,
    "graphic violence": 0.78,
    "blood or gore": 0.78,
    "weapon": 0.70,
    "a person showing a middle finger": 0.75,
    "drug use": 0.60
}

shirtless_kw = ["shirtless", "topless", "bare chest", "without a shirt"]
safe_shirtless_ctx = ["beach", "pool", "swimming", "gym", "workout", "fitness", "surfing", "sport"]

food_ctx = ["kitchen", "cooking", "food", "chef", "restaurant", "pan"]
sign_ctx = ["sign", "poster", "menu", "logo", "banner", "billboard"]
medical_ctx = ["doctor", "hospital", "medical"]


# -------------------------
# Frame Sampling
# -------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = int(cap.get(cv2.CAP_PROP_FPS))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

interval = fps * 2 if fps > 0 else 1   # 1 frame every 2 seconds
print("FPS:", fps, "Frames:", total)


# -------------------------
# Main Loop
# -------------------------
BATCH = 4

frames = []
frame_ids = []
captions = []

flagged_frames = []
shirtless_ambiguous = []

frame_no = 0
stop_early = False

pbar = tqdm(total=max(1, total // max(1, interval)))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_no % interval != 0:
        frame_no += 1
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    frames.append(preprocess(pil))
    frame_ids.append(frame_no)
    captions.append(get_caption(pil))

    if len(frames) == BATCH:

        with torch.no_grad():
            img_batch = torch.stack(frames).to(device)
            logits, _ = clip_model(img_batch, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()

        for i in range(len(frames)):

            caption = captions[i]
            prob = probs[i]
            fid = frame_ids[i]

            conf = dict(zip(prompts, prob))
            detected = []

            for key in TH:
                if conf[key] > TH[key]:
                    detected.append((key, conf[key]))

            # Context filtering
            if any(s in caption for s in sign_ctx):
                detected = [x for x in detected if x[0] not in
                            ["nudity", "pornographic content", "explicit content"]]

            if any(s in caption for s in food_ctx):
                detected = [x for x in detected if x[0] not in
                            ["graphic violence", "blood or gore"]]

            if any(s in caption for s in medical_ctx):
                detected = [x for x in detected if x[0] not in
                            ["nudity", "explicit content"]]

            # Shirtless logic
            is_shirtless = any(k in caption for k in shirtless_kw)
            is_safe_ctx = any(k in caption for k in safe_shirtless_ctx)

            if is_shirtless and not is_safe_ctx:
                shirtless_ambiguous.append((fid, caption))

            if detected:
                detected.sort(key=lambda x: x[1], reverse=True)
                flagged_frames.append({
                    "frame": fid,
                    "issues": detected,
                    "caption": caption
                })

                if detected[0][1] > 0.90:
                    stop_early = True
                    break

        frames, frame_ids, captions = [], [], []

        if stop_early:
            break

    frame_no += 1
    pbar.update(1)

cap.release()
pbar.close()


# -------------------------
# VISUAL RESULT
# -------------------------
print("\n==============================")

if flagged_frames or shirtless_ambiguous:

    print("🚫 VIDEO VISUALLY FLAGGED AS UNSAFE\n")

    if flagged_frames:
        top = sorted(
            flagged_frames,
            key=lambda x: x["issues"][0][1],
            reverse=True
        )[0]

        t = int(top["frame"] / fps) if fps > 0 else 0

        print("Main reason :", top["issues"][0][0])
        print("Confidence  :", round(top["issues"][0][1], 2))
        print("Time        :", t, "sec")
        print("BLIP reason :", top["caption"])

    visual_safe = False

else:
    print("✅ VIDEO VISUALLY SAFE")
    visual_safe = True

print("==============================")


# -------------------------
# AUDIO ANALYSIS (NO WAV FILE SAVED)
# -------------------------
flagged_words = []

if visual_safe:

    print("\nRunning audio analysis...")

    clipv = VideoFileClip(video_path)

    if clipv.audio is not None:

        audio_array = clipv.audio.to_soundarray(fps=16000)
        audio_array = audio_array.astype("float32")

        result = whisper_model.transcribe(
            audio_array,
            fp16=torch.cuda.is_available(),
            temperature=0,
            beam_size=1,
            best_of=1
        )

        transcript = result["text"].lower()

        print("\nFull Transcript:")
        print(transcript)

        bad_patterns = [
            r"\bfuck\b", r"\bshit\b", r"\bbitch\b", r"\basshole\b",
            r"\bsex\b", r"\bnude\b", r"\bnaked\b", r"\bporn\b",
            r"\bkill\b", r"\bmurder\b", r"\brape\b",
            r"\bgun\b", r"\bknife\b", r"\bbomb\b",
            r"\bdrugs\b", r"\bcocaine\b", r"\bweed\b"
        ]

        for pattern in bad_patterns:
            if re.search(pattern, transcript):
                flagged_words.append(pattern.replace("\\b", ""))

        if flagged_words:
            print("\n🚫 Unsafe words detected:")
            for word in flagged_words:
                print(" -", word)

    clipv.close()


# -------------------------
# FINAL RESULT
# -------------------------
print("\n==============================")

if (not visual_safe) or flagged_words:

    print("🚫 FINAL RESULT : UNSAFE")

    if flagged_words:
        print("\nAudio reasons:")
        for word in flagged_words:
            print(" -", word)

else:
    print("✅ FINAL RESULT : SAFE")

print("==============================")