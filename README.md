<h1 align="center" style="color:#FF5733;">🎬 Multi-Modal Video Emotion Analysis Pipeline 🧩</h1>

<p align="center">
🧍‍♂️ 👩‍🦰 🎭 🤖
</p>

---

## 🌈 Overview

This pipeline combines **YOLO Pose**, **Facial Expression Analysis**, **Dataset Metrics**, and **LLM-based descriptions** to analyze emotions in videos.

🧩 **Key Models Used**:

### <span style="color:#1E90FF;">🧍‍♂️ YOLOv8 Pose – Body Landmarks</span>
- Detects **human body landmarks**: 33 keypoints per frame (x, y, z, visibility)
- Used to analyze **movement intensity, posture, and group interactions**
  
![results (3)](https://github.com/user-attachments/assets/f9cc506c-ca48-4996-b0eb-8c939b28daf5)


### <span style="color:#FF8C00;">🎭 Facial Expression Model – Emotion Detection</span>
- Detects **emotional state from faces**: happy, sad, angry, surprised, etc.
- Enhances **accuracy of emotional analysis per person**
- GIF explaining its purpose:

  
  ![results (2)](https://github.com/user-attachments/assets/2076885e-b470-4baa-9dee-b170b09fda37)


### <span style="color:#32CD32;">📊 Dataset Metrics – Valence & Arousal</span>
- Quantitative **valence (positive/negative)** and **arousal (energy)** scores
- Provides **numerical context** for emotions


### <span style="color:#FF1493;">🤖 LLM (TinyLlama) – Textual Description</span>
- Converts **pose, facial expressions, and metrics** into **natural-language descriptions**
- Generates **concise emotional narratives**
![LLM GIF]([https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif](https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/aygalic/tiny-llama.png))
final Results:
<img width="2005" height="960" alt="télécharger (3)" src="https://github.com/user-attachments/assets/9f650549-7e0d-4f7d-a52c-3e0740068b11" />

---

## 📌 Pipeline Steps

### <span style="color:#1E90FF;">1️⃣ Explore Dataset 🔍</span>
- Preview video & ratings  
- Check resolution, FPS, missing values

### <span style="color:#FF8C00;">2️⃣ Match Videos with Ratings 🔗</span>
- Match video files with rating CSVs  
- Store video–rating pairs

### <span style="color:#32CD32;">3️⃣ Extract Pose & Facial Features 🧍‍♂️🎭</span>
- YOLO Pose → body landmarks  
- Facial Expression → detect emotions  
- Merge with dataset metrics → save CSV

### <span style="color:#FF1493;">4️⃣ Combine All Video Features 🔄</span>
- Merge per-video CSVs  
- Add video IDs

### <span style="color:#00CED1;">5️⃣ Normalize and Label Emotions ⚖️</span>
- Normalize valence/arousal 0–1  
- Map to categories (😃😌😠😔)

### <span style="color:#8A2BE2;">6️⃣ Summarize Video Behavior 📊</span>
- Average valence, arousal, movement intensity  
- Prepare for LLM

### <span style="color:#FF4500;">7️⃣ Generate LLM Descriptions 📝</span>
- Input: pose + facial + metrics  
- Output: natural-language emotion description

### <span style="color:#228B22;">8️⃣ Multi-Frame Emotional Storytelling 🎬</span>
- Sample frames → display landmarks, facial emotion, metrics, LLM description

### <span style="color:#DAA520;">9️⃣ Full Video Narrative 🧩</span>
- Combine frame-level descriptions into **smooth story**  

### <span style="color:#FF6347;">🔟 Visualization 🖼️</span>
- Annotated frames + full narrative

### <span style="color:#00BFFF;">1️⃣1️⃣ Save & Deploy Results 💾</span>
- Save CSVs / Pickle files / Plots

---

## 🛠️ Requirements

```bash
pip install ultralytics transformers torch torchvision opencv-python pillow pandas numpy matplotlib
