<h1 align="center" style="color:#FF5733;">🎬 Multi-Modal Video Emotion Analysis Pipeline 🧩</h1>

<p align="center">
<img src="https://img.icons8.com/color/96/video.png" width="50" /> 
<img src="https://img.icons8.com/color/96/pose.png" width="50" /> 
<img src="https://img.icons8.com/color/96/robot-emoji.png" width="50" />
<img src="https://img.icons8.com/color/96/ai.png" width="50" />
</p>

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🌈 Overview

This pipeline combines **YOLO Pose**, **Facial Expression Analysis**, **Dataset Metrics**, and **LLM-based descriptions** to analyze emotions in videos.

<img src="https://img.icons8.com/color/48/emotions.png" width="40" /> **Key Models Used**:
- **YOLOv8 Pose** → Body landmarks (movement, posture, interactions)
- **Facial Expression Model** → Detects emotions from faces
- **Valence & Arousal Dataset Metrics** → Quantitative context
- **LLM (TinyLlama)** → Generates textual descriptions

---

## 📌 Pipeline Steps

### <span style="color:#1E90FF;">1️⃣ Explore Dataset 🔍</span>
<img src="https://img.icons8.com/color/48/folder-invoices.png" width="25"/>
- Inspect `videos/` and `rating_averaged/`  
- Check video resolution, FPS, duration  
- Preview valence/arousal metrics  

### <span style="color:#FF8C00;">2️⃣ Match Videos with Ratings 🔗</span>
<img src="https://img.icons8.com/color/48/link.png" width="25"/>
- Match videos with rating files  
- Store video-rating pairs  

### <span style="color:#32CD32;">3️⃣ Extract Pose & Facial Features 🧍‍♂️😃</span>
<img src="https://img.icons8.com/color/48/body.png" width="25"/>
- **YOLOv8 Pose**: Body keypoints  
- **Facial Expression Model**: Detect emotions  
- Sample frames, merge with dataset metrics  
- Save features as CSVs  

### <span style="color:#FF1493;">4️⃣ Combine All Video Features 🔄</span>
<img src="https://img.icons8.com/color/48/merge.png" width="25"/>
- Merge per-video CSVs  
- Add video IDs  

### <span style="color:#00CED1;">5️⃣ Normalize and Label Emotions ⚖️</span>
<img src="https://img.icons8.com/color/48/scale.png" width="25"/>
- Normalize valence & arousal  
- Map to emotion categories:  
  - 😃 Happy/Excited  
  - 😌 Calm/Content  
  - 😠😨 Angry/Fearful  
  - 😔😴 Sad/Tired  

### <span style="color:#8A2BE2;">6️⃣ Summarize Video Behavior 📊</span>
<img src="https://img.icons8.com/color/48/chart.png" width="25"/>
- Average valence, arousal, movement intensity  
- Quantitative summary for LLM  

### <span style="color:#FF4500;">7️⃣ Generate Natural-Language Descriptions 📝</span>
<img src="https://img.icons8.com/color/48/text.png" width="25"/>
- LLM input: pose, facial expressions, metrics  
- Output: concise text per frame/video  

### <span style="color:#228B22;">8️⃣ Multi-Frame Emotional Storytelling 🎬</span>
<img src="https://img.icons8.com/color/48/story.png" width="25"/>
- Sample 5–8 frames per video  
- Show pose, facial, metrics, LLM description  
- Capture temporal trends  

### <span style="color:#DAA520;">9️⃣ Full Video Narrative 🧩</span>
<img src="https://img.icons8.com/color/48/book.png" width="25"/>
- Combine frame-level descriptions  
- Example: *"The person starts tense and alert, gradually becomes calmer, ends relaxed and content."*  

### <span style="color:#FF6347;">🔟 Visualization 🖼️</span>
<img src="https://img.icons8.com/color/48/picture.png" width="25"/>
- Annotated frames: pose + facial + descriptions  
- Include full narrative as figure text  

### <span style="color:#00BFFF;">1️⃣1️⃣ Save & Deploy Results 💾</span>
<img src="https://img.icons8.com/color/48/save-as.png" width="25"/>
- Save CSVs / Pickle files / Plots  
- Ready for dashboards & analysis  

---

## 🛠️ Requirements

```bash
pip install ultralytics transformers torch torchvision opencv-python pillow pandas numpy matplotlib
