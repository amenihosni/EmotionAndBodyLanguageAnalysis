import os
import uuid
import json
import gc
import logging
import torch
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# -----------------------------
# 🧠 Matplotlib low-memory setup
# -----------------------------
import matplotlib
matplotlib.use("Agg")  # headless backend
from matplotlib import pyplot as plt
try:
    from matplotlib import _image  # preload DLL
except ImportError:
    pass

# Limit cache
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "mplcache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# -----------------------------
# ⚙️ Flask setup
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Interview-analysis'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# -----------------------------
# 🧰 Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# 🧩 Analyzer
# -----------------------------
analyzer = None
model_metrics_info = {}

def init_analyzer():
    global analyzer
    try:
        from video_analyzer import VideoAnalyzer
        if os.path.exists("video_analyzer_light.pkl"):
            logger.info("✅ Loading analyzer from video_analyzer_light.pkl")
            analyzer = VideoAnalyzer.load_lightweight("video_analyzer_light.pkl")
            return True
        else:
            logger.warning("❌ video_analyzer_light.pkl not found, using mock mode")
            return False
    except Exception as e:
        logger.error(f"Analyzer init failed: {e}")
        return False

# -----------------------------
# 🔧 Utilities
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_video_size(video_path, max_mb=100):
    try:
        size = os.path.getsize(video_path) / (1024 * 1024)
        if size > max_mb:
            logger.warning(f"⚠️ Video too large: {size:.1f}MB")
            return False
        return True
    except:
        return True

# -----------------------------
# 🎥 Analysis
# -----------------------------
def perform_analysis(video_path, analysis_id):
    try:
        if analyzer and hasattr(analyzer, "analyze_video"):
            if not check_video_size(video_path):
                raise Exception("Video too large")
            
            logger.info("🔍 Starting real analysis...")
            results = analyzer.analyze_video(video_path)
            return format_real_results(results, analysis_id)
        else:
            raise Exception("Analyzer unavailable")
    except Exception as e:
        logger.warning(f"Real analysis failed: {e}")
        return generate_mock_analysis(video_path, analysis_id)

def format_real_results(results, analysis_id):
    try:
        # Extract the actual metrics from your model
        emotion_metrics = results.get('emotion_metrics', {})
        body_metrics = results.get('body_language_metrics', {})
        desc = results.get('comprehensive_description', '')
        timeline = results.get('timeline_analysis', [])
        all_emotions = results.get('all_emotions', [])
        
        logger.info(f"🎭 Processing emotion metrics: {emotion_metrics}")
        logger.info(f"💃 Processing body metrics: {list(body_metrics.keys())}")

        # -----------------------------
        # Compute highlights from ACTUAL metrics
        # -----------------------------
        # Calculate confidence from engaged vs neutral
        engaged_score = emotion_metrics.get('engaged', 50)
        neutral_score = emotion_metrics.get('neutral', 50)
        confidence_score = int((engaged_score + (100 - neutral_score)) / 2)
        
        # Calculate engagement from body language metrics
        engagement_scores = body_metrics.get('engagement_scores', [0])
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
        engagement_score = int(avg_engagement * 100)
        
        # Calculate nervousness from movement and posture
        movement_scores = body_metrics.get('movement_intensity', [0])
        avg_movement = sum(movement_scores) / len(movement_scores) if movement_scores else 0
        posture_scores = body_metrics.get('posture_asymmetry', [0])
        avg_posture = sum(posture_scores) / len(posture_scores) if posture_scores else 0
        
        # Higher movement + poor posture = more nervousness
        nervousness_score = min(100, int((avg_movement * 2) + (avg_posture / 2)))

        # -----------------------------
        # Format emotions - using ACTUAL emotion metrics
        # -----------------------------
        emotions = []
        
        # Process engaged metric
        if 'engaged' in emotion_metrics:
            engaged_value = emotion_metrics['engaged']
            emotions.append({
                'name': 'ENGAGEMENT LEVEL',
                'confidence': int(engaged_value),
                'level': 'high' if engaged_value > 70 else 'medium' if engaged_value > 40 else 'low',
                'icon': 'fas fa-handshake',
                'description': f'Shows {"high" if engaged_value > 70 else "moderate" if engaged_value > 40 else "low"} engagement'
            })
        
        # Process neutral metric  
        if 'neutral' in emotion_metrics:
            neutral_value = emotion_metrics['neutral']
            emotions.append({
                'name': 'EMOTIONAL EXPRESSIVITY',
                'confidence': int(100 - neutral_value),  # Invert - lower neutral = more expressive
                'level': 'high' if neutral_value < 30 else 'medium' if neutral_value < 60 else 'low',
                'icon': 'fas fa-comments',
                'description': f'{"Highly expressive" if neutral_value < 30 else "Moderately expressive" if neutral_value < 60 else "Mostly neutral"} expression'
            })
        
        # Add calculated confidence metric
        emotions.append({
            'name': 'CONFIDENCE LEVEL',
            'confidence': confidence_score,
            'level': 'high' if confidence_score > 70 else 'medium' if confidence_score > 40 else 'low',
            'icon': 'fas fa-user-shield',
            'description': f'Demonstrates {"high" if confidence_score > 70 else "moderate" if confidence_score > 40 else "low"} confidence'
        })

        # -----------------------------
        # Format body language - using ACTUAL body metrics
        # -----------------------------
        body_language = []
        
        # Movement Intensity
        if 'movement_intensity' in body_metrics:
            movement_values = body_metrics['movement_intensity']
            avg_movement = sum(movement_values) / len(movement_values) if movement_values else 0
            movement_pct = min(100, int(avg_movement * 10))  # Scale 0-10 to 0-100
            body_language.append({
                'name': 'MOVEMENT INTENSITY',
                'confidence': movement_pct,
                'level': 'high' if movement_pct > 70 else 'medium' if movement_pct > 40 else 'low',
                'icon': 'fas fa-running',
                'description': f'{"High" if movement_pct > 70 else "Moderate" if movement_pct > 40 else "Low"} physical movement'
            })
        
        # Engagement Scores
        if 'engagement_scores' in body_metrics:
            engagement_values = body_metrics['engagement_scores']
            avg_engagement = sum(engagement_values) / len(engagement_values) if engagement_values else 0
            engagement_pct = int(avg_engagement * 100)
            body_language.append({
                'name': 'BODY ENGAGEMENT',
                'confidence': engagement_pct,
                'level': 'high' if engagement_pct > 70 else 'medium' if engagement_pct > 40 else 'low',
                'icon': 'fas fa-users',
                'description': f'Body language shows {"high" if engagement_pct > 70 else "moderate" if engagement_pct > 40 else "low"} engagement'
            })
        
        # Openness Scores
        if 'openness_scores' in body_metrics:
            openness_values = body_metrics['openness_scores']
            avg_openness = sum(openness_values) / len(openness_values) if openness_values else 0
            openness_pct = int(avg_openness * 100)
            body_language.append({
                'name': 'BODY OPENNESS',
                'confidence': openness_pct,
                'level': 'high' if openness_pct > 70 else 'medium' if openness_pct > 40 else 'low',
                'icon': 'fas fa-arms',
                'description': f'{"Open" if openness_pct > 70 else "Moderately open" if openness_pct > 40 else "Closed"} body posture'
            })
        
        # Posture Asymmetry (inverted - lower is better)
        if 'posture_asymmetry' in body_metrics:
            posture_values = body_metrics['posture_asymmetry']
            avg_posture = sum(posture_values) / len(posture_values) if posture_values else 0
            posture_pct = max(0, 100 - int(avg_posture))  # Invert - lower asymmetry = better
            body_language.append({
                'name': 'POSTURE STABILITY',
                'confidence': posture_pct,
                'level': 'high' if posture_pct > 70 else 'medium' if posture_pct > 40 else 'low',
                'icon': 'fas fa-user',
                'description': f'Posture is {"stable" if posture_pct > 70 else "moderately stable" if posture_pct > 40 else "asymmetric"}'
            })
        
        # Activity Levels
        if 'activity_levels' in body_metrics:
            activity_values = body_metrics['activity_levels']
            avg_activity = sum(activity_values) / len(activity_values) if activity_values else 0
            activity_pct = min(100, int(avg_activity * 20))  # Scale 0-5 to 0-100
            body_language.append({
                'name': 'ACTIVITY LEVEL',
                'confidence': activity_pct,
                'level': 'high' if activity_pct > 70 else 'medium' if activity_pct > 40 else 'low',
                'icon': 'fas fa-tachometer-alt',
                'description': f'{"High" if activity_pct > 70 else "Moderate" if activity_pct > 40 else "Low"} activity level'
            })

        # -----------------------------
        # Generate insights from ACTUAL data
        # -----------------------------
        insights = []
        
        # Insight 1: Overall engagement
        if engagement_score > 70:
            insights.append("High engagement detected throughout the interview with consistent focus.")
        elif engagement_score > 40:
            insights.append("Moderate engagement with some variation in attention levels.")
        else:
            insights.append("Lower engagement observed - may indicate distraction or discomfort.")
        
        # Insight 2: Movement analysis
        avg_movement_val = sum(body_metrics.get('movement_intensity', [0])) / len(body_metrics.get('movement_intensity', [1]))
        if avg_movement_val > 7:
            insights.append("Energetic movement patterns suggest enthusiasm or potential nervous energy.")
        elif avg_movement_val > 4:
            insights.append("Balanced movement indicates comfortable communication style.")
        else:
            insights.append("Minimal movement shows calm and controlled demeanor.")
        
        # Insight 3: Emotional expression
        if emotion_metrics.get('neutral', 0) > 70:
            insights.append("Predominantly neutral expression maintains professional composure.")
        elif emotion_metrics.get('engaged', 0) > 60:
            insights.append("Active emotional engagement demonstrates interest in conversation.")
        
        # Insight 4: Posture analysis
        avg_posture_val = sum(body_metrics.get('posture_asymmetry', [0])) / len(body_metrics.get('posture_asymmetry', [1]))
        if avg_posture_val < 10:
            insights.append("Excellent posture alignment indicates confidence and attention.")
        elif avg_posture_val < 20:
            insights.append("Generally good posture with minor variations.")
        else:
            insights.append("Posture adjustments observed - may indicate comfort-seeking.")
        
        # Add comprehensive description if available
        if desc and len(insights) < 4:
            insight_parts = [p.strip() + '.' for p in desc.split('.') if p.strip()]
            insights.extend(insight_parts[:4-len(insights)])
        
        # Ensure we have at least 3 insights
        while len(insights) < 3:
            insights.append("Analysis completed with comprehensive behavioral assessment.")

        return {
            'video_name': results.get('video_name', os.path.basename(results.get('video_path', ''))),
            'emotions': emotions,
            'body_language': body_language,
            'insights': insights[:4],  # Max 4 insights
            'highlights': {
                'confidence': confidence_score,
                'engagement': engagement_score, 
                'nervousness': min(100, nervousness_score)
            },
            'comprehensive_description': desc,
            'timeline_analysis': timeline,
            'analysis_id': analysis_id,
            'analysis_type': 'REAL_ANALYSIS',
            'raw_metrics': {  # Include raw metrics for transparency
                'emotion_metrics': emotion_metrics,
                'body_metrics_summary': {k: f"{sum(v)/len(v):.2f}" if v else "0" for k, v in body_metrics.items()}
            }
        }

    except Exception as e:
        logger.error(f"Error formatting real results: {e}")
        return generate_mock_analysis("unknown", analysis_id)

def generate_mock_analysis(video_path, analysis_id):
    """Fallback mock analysis when real analysis fails"""
    return {
        'video_name': os.path.basename(video_path),
        'emotions': [
            {'name': 'CONFIDENCE LEVEL', 'confidence': 78, 'level': 'high', 'icon': 'fas fa-user-shield'},
            {'name': 'ENGAGEMENT', 'confidence': 82, 'level': 'high', 'icon': 'fas fa-handshake'}
        ],
        'body_language': [
            {'name': 'POSTURE STABILITY', 'confidence': 72, 'level': 'high', 'icon': 'fas fa-user'},
            {'name': 'BODY OPENNESS', 'confidence': 68, 'level': 'medium', 'icon': 'fas fa-arms'}
        ],
        'insights': [
            "Mock analysis used - real model not available.",
            "This is sample data for demonstration.",
            "Check if video_analyzer_light.pkl is properly configured."
        ],
        'highlights': {'confidence': 78, 'engagement': 82, 'nervousness': 34},
        'analysis_id': analysis_id,
        'analysis_type': 'MOCK_ANALYSIS'
    }

# -----------------------------
# 🌐 Flask Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html') if os.path.exists('templates/index.html') else """
    <h3>Interview Analysis</h3>
    <p>Upload your interview videos to analyze emotions and body language.</p>
    <p><strong>Real Model Active:</strong> Using engaged/neutral emotions and body language metrics</p>
    """

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error':'No file provided'}),400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':'No file selected'}),400
    if file and allowed_file(file.filename):
        analysis_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{analysis_id}_{filename}")
        file.save(filepath)
        return jsonify({'success':True,'analysis_id':analysis_id,'filename':filename})
    return jsonify({'error':'Invalid file'}),400

@app.route('/analyze', methods=['POST'])
def analyze_video():
    data = request.get_json()
    analysis_id = data.get('analysis_id')
    filename = data.get('filename')
    if not analysis_id or not filename:
        return jsonify({'error':'Missing data'}),400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{analysis_id}_{filename}")
    if not os.path.exists(filepath):
        return jsonify({'error':'File not found'}),404

    results = perform_analysis(filepath, analysis_id)

    # Save results
    results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}_results.json")
    with open(results_file,'w') as f:
        json.dump(results,f,indent=2)

    # Cleanup
    try: os.remove(filepath)
    except: pass
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return jsonify({'success':True,'analysis_id':analysis_id,'results':results})

@app.route('/health')
def health_check():
    return jsonify({
        'status':'healthy',
        'analyzer_ready': analyzer is not None and hasattr(analyzer, "analyze_video"),
        'mode':'REAL_ANALYSIS' if analyzer and hasattr(analyzer, "analyze_video") else 'MOCK_ANALYSIS',
        'model_file':'video_analyzer_light.pkl' if os.path.exists("video_analyzer_light.pkl") else 'NOT_FOUND',
        'supported_metrics': {
            'emotions': ['engaged', 'neutral'],
            'body_language': ['movement_intensity', 'engagement_scores', 'activity_levels', 'openness_scores', 'posture_asymmetry']
        }
    })

# -----------------------------
# 🚀 Run server
# -----------------------------
if __name__=="__main__":
    if init_analyzer(): 
        print("✅ Analyzer ready (REAL mode)")
        print("📊 Supported Metrics:")
        print("   Emotions: engaged, neutral")
        print("   Body Language: movement_intensity, engagement_scores, activity_levels, openness_scores, posture_asymmetry")
    else: 
        print("⚠️ Mock mode active")

    # Low RAM server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)