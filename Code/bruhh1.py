from flask import Flask, render_template_string, Response, url_for
import cv2
import mediapipe as mp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ----- Gesture Detection Functions -----
def is_help_signal(hand_landmarks):
    """HELP Signal: Closed fist with thumb tucked inside."""
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    fingers_folded = (
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )
    thumb_tucked = thumb_tip.y > index_mcp.y
    return fingers_folded and thumb_tucked

def is_stop_signal(hand_landmarks):
    """STOP Signal: Open palm with all four fingers extended."""
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    fingers_extended = (
        index_tip.y < index_mcp.y and
        middle_tip.y < middle_mcp.y and
        ring_tip.y < ring_mcp.y and
        pinky_tip.y < pinky_mcp.y
    )
    return fingers_extended

def is_victory_signal(hand_landmarks):
    """VICTORY Signal: V-sign (index & middle extended, others folded)."""
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y
    return index_extended and middle_extended and ring_folded and pinky_folded

def is_phone_signal(hand_landmarks):
    """PHONE Signal: Thumb and index extended (like a phone), other fingers folded."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    thumb_extended = thumb_tip.y < thumb_ip.y  # Naive check for thumb extension
    index_extended = index_tip.y < index_mcp.y
    middle_folded = middle_tip.y > middle_mcp.y
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y
    return thumb_extended and index_extended and middle_folded and ring_folded and pinky_folded

def is_point_signal(hand_landmarks):
    """POINT Signal: Only the index finger extended; others folded."""
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    index_extended = index_tip.y < index_mcp.y
    thumb_folded = thumb_tip.y > thumb_ip.y
    middle_folded = middle_tip.y > middle_mcp.y
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y
    return index_extended and thumb_folded and middle_folded and ring_folded and pinky_folded

# Map gesture names to detection functions and assign overlay colors (BGR)
GESTURE_FUNCTIONS = {
    "HELP": is_help_signal,
    "STOP": is_stop_signal,
    "VICTORY": is_victory_signal,
    "PHONE": is_phone_signal,
    "POINT": is_point_signal
}
SIGNAL_COLORS = {
    "HELP": (0, 0, 255),        # Red
    "STOP": (255, 0, 0),        # Blue
    "VICTORY": (0, 255, 0),     # Green
    "PHONE": (0, 165, 255),     # Orange
    "POINT": (255, 0, 255)      # Magenta
}

# ----- Video Streaming & Annotation -----
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera.")
        return

    with mp_hands.Hands(min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame.")
                break

            # Flip frame for mirror view and convert color space
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            detected_signal = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check for gestures in priority order (HELP > PHONE > POINT > STOP > VICTORY)
                    for gesture in ["HELP", "PHONE", "POINT", "STOP", "VICTORY"]:
                        if GESTURE_FUNCTIONS[gesture](hand_landmarks):
                            detected_signal = gesture
                            break

                    # Compute bounding box for the hand
                    h, w, _ = frame.shape
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    ys = [lm.y for lm in hand_landmarks.landmark]
                    min_x, max_x = int(min(xs) * w), int(max(xs) * w)
                    min_y, max_y = int(min(ys) * h), int(max(ys) * h)
                    if detected_signal:
                        color = SIGNAL_COLORS.get(detected_signal, (0, 255, 255))
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
                        cv2.putText(frame, detected_signal, (min_x, min_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame = cv2.resize(frame, (700, 450))
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error("Failed to encode frame.")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# ----- Flask Routes -----
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Advanced HTML with CSS styling, project requirements, and examples of supported signals
    html_content = '''
    <!DOCTYPE html>
    <html>
      <head>
        <title>Advanced CCTV Hand Signal Detection</title>
        <style>
          body {
            font-family: 'Arial', sans-serif;
            background-color: #f0f2f5;
            margin: 0;
            padding: 0;
            color: #333;
          }
          header {
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            text-align: center;
          }
          .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
          }
          .video-feed, .info {
            background: white;
            margin: 10px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
          }
          .video-feed {
            flex: 1 1 700px;
          }
          .info {
            flex: 1 1 300px;
            max-width: 300px;
          }
          h1, h2 {
            margin-top: 0;
          }
          ul {
            list-style-type: none;
            padding: 0;
          }
          li {
            margin-bottom: 10px;
          }
          .signal {
            font-weight: bold;
          }
          footer {
            text-align: center;
            padding: 10px;
            background-color: #ddd;
          }
        </style>
      </head>
      <body>
        <header>
          <h1>Advanced CCTV Hand Signal Detection</h1>
          <p>Real-time detection of danger signals via hand gestures.</p>
        </header>
        <div class="container">
          <div class="video-feed">
            <h2>Live Camera Feed</h2>
            <img src="{{ url_for('video_feed') }}" style="width:100%; height:auto;" />
          </div>
          <div class="info">
            <h2>Project Requirements</h2>
            <ul>
              <li><strong>Camera:</strong> CCTV or Laptop Webcam</li>
              <li><strong>Computer:</strong> Python 3.x</li>
              <li><strong>Dependencies:</strong> Flask, OpenCV, MediaPipe</li>
              <li><strong>Network:</strong> Reliable connectivity for remote monitoring</li>
            </ul>
            <h2>Supported Danger Signals</h2>
            <ul>
              <li><span class="signal">HELP:</span> Closed fist with thumb tucked inside.</li>
              <li><span class="signal">PHONE:</span> Thumb and index extended (phone gesture).</li>
              <li><span class="signal">POINT:</span> Only index finger extended.</li>
              <li><span class="signal">STOP:</span> Open palm with fingers extended.</li>
              <li><span class="signal">VICTORY:</span> V-sign with index and middle extended.</li>
            </ul>
          </div>
        </div>
        <footer>
          <p>&copy; 2025 Advanced Hand Signal Detection Project</p>
        </footer>
      </body>
    </html>
    '''
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
