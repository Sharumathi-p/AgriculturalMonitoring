from flask import Flask, render_template_string, Response
import cv2
import mediapipe as mp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def is_help_signal(hand_landmarks):
    """
    Detects the 'Signal for Help' gesture.
    - A fist with the thumb tucked inside.
    """
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    
    # Check if all fingers (except thumb) are folded
    fingers_folded = (
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )
    
    # Check if thumb is inside the fist
    thumb_tucked = thumb_tip.y > index_mcp.y

    return fingers_folded and thumb_tucked

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        logger.error("Failed to open camera.")
        return

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                logger.error("Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)  # Flip horizontally for natural webcam effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            danger_detected = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if is_help_signal(hand_landmarks):
                        danger_detected = True

                        # Compute bounding box
                        h, w, _ = frame.shape
                        xs = [lm.x for lm in hand_landmarks.landmark]
                        ys = [lm.y for lm in hand_landmarks.landmark]
                        min_x, max_x = int(min(xs) * w), int(max(xs) * w)
                        min_y, max_y = int(min(ys) * h), int(max(ys) * h)

                        # Draw red rectangle
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show "DANGER" in the top-right if detected
            if danger_detected:
                h, w, _ = frame.shape
                text = "DANGER"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                cv2.rectangle(frame, (w - text_width - 20, 10), (w - 10, 10 + text_height + baseline), (0, 0, 255), -1)
                cv2.putText(frame, text, (w - text_width - 10, 10 + text_height), font, font_scale, (255, 255, 255), thickness)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error("Failed to encode frame.")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
      <head>
        <title>Signal for Help Detection</title>
      </head>
      <body>
        <h1>Camera Feed - "Signal for Help" Detection</h1>
        <p>Shows "DANGER" if a closed fist with thumb inside is detected.</p>
        <img src="{{ url_for('video_feed') }}" style="width:700px; height:450px;" />
      </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
