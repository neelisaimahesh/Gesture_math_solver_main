# Gesture_math_solver_main
🚀 What is a Gesture-Based Math Solver?
A Gesture-Based Math Solver is an interactive Python app that:

Tracks your hand movements
Recognizes how many fingers you show
Converts those into numbers or operations
Evaluates the expression in real-time
All using just your webcam + hands + Python magic!

🎯 What You Will Learn
✅ Real-time webcam input using OpenCV
✅ Hand gesture recognition using MediaPipe
✅ Finger counting logic with Python
✅ Building and evaluating expressions
✅ Gesture commands like “delete”, “clear”, “exit”

🧰 Tech Stack Used
Tool	Purpose
Python	Main programming language
OpenCV	Capture and display webcam input
MediaPipe	AI-powered hand tracking (21 keypoints/hand)
NumPy	Math calculations (like finger distance)
eval()	Python’s built-in method to evaluate math
📷 How it Works – Behind the Scenes
Step-by-Step Process:
Start Webcam
OpenCV captures your video feed in real time.
Detect Hands
MediaPipe finds and tracks your hand landmarks (21 points per hand).
Count Fingers
Logic checks which fingers are up or down to detect:
Numbers (0–9)
Operators (+, −, ×, ÷)
Commands (=, delete, clear, exit)
Build Math Expression
Each gesture adds a part to the expression.
Evaluate Expression
Show the “equal” gesture to solve it.
🧠 Hand Gesture Reference Guide
Gesture (Fingers)	Action	Meaning
0 fingers (1 hand)	Add 0	Digit
1 to 5 fingers	Add 1 to 5	Digit
5 + 1 fingers	Add 6	Digit
5 + 2 fingers	Add 7	Digit
5 + 3 fingers	Add 8	Digit
5 + 4 fingers	Add 9	Digit
1 + 1 fingers (2 hands)	Add +	Addition
1 + 2 fingers	Add −	Subtraction
1 + 3 fingers	Add ×	Multiplication
1 + 4 fingers	Add ÷	Division
0 + 0 fingers	=	Evaluate
2 + 2 fingers	del	Delete last
5 + 5 fingers	clear	Clear input
Index finger close	Exit	Quit app
🔨 Step-by-Step Guide to Build the Project
1. Create a Project Folder
mkdir GestureMathSolver
cd GestureMathSolver
python -m venv venv
🧠 2. Activate Virtual Environment
.\venv\Scripts\activate   # For Windows
📦 3. Install Required Libraries
pip install opencv-python mediapipe numpy
📄 4. Create a Python File
Save as gesture_math_solver.py

Complete Python Code: gesture_math_solver.py
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
# MediaPipe modules for hand detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
# Function to calculate Euclidean distance between two hand landmarks
def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
# Function to count number of fingers raised
def count_fingers(hand_landmarks, label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    # Thumb logic
    if label == "Left":
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    # Other four fingers
    for i in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i]-2].y else 0)
    return fingers.count(1)
# Function to detect gestures using two hands
def detectGesture(hand1_data, hand2_data):
    (hand1, label1), (hand2, label2) = hand1_data, hand2_data
    f1 = count_fingers(hand1, label1)
    f2 = count_fingers(hand2, label2)
    dist = euclidean_distance(hand1.landmark[8], hand2.landmark[8])
    if f1 == 1 and f2 == 1:
        if dist < 0.06:
            return "exit"
        return "+"
    elif (f1 == 1 and f2 == 2) or (f1 == 2 and f2 == 1):
        return "-"
    elif (f1 == 1 and f2 == 3) or (f1 == 3 and f2 == 1):
        return "*"
    elif (f1 == 1 and f2 == 4) or (f1 == 4 and f2 == 1):
        return "/"
    elif f1 == 2 and f2 == 2:
        return "del"
    elif (f1 == 1 and f2 == 5) or (f1 == 5 and f2 == 1):
        return "6"
    elif (f1 == 2 and f2 == 5) or (f1 == 5 and f2 == 2):
        return "7"
    elif (f1 == 3 and f2 == 5) or (f1 == 5 and f2 == 3):
        return "8"
    elif (f1 == 4 and f2 == 5) or (f1 == 5 and f2 == 4):
        return "9"
    elif f1 == 0 and f2 == 0:
        return "="
    elif f1 == 5 and f2 == 5:
        return "clear"
    return None
# Initialize variables
last_update_time = 0
delay = 1.25
expression = ""
res = ""
# Open webcam
cap = cv.VideoCapture(0)
while True:
    success, image = cap.read()
    if not success:
        continue
    image = cv.flip(image, 1)
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    current_time = time.time()
    hand_data = []
    # Detect hands and count fingers
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = hand_handedness.classification[0].label
            hand_data.append((hand_landmarks, label))
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Single hand input for digits 0–5
        if len(hand_data) == 1:
            hand_landmarks, label = hand_data[0]
            fingers_up = count_fingers(hand_landmarks, label)
            if fingers_up in [0, 1, 2, 3, 4, 5] and current_time - last_update_time > delay:
                expression += str(fingers_up)
                last_update_time = current_time
        # Two-hand gestures for operations/commands
        if len(hand_data) == 2:
            gesture = detectGesture(hand_data[0], hand_data[1])
            if gesture == "exit":
                break
            elif gesture == "clear":
                expression = ""
                res = ""
                last_update_time = current_time
            elif gesture == "del" and current_time - last_update_time > delay:
                expression = expression[:-1]
                last_update_time = current_time
            elif gesture == "=" and current_time - last_update_time > delay:
                try:
                    res = str(eval(expression))
                except:
                    res = "Error"
                last_update_time = current_time
            elif gesture and current_time - last_update_time > delay:
                expression += gesture
                last_update_time = current_time
    # Display on screen
    cv.putText(image, f'Expr: {expression}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.putText(image, f'Result: {res}', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv.imshow("Gesture Math Solver", image)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('c'):
        expression = ""
        res = ""
# Cleanup
cap.release()
cv.destroyAllWindows()
💡 Explanation of Key Modules
🧠 OpenCV
Captures video from the webcam
Flips the frame (mirror effect)
Displays text like Expression: and Result:
✋ MediaPipe
Detects your hand
Tracks finger tips and joints in real-time
Provides landmark points for every finger
📏 NumPy
Used to calculate distance between index fingers
Helps detect gestures like exit (when two index fingers are close)
🧮 Python eval()
Takes your expression like 2+3*4
Returns result: 14
📌 Example: Real-Life Use Case
You do the following:

✌️ Show 2 fingers → Adds 2
☝️✌️ Show 1 + 2 fingers → Adds -
🤞 Show 3 fingers → Adds 3
👊👊 Closed fists → Triggers =
👉 Output:

Expression: 2-3
Result: -1
✅ Final Touch: Run the App
python gesture_math_solver.py
