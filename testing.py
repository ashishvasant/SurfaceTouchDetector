import cv2
import mediapipe as mp
import joblib
import numpy as np

def extract_features(landmarks):
    index_finger_tip = landmarks[8]  # MediaPipe landmark index for index fingertip
    index_finger_base = landmarks[5]  # MediaPipe landmark index for index finger base
    
    relative_x = index_finger_tip.x - index_finger_base.x
    relative_y = index_finger_tip.y - index_finger_base.y
    relative_z = index_finger_tip.z - index_finger_base.z
    
    return [relative_x, relative_y, relative_z]

def realtime_prediction():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    
    model = joblib.load('finger_detection_model.joblib')
    scaler = joblib.load('finger_detection_scaler.joblib')
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                features = extract_features(hand_landmarks.landmark)
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)
                
                result = "Touching" if prediction[0] == 1 else "Not touching"
                cv2.putText(image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time prediction
realtime_prediction()