import os
import warnings
import cv2
import mediapipe as mp
import pickle
import numpy as np

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all logs except for errors
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Mediapipe objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load the model and label encoder
model_dict = pickle.load(open(os.path.join('Data and Model', 'model.p'), 'rb'))
model = model_dict['model']
label_encoder = model_dict['label_encoder']  # Load the LabelEncoder

# For webcam input
cap = cv2.VideoCapture(0)

# Ensure the target length is the same as what the model was trained on (84 in this case)
target_length = 84

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        # Pad the data_aux array to match the expected feature length (84)
        if len(data_aux) < target_length:
            data_aux.extend([0] * (target_length - len(data_aux)))

        # Predict
        prediction = model.predict([np.asarray(data_aux)])

        # Decode the prediction to get the corresponding character
        predicted_character = label_encoder.inverse_transform([prediction[0]])[0]

        # Draw the prediction on the frame
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(5)

    # Break the loop if the window is closed
    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
