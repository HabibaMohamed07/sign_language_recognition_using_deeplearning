import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands(by google) module (hand landmark estimation)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory containing image data
DATA_DIR = './data'

# Initialize lists to store hand landmark data and corresponding labels
data = []
labels = []

# Iterate over each directory in the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Check if it's a directory
        # Iterate over each image file in the directory
        for img_path in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_path)
            # Check if the file is a valid image file (png, jpg, jpeg)
            if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                data_aux = []  # Temporary list to store hand landmark data for each image
                x_ = []       # Temporary lists to store x and y coordinates for normalization

                # Read and process the image using MediaPipe Hands
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Obtain hand landmarks from the image
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract x and y coordinates of each hand landmark
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            data_aux.append(x)
                            y_.append(y)
                            data_aux.append(y)

                    # Normalize hand landmark data by subtracting the minimum x and y values
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux[i * 2] -= min(x_)
                        data_aux[i * 2 + 1] -= min(y_)

                    # Append the normalized hand landmark data to the main data list
                    data.append(data_aux)
                    # Append the corresponding label (class) to the labels list
                    labels.append(dir_)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
