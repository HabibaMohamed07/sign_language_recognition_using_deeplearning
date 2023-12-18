import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class SignLanguageRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']

        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L', 11:'M', 12:'N', 13:'O', 14:'O', 15:'Q',
                           16:'R', 17:'S', 18:'T', 19:'U', 20:'U', 21:'V', 22:'W', 23:'X' , 24:'DrAlaa'}

        self.sign_images = {
            'Q': 'DrAlaa.png',  
        }

        self.init_ui()

    def init_ui(self):
        self.start_button = QPushButton('Start Camera', self)
        self.stop_button = QPushButton('Stop Camera', self)
        self.letter_label = QLabel('Recognized Letter: ', self)
        self.image_label = QLabel(self)

        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.letter_label)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.show()

    def start_camera(self):
        self.timer.start(1)

    def stop_camera(self):
        self.timer.stop()

    def update_frame(self):
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = self.cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Padding
            max_length = 84  # Change this to the actual length used during training
            data_aux_padded = pad_sequences([data_aux], maxlen=max_length, padding='post', dtype='float32')[0]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = self.model.predict([np.asarray(data_aux_padded)])

            predicted_character = self.labels_dict[int(prediction[0])]

            self.letter_label.setText(f'Recognized Letter: Dr Alaa ❤️')

            # Display image for the recognized sign
            if predicted_character in self.sign_images:
                image_path = self.sign_images[predicted_character]
                pixmap = QPixmap(image_path)
                self.image_label.setPixmap(pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
            else:
                self.image_label.clear()

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

def main():
    app = QApplication(sys.argv)
    window = SignLanguageRecognitionApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
