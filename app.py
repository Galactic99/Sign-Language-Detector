import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mediapipe as mp
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from collections import Counter
import time  # Add this import


class SignLanguageRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Define gestures (alphabet only)
        self.gestures = {chr(i + 65): i for i in range(26)}  # A-Z

        # Initialize GPU memory growth to avoid memory issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU initialization error: {e}")

    def create_model(self):
        """Create and return the CNN model architecture"""
        model = models.Sequential([
            # Input Layer
            layers.Input(shape=(64, 64, 3)),

            # First Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Second Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Third Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Fourth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Residual Block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.gestures), activation='softmax')
        ])
        return model

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.hands.process(rgb_image)

        # Create a black canvas
        processed_image = np.zeros((64, 64, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            # Get hand bounding box
            h, w, _ = image.shape
            landmarks = results.multi_hand_landmarks[0].landmark
            x_coords = [int(lm.x * w) for lm in landmarks]
            y_coords = [int(lm.y * h) for lm in landmarks]

            # Add padding to bounding box
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop hand region
            hand_region = image[y_min:y_max, x_min:x_max]

            if hand_region.size > 0:
                # Resize to target size
                processed_image = cv2.resize(hand_region, (64, 64))

                # Enhance contrast
                lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced = cv2.merge((cl, a, b))
                processed_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Normalize pixel values
        processed_image = processed_image / 255.0
        return processed_image

    def train_model(self, train_data_path, batch_size=32, epochs=50):
        """Train the sign language recognition model"""
        print("Loading and preprocessing training data...")
        X = []
        y = []

        # Load and preprocess training data
        for gesture, label in self.gestures.items():
            gesture_path = os.path.join(train_data_path, gesture)
            if os.path.exists(gesture_path):
                image_files = os.listdir(gesture_path)
                print(f"Processing {len(image_files)} images for gesture {gesture}")

                for image_name in image_files:
                    image_path = os.path.join(gesture_path, image_name)
                    image = cv2.imread(image_path)
                    if image is not None:
                        processed_image = self.preprocess_image(image)
                        X.append(processed_image)
                        y.append(label)

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        print(f"Total samples: {len(X)}")

        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Data augmentation
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )

        # Create and compile model
        print("Creating and compiling model...")
        model = self.create_model()
        model.compile(optimizer=optimizers.AdamW(learning_rate=1e-4),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Define callbacks
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

        # Train model
        print("Training model...")
        history = model.fit(data_gen.flow(X_train, y_train, batch_size=batch_size),
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            callbacks=[early_stop, lr_scheduler],
                            verbose=1)

        return model, history

    def real_time_prediction(self, model):
        """Perform real-time sign language recognition"""
        print("Starting real-time recognition...")
        print("Press 'q' to quit")
        print("Press 'c' to confirm the prediction")

        cap = cv2.VideoCapture(0)
        gesture_buffer = []  # Buffer for smoothing predictions
        confirmed_gesture = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break

            # Process frame
            processed_frame = self.preprocess_image(frame)
            prediction = model.predict(np.expand_dims(processed_frame, axis=0), verbose=0)

            # Get prediction and confidence
            predicted_idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            # Smooth predictions using a buffer
            gesture_buffer.append(predicted_idx)
            if len(gesture_buffer) > 5:
                gesture_buffer.pop(0)

            # Get most common prediction from buffer
            most_common_prediction = Counter(gesture_buffer).most_common(1)[0][0]

            # Get gesture name
            predicted_gesture = [k for k, v in self.gestures.items() if v == most_common_prediction][0]

            # Draw hand landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

            # Display results
            cv2.putText(frame, f'Sign: {predicted_gesture}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Confidence: {confidence:.2f}%', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if confirmed_gesture:
                cv2.putText(frame, f'Confirmed: {confirmed_gesture}', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw ROI rectangle
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

            cv2.imshow('Sign Language Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                confirmed_gesture = predicted_gesture
                print(f"Gesture '{confirmed_gesture}' confirmed")

                # Save the frame to the corresponding folder
                gesture_folder = os.path.join("data", confirmed_gesture)
                os.makedirs(gesture_folder, exist_ok=True)
                image_path = os.path.join(gesture_folder, f"{confirmed_gesture}_{int(time.time())}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Image saved to {image_path}")

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the sign language recognition system"""
    # Create necessary directories
    os.makedirs("models", exist_ok=True)

    # Initialize recognizer
    recognizer = SignLanguageRecognizer()

    # Set paths
    train_data_path = "data"
    model_save_path = "models/sign_language_model.keras"

    # Check if model exists
    if os.path.exists(model_save_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_save_path)
        print("Model loaded successfully!")
    else:
        print("Training new model...")
        try:
            # Train model
            model, history = recognizer.train_model(train_data_path)

            # Save model
            print("Saving model...")
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")

            # Plot training history
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.savefig('models/training_history.png')
            print("Training history plot saved to models/training_history.png")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            return

    # Start real-time recognition
    recognizer.real_time_prediction(model)


if __name__ == "__main__":
    main()
