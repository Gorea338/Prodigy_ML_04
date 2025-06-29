import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from collections import Counter
import mediapipe as mp
import joblib
from PIL import Image
import warnings
import gc
import time
import threading
from queue import Queue
import logging
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Configure TensorFlow for stability
tf.config.run_functions_eagerly(False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU optimization with error handling
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU acceleration enabled with {len(gpus)} GPU(s)")
    else:
        print("⚠️ No GPU found, using CPU")
except Exception as e:
    print(f"⚠️ GPU setup failed: {e}")

# ================================
# 1. ENHANCED DATASET LOADER WITH DEBUGGING
# ================================

class EnhancedDataLoader:
    def __init__(self, dataset_path, target_size=(64, 64), max_samples_per_class=500):
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size
        self.max_samples_per_class = max_samples_per_class
        self.gesture_classes = [
            'palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 
            'ok', 'palm_moved', 'c', 'down'
        ]
        
    def load_and_preprocess_image(self, img_path):
        """Enhanced image loading with better error handling"""
        try:
            # Use OpenCV for fastest loading
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Failed to load image: {img_path}")
                return None
            
            # Fast resize with nearest neighbor
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Normalize to 0-1 range
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logging.warning(f"Failed to load {img_path}: {e}")
            return None
    
    def load_dataset_optimized(self):
        """Enhanced dataset loading with detailed progress tracking"""
        images = []
        labels = []
        
        print("Loading dataset with balanced sampling...")
        
        total_files = 0
        loaded_files = 0
        
        for subject_dir in sorted(self.dataset_path.glob('*')):
            if not subject_dir.is_dir():
                continue
                
            subject_id = subject_dir.name
            print(f"Processing subject: {subject_id}")
            
            for gesture_dir in sorted(subject_dir.glob('*')):
                if not gesture_dir.is_dir():
                    continue
                    
                gesture_name = gesture_dir.name.split('_', 1)[-1]
                if gesture_name not in self.gesture_classes:
                    print(f"⚠️ Skipping unknown gesture: {gesture_name}")
                    continue
                
                # Get all image files
                image_files = list(gesture_dir.glob('*.png'))
                total_files += len(image_files)
                
                print(f"  Found {len(image_files)} images for gesture '{gesture_name}'")
                
                # Limit samples per class for balanced dataset
                if len(image_files) > self.max_samples_per_class:
                    image_files = np.random.choice(
                        image_files, 
                        self.max_samples_per_class, 
                        replace=False
                    )
                
                # Process images
                for img_path in image_files:
                    img = self.load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(gesture_name)
                        loaded_files += 1
        
        print(f"✅ Loaded {loaded_files}/{total_files} images from {len(set(labels))} classes")
        
        if len(images) == 0:
            raise ValueError("No images were loaded! Check your dataset path and file formats.")
        
        return np.array(images), np.array(labels)

# ================================
# 2. ENHANCED FEATURE EXTRACTOR WITH DEBUGGING
# ================================

class EnhancedFeatureExtractor:
    def __init__(self):
        self.mediapipe_available = False
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.3,
                model_complexity=0
            )
            self.feature_dim = 42  # 21 landmarks * 2 (x,y only)
            self.mediapipe_available = True
            print("✅ MediaPipe initialized successfully")
        except Exception as e:
            print(f"⚠️ MediaPipe initialization failed: {e}")
            self.hands = None
            self.feature_dim = 42
    
    def extract_features_batch(self, images):
        """Enhanced batch feature extraction with detailed logging"""
        if not self.mediapipe_available:
            print("⚠️ MediaPipe not available, returning dummy features")
            return np.random.random((len(images), self.feature_dim)).astype(np.float32)
        
        features = []
        successful_extractions = 0
        
        print(f"Extracting features from {len(images)} images...")
        
        for i, img in enumerate(images):
            try:
                # Convert grayscale to RGB
                if len(img.shape) == 2:
                    rgb_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    rgb_img = (img * 255).astype(np.uint8)
                
                results = self.hands.process(rgb_img)
                
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]
                    feature_vector = []
                    for landmark in landmarks.landmark:
                        feature_vector.extend([landmark.x, landmark.y])
                    successful_extractions += 1
                else:
                    feature_vector = [0.0] * self.feature_dim
                
                features.append(feature_vector)
                
            except Exception as e:
                print(f"⚠️ Feature extraction failed for image {i}: {e}")
                features.append([0.0] * self.feature_dim)
                
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} images")
        
        print(f"✅ Successfully extracted features from {successful_extractions}/{len(images)} images")
        return np.array(features, dtype=np.float32)

# ================================
# 3. ENHANCED MODELS WITH BETTER ARCHITECTURE
# ================================

def create_enhanced_cnn(input_shape, num_classes):
    """Enhanced CNN with better architecture"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Enhanced convolutional layers
        layers.Conv2D(16, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Enhanced dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_enhanced_feature_model(input_dim, num_classes):
    """Enhanced feature model"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# ================================
# 4. ENHANCED TRAINER WITH VISUALIZATION
# ================================

class EnhancedTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.loader = EnhancedDataLoader(dataset_path)
        self.feature_extractor = EnhancedFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.training_history = {}
        
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Training History', fontsize=16)
        
        # Accuracy plot
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Final metrics
        final_acc = history.history['val_accuracy'][-1]
        final_loss = history.history['val_loss'][-1]
        axes[1, 1].text(0.1, 0.8, f'Final Validation Accuracy: {final_acc:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Final Validation Loss: {final_loss:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Total Epochs: {len(history.history["loss"])}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_class_distribution(self, labels):
        """Plot class distribution"""
        class_counts = Counter(labels)
        
        plt.figure(figsize=(12, 6))
        gestures = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = plt.bar(gestures, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Gesture Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_data_safe(self):
        """Enhanced data preparation with visualization"""
        try:
            # Load dataset
            images, labels = self.loader.load_dataset_optimized()
            
            if len(images) == 0:
                raise ValueError("No images loaded!")
            
            # Plot class distribution
            self.plot_class_distribution(labels)
            
            # Show class distribution
            print("\nClass Distribution:")
            class_counts = Counter(labels)
            for gesture, count in sorted(class_counts.items()):
                print(f"{gesture}: {count} images")
            
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(labels)
            num_classes = len(self.label_encoder.classes_)
            
            # Add channel dimension
            images = np.expand_dims(images, axis=-1)
            
            # Extract features
            print("Extracting features...")
            features = self.feature_extractor.extract_features_batch(images.squeeze())
            
            print(f"✅ Data preparation complete: {len(images)} samples, {num_classes} classes")
            return images, features, encoded_labels, num_classes
            
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            raise
    
    def train_models_safe(self):
        """Enhanced training with comprehensive visualization"""
        try:
            # Prepare data
            images, features, labels, num_classes = self.prepare_data_safe()
            
            # Split data
            X_img_train, X_img_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
                images, features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Convert to categorical
            y_train_cat = keras.utils.to_categorical(y_train, num_classes)
            y_test_cat = keras.utils.to_categorical(y_test, num_classes)
            
            print(f"\nTraining shapes - Images: {X_img_train.shape}, Features: {X_feat_train.shape}")
            
            # Train CNN model
            print("\n" + "="*50)
            print("TRAINING ENHANCED CNN MODEL")
            print("="*50)
            
            cnn_model = create_enhanced_cnn(X_img_train.shape[1:], num_classes)
            cnn_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("CNN Model Summary:")
            cnn_model.summary()
            
            # Enhanced callbacks
            cnn_callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    'best_cnn_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train CNN
            cnn_history = cnn_model.fit(
                X_img_train, y_train_cat,
                epochs=50,
                batch_size=32,
                validation_data=(X_img_test, y_test_cat),
                callbacks=cnn_callbacks,
                verbose=1
            )
            
            # Plot CNN training history
            self.plot_training_history(cnn_history, 'CNN')
            
            # Train feature model
            print("\n" + "="*50)
            print("TRAINING ENHANCED FEATURE MODEL")
            print("="*50)
            
            feature_model = create_enhanced_feature_model(X_feat_train.shape[1], num_classes)
            feature_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.002),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Feature Model Summary:")
            feature_model.summary()
            
            feature_callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    'best_feature_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            feature_history = feature_model.fit(
                X_feat_train, y_train_cat,
                epochs=100,
                batch_size=64,
                validation_data=(X_feat_test, y_test_cat),
                callbacks=feature_callbacks,
                verbose=1
            )
            
            # Plot feature model training history
            self.plot_training_history(feature_history, 'Feature')
            
            # Comprehensive evaluation
            self.evaluate_models_comprehensive(cnn_model, feature_model, 
                                             X_img_test, X_feat_test, y_test)
            
            # Save models
            cnn_model.save('enhanced_cnn_model.h5')
            feature_model.save('enhanced_feature_model.h5')
            joblib.dump(self.label_encoder, 'enhanced_label_encoder.pkl')
            joblib.dump(self.feature_extractor, 'enhanced_feature_extractor.pkl')
            print("✅ Enhanced models saved successfully")
            
            # Clean up memory
            del X_img_train, X_img_test, X_feat_train, X_feat_test
            gc.collect()
            
            return cnn_model, feature_model
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def evaluate_models_comprehensive(self, cnn_model, feature_model, X_img_test, X_feat_test, y_test):
        """Comprehensive model evaluation with detailed metrics"""
        print("\n" + "="*50)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*50)
        
        # Evaluate CNN
        print("\n📊 CNN Model Evaluation:")
        cnn_pred = cnn_model.predict(X_img_test, batch_size=32, verbose=0)
        cnn_pred_classes = np.argmax(cnn_pred, axis=1)
        cnn_accuracy = accuracy_score(y_test, cnn_pred_classes)
        
        print(f"CNN Accuracy: {cnn_accuracy:.4f}")
        print("\nCNN Classification Report:")
        print(classification_report(y_test, cnn_pred_classes, 
                                  target_names=self.label_encoder.classes_))
        
        # Plot CNN confusion matrix
        cnn_cm = self.plot_confusion_matrix(y_test, cnn_pred_classes, 'CNN')
        
        # Evaluate Feature model
        print("\n📊 Feature Model Evaluation:")
        feat_pred = feature_model.predict(X_feat_test, batch_size=64, verbose=0)
        feat_pred_classes = np.argmax(feat_pred, axis=1)
        feat_accuracy = accuracy_score(y_test, feat_pred_classes)
        
        print(f"Feature Model Accuracy: {feat_accuracy:.4f}")
        print("\nFeature Model Classification Report:")
        print(classification_report(y_test, feat_pred_classes, 
                                  target_names=self.label_encoder.classes_))
        
        # Plot Feature model confusion matrix
        feat_cm = self.plot_confusion_matrix(y_test, feat_pred_classes, 'Feature')
        
        # Ensemble evaluation
        print("\n📊 Ensemble Model Evaluation:")
        ensemble_pred = 0.6 * cnn_pred + 0.4 * feat_pred
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred_classes)
        
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, ensemble_pred_classes, 
                                  target_names=self.label_encoder.classes_))
        
        # Plot Ensemble confusion matrix
        ensemble_cm = self.plot_confusion_matrix(y_test, ensemble_pred_classes, 'Ensemble')
        
        # Summary comparison
        print("\n📈 MODEL COMPARISON SUMMARY:")
        print(f"CNN Model:      {cnn_accuracy:.4f}")
        print(f"Feature Model:  {feat_accuracy:.4f}")
        print(f"Ensemble Model: {ensemble_accuracy:.4f}")
        
        # Create comparison plot
        models = ['CNN', 'Feature', 'Ensemble']
        accuracies = [cnn_accuracy, feat_accuracy, ensemble_accuracy]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'])
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# ================================
# 5. ENHANCED REAL-TIME RECOGNIZER WITH DEBUGGING
# ================================

class EnhancedGestureRecognizer:
    def __init__(self, cnn_model_path, feature_model_path, label_encoder_path, feature_extractor_path):
        self.models_loaded = False
        self.debug_mode = True
        
        print("🔧 Initializing Enhanced Gesture Recognizer...")
        
        try:
            # Load models with error handling
            print("Loading CNN model...")
            self.cnn_model = keras.models.load_model(cnn_model_path)
            print("✅ CNN model loaded")
            
            print("Loading Feature model...")
            self.feature_model = keras.models.load_model(feature_model_path)
            print("✅ Feature model loaded")
            
            print("Loading label encoder...")
            self.label_encoder = joblib.load(label_encoder_path)
            print("✅ Label encoder loaded")
            
            print("Loading feature extractor...")
            self.feature_extractor = joblib.load(feature_extractor_path)
            print("✅ Feature extractor loaded")
            
            self.models_loaded = True
            print("✅ All models loaded successfully")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.models_loaded = False
            return
        
        # Initialize MediaPipe for real-time
        try:
            print("Initializing MediaPipe...")
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.3,
                model_complexity=0
            )
            self.mediapipe_ready = True
            print("✅ MediaPipe initialized")
        except Exception as e:
            print(f"⚠️ MediaPipe initialization failed: {e}")
            self.mediapipe_ready = False
        
        # Pre-allocate arrays
        self.input_size = (64, 64)
        self.prediction_history = []
        
    def test_camera_access(self):
        """Test camera access before starting recognition"""
        print("🎥 Testing camera access...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not accessible!")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera!")
            cap.release()
            return False
        
        print(f"✅ Camera accessible - Frame shape: {frame.shape}")
        cap.release()
        return True
    
    def predict_gesture_enhanced(self, frame):
        """Enhanced gesture prediction with detailed debugging"""
        if not self.models_loaded:
            return "Models not loaded", 0.0, False, {}
        
        debug_info = {}
        
        try:
            # Preprocessing
            start_time = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_NEAREST)
            normalized = resized.astype(np.float32) / 255.0
            cnn_input = normalized.reshape(1, *self.input_size, 1)
            
            debug_info['preprocess_time'] = time.time() - start_time
            
            # CNN prediction
            start_time = time.time()
            cnn_pred = self.cnn_model.predict(cnn_input, verbose=0)
            debug_info['cnn_time'] = time.time() - start_time
            debug_info['cnn_confidence'] = float(np.max(cnn_pred[0]))
            
            # Feature extraction and prediction
            hand_detected = False
            if self.mediapipe_ready:
                try:
                    start_time = time.time()
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)
                    
                    if results.multi_hand_landmarks:
                        landmarks = results.multi_hand_landmarks[0]
                        feature_vector = []
                        for landmark in landmarks.landmark:
                            feature_vector.extend([landmark.x, landmark.y])
                        feature_input = np.array([feature_vector], dtype=np.float32)
                        
                        feat_pred = self.feature_model.predict(feature_input, verbose=0)
                        ensemble_pred = 0.6 * cnn_pred + 0.4 * feat_pred
                        
                        hand_detected = True
                        debug_info['feature_time'] = time.time() - start_time
                        debug_info['feature_confidence'] = float(np.max(feat_pred[0]))
                    else:
                        ensemble_pred = cnn_pred
                        debug_info['feature_time'] = time.time() - start_time
                        debug_info['feature_confidence'] = 0.0
                        
                except Exception as e:
                    if self.debug_mode:
                        print(f"⚠️ Feature extraction error: {e}")
                    ensemble_pred = cnn_pred
                    debug_info['feature_error'] = str(e)
            else:
                ensemble_pred = cnn_pred
            
            # Final prediction
            predicted_class = np.argmax(ensemble_pred[0])
            confidence = ensemble_pred[0][predicted_class]
            gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
            
            debug_info['ensemble_confidence'] = float(confidence)
            debug_info['predicted_class'] = int(predicted_class)
            debug_info['gesture_name'] = gesture_name
            
            return gesture_name, float(confidence), hand_detected, debug_info
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Prediction error: {e}")
            return "Error", 0.0, False, {'error': str(e)}
    
    def start_enhanced_recognition(self):
        """Enhanced real-time recognition with comprehensive debugging"""
        # Test camera first
        if not self.test_camera_access():
            return
        
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Cannot open camera")
                return
            
            # Optimize camera settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("🎥 Enhanced real-time recognition started")
            print("Controls:")
            print("  'q' - Quit")
            print("  'd' - Toggle debug mode")
            print("  'r' - Reset prediction history")
            print("  's' - Save current frame")
            
            frame_count = 0
            gesture_name, confidence = "None", 0.0
            fps_counter = time.time()
            fps = 0
            debug_info = {}
            
            # Performance tracking
            prediction_times = []
            confidence_history = []
            
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠️ Failed to read frame")
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    frame_count += 1
                    
                    # Make prediction every 3 frames for better performance
                    if frame_count % 3 == 0:
                        start_pred_time = time.time()
                        gesture_name, confidence, hand_detected, debug_info = self.predict_gesture_enhanced(frame)
                        pred_time = time.time() - start_pred_time
                        prediction_times.append(pred_time)
                        confidence_history.append(confidence)
                        
                        # Keep only recent history
                        if len(prediction_times) > 100:
                            prediction_times.pop(0)
                            confidence_history.pop(0)
                    
                    # Draw main UI
                    frame_display = frame.copy()
                    
                    # Main prediction display
                    if confidence > 0.4:
                        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                        cv2.rectangle(frame_display, (10, 10), (450, 80), (0, 0, 0), -1)
                        cv2.putText(frame_display, f"GESTURE: {gesture_name.upper()}", 
                                  (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.putText(frame_display, f"Confidence: {confidence:.3f}", 
                                  (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        cv2.rectangle(frame_display, (10, 10), (450, 60), (0, 0, 0), -1)
                        cv2.putText(frame_display, "Show your hand gesture", 
                                  (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Hand detection indicator
                    if hand_detected:
                        cv2.circle(frame_display, (600, 30), 10, (0, 255, 0), -1)
                        cv2.putText(frame_display, "Hand", (570, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.circle(frame_display, (600, 30), 10, (0, 0, 255), -1)
                        cv2.putText(frame_display, "No Hand", (560, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # FPS counter
                    if frame_count % 30 == 0:
                        fps = 30 / (time.time() - fps_counter)
                        fps_counter = time.time()
                    
                    cv2.putText(frame_display, f"FPS: {fps:.1f}", 
                              (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Debug information
                    if self.debug_mode and debug_info:
                        y_offset = 100
                        cv2.rectangle(frame_display, (10, y_offset), (400, y_offset + 150), (0, 0, 0), -1)
                        cv2.putText(frame_display, "DEBUG INFO:", 
                                  (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        if 'cnn_confidence' in debug_info:
                            cv2.putText(frame_display, f"CNN Conf: {debug_info['cnn_confidence']:.3f}", 
                                      (20, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        if 'feature_confidence' in debug_info:
                            cv2.putText(frame_display, f"Feat Conf: {debug_info['feature_confidence']:.3f}", 
                                      (20, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        if prediction_times:
                            avg_pred_time = np.mean(prediction_times[-10:])
                            cv2.putText(frame_display, f"Pred Time: {avg_pred_time*1000:.1f}ms", 
                                      (20, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        if confidence_history:
                            avg_confidence = np.mean(confidence_history[-10:])
                            cv2.putText(frame_display, f"Avg Conf: {avg_confidence:.3f}", 
                                      (20, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        cv2.putText(frame_display, f"Frame: {frame_count}", 
                                  (20, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Status bar
                    status_text = f"Models: {'OK' if self.models_loaded else 'FAIL'} | "
                    status_text += f"MediaPipe: {'OK' if self.mediapipe_ready else 'FAIL'} | "
                    status_text += f"Debug: {'ON' if self.debug_mode else 'OFF'}"
                    
                    cv2.rectangle(frame_display, (0, frame_display.shape[0]-25), 
                                (frame_display.shape[1], frame_display.shape[0]), (50, 50, 50), -1)
                    cv2.putText(frame_display, status_text, 
                              (10, frame_display.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    cv2.imshow('Enhanced Gesture Recognition', frame_display)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('d'):
                        self.debug_mode = not self.debug_mode
                        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                    elif key == ord('r'):
                        gesture_name, confidence = "Reset", 0.0
                        prediction_times.clear()
                        confidence_history.clear()
                        print("Prediction history reset")
                    elif key == ord('s'):
                        filename = f"gesture_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Frame saved as {filename}")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"⚠️ Frame processing error: {e}")
                    continue
            
            # Show performance summary
            if prediction_times:
                print(f"\n📊 Performance Summary:")
                print(f"Average prediction time: {np.mean(prediction_times)*1000:.2f}ms")
                print(f"Max prediction time: {np.max(prediction_times)*1000:.2f}ms")
                print(f"Min prediction time: {np.min(prediction_times)*1000:.2f}ms")
                print(f"Average confidence: {np.mean(confidence_history):.3f}")
                print(f"Total frames processed: {frame_count}")
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera released safely")

# ================================
# 6. ENHANCED MAIN EXECUTION
# ================================

def main():
    """Enhanced main execution with comprehensive error handling and user interaction"""
    
    print("="*70)
    print("🚀 ENHANCED GESTURE RECOGNITION WITH DEBUGGING & VISUALIZATION")
    print("="*70)
    
    # Default dataset path - UPDATE THIS PATH!
    default_dataset_path = r"C:\Users\vikas\Downloads\archive (2)\leapGestRecog\leapGestRecog"
    
    # Ask user for dataset path
    print(f"\n📂 Current dataset path: {default_dataset_path}")
    user_path = input("Enter new path (or press Enter to use current): ").strip()
    
    if user_path:
        dataset_path = user_path
    else:
        dataset_path = default_dataset_path
    
    # Check dataset
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("\n📥 Please download the LeapGestRecog dataset from:")
        print("   https://www.kaggle.com/gti-upm/leapgestrecog")
        
        # Try alternative paths
        alternative_paths = [
            "./leapgestrecog",
            "./leapGestRecog", 
            "./dataset",
            "./data",
            "leapGestRecog"
        ]
        
        print(f"\n🔍 Searching for dataset in alternative locations...")
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                dataset_path = alt_path
                print(f"✅ Found dataset at: {dataset_path}")
                break
        else:
            print("❌ Dataset not found in any alternative location")
            print("Please ensure the dataset is downloaded and path is correct")
            return
    
    print(f"✅ Using dataset path: {dataset_path}")
    
    # Ask user what to do
    print("\n🎯 What would you like to do?")
    print("1. Train new models (with full visualization)")
    print("2. Load existing models and start recognition")
    print("3. Train models and then start recognition")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    models_exist = (os.path.exists('enhanced_cnn_model.h5') and 
                   os.path.exists('enhanced_feature_model.h5') and
                   os.path.exists('enhanced_label_encoder.pkl') and
                   os.path.exists('enhanced_feature_extractor.pkl'))
    
    cnn_model = None
    feature_model = None
    
    if choice in ['1', '3'] or not models_exist:
        # Training phase
        try:
            print("\n🚀 Starting enhanced training with visualization...")
            trainer = EnhancedTrainer(dataset_path)
            cnn_model, feature_model = trainer.train_models_safe()
            
            print("\n✅ Training completed successfully!")
            print("📁 Enhanced models saved:")
            print("   - enhanced_cnn_model.h5")
            print("   - enhanced_feature_model.h5") 
            print("   - enhanced_label_encoder.pkl")
            print("   - enhanced_feature_extractor.pkl")
            print("📊 Training visualizations saved:")
            print("   - cnn_training_history.png")
            print("   - feature_training_history.png")
            print("   - cnn_confusion_matrix.png")
            print("   - feature_confusion_matrix.png")
            print("   - ensemble_confusion_matrix.png")
            print("   - model_comparison.png")
            print("   - class_distribution.png")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            if not models_exist:
                print("No existing models found. Cannot proceed to recognition.")
                return
            print("Attempting to use existing models...")
    
    # Real-time recognition
    if choice in ['2', '3'] or models_exist:
        try:
            start_recognition = input("\n🎥 Start real-time recognition? (y/n): ").lower()
            if start_recognition == 'y':
                print("\n🔧 Initializing enhanced gesture recognizer...")
                recognizer = EnhancedGestureRecognizer(
                    'enhanced_cnn_model.h5',
                    'enhanced_feature_model.h5', 
                    'enhanced_label_encoder.pkl',
                    'enhanced_feature_extractor.pkl'
                )
                
                if recognizer.models_loaded:
                    print("✅ Models loaded successfully")
                    recognizer.start_enhanced_recognition()
                else:
                    print("❌ Failed to load models")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Real-time recognition error: {e}")
    
    print("\n🎉 Program completed successfully!")
    print("📊 Check the generated visualization files for detailed analysis")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("Program terminated safely")
        import traceback
        traceback.print_exc()
        