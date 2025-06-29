# PRODIGY_ML_04
develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video data,enabling intuitive human- computer interaction and gesture based control systems.
# Hand Gesture Recognition Model Results

## Model Information
- **Generated**: 20250628_130944
- **Architecture**: Advanced CNN with Residual Connections
- **Input Shape**: [128, 128, 1]
- **Number of Classes**: 10
- **Total Parameters**: 2,919,946

## Performance Summary
- **Test Accuracy**: 0.8590
- **Best Validation Accuracy**: 1.0000
- **Training Epochs**: 23

## Gesture Classes
- 0: 01_palm
- 1: 02_l
- 2: 03_fist
- 3: 04_fist_moved
- 4: 05_thumb
- 5: 06_index
- 6: 07_ok
- 7: 08_palm_moved
- 8: 09_c
- 9: 10_down

## Output Files Description

### Model Files
- `hand_gesture_recognition_model.h5` - Keras model in HDF5 format
- `hand_gesture_recognition_model.keras` - Keras model in native format  
- `gesture_model_savedmodel/` - TensorFlow SavedModel format (recommended for deployment)
- `label_encoder.pkl` - Label encoder for converting between class names and indices

### Performance Analysis
- `model_results_summary_20250628_130944.json` - Overall model performance metrics
- `classification_report_20250628_130944.json` - Detailed per-class precision, recall, F1-score
- `per_class_performance_20250628_130944.csv` - Detailed per-class analysis with confidence scores
- `confusion_matrix_20250628_130944.csv` - Confusion matrix in CSV format

### Training Data
- `training_history_20250628_130944.csv` - Training and validation metrics for each epoch
- `training_plots_20250628_130944.png` - Visualization of training progress

### Predictions
- `test_predictions_20250628_130944.csv` - Individual test predictions with:
  - True and predicted labels
  - Confidence scores
  - Probability distributions for all classes
  - Correctness indicators

### Visualizations
- `confusion_matrix_20250628_130944.png` - Confusion matrix heatmap
- `training_plots_20250628_130944.png` - Training history plots

## Usage Instructions

### Loading the Model
```python
import tensorflow as tf
import joblib

# Load model (choose one format)
model = tf.keras.models.load_model('hand_gesture_recognition_model.h5')  # H5 format
# OR
model = tf.keras.models.load_model('hand_gesture_recognition_model.keras')  # Native Keras format
# OR load SavedModel format
model = tf.saved_model.load('gesture_model_savedmodel')

# Load label encoder
label_encoder = joblib.load('label_encoder.pkl')
```

### Making Predictions
```python
import cv2
import numpy as np

# Load and preprocess image
img = cv2.imread('your_image.png', cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (128, 128))
img_normalized = img_resized.astype(np.float32) / 255.0
img_input = img_normalized.reshape(1, 128, 128, 1)

# Predict
prediction = model.predict(img_input)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)
gesture_name = label_encoder.classes_[predicted_class]

print(f"Predicted gesture: {gesture_name} (confidence: {confidence:.3f})")
```

## Model Architecture
See `model_architecture_20250628_130944.txt` for detailed layer information.

## Notes
- Model was trained with subject-independent split for better generalization
- Data augmentation was applied during training
- Early stopping and learning rate reduction were used for optimal performance# Prodigy_ML_04
