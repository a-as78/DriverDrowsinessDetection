# Driver Drowsiness Detection

## Introduction
Drowsiness detection is critical in environments where constant alertness is necessary, such as for drivers or machinery operators. The inspiration for this project came from an online video by Volkswagen showcasing their new Emergency Assist feature, which emphasized the importance of real-time drowsiness detection for driver safety. This led to a CNN-based approach that classifies whether a person is drowsy or not by analyzing facial features from images.

## Dataset and Preprocessing
The **Driver Drowsiness Dataset (DDD)** was downloaded via Kaggle. The dataset contains face images labeled as:
- `0` - Drowsy
- `1` - Non-drowsy

### Data Splitting and Scaling
The dataset was split into:
- **Training:** 80%
- **Validation:** 15%
- **Test:** 5%

Images were resized to **224 × 224** and normalized between **[0, 1]**.

## Model Architecture and Training
The CNN model is built using TensorFlow Keras with three convolutional layers, max-pooling, and dense layers:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(256, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

The model was compiled using the **Adam optimizer** and trained for **five epochs** with a **batch size of 32**.

## Installation and Usage
To use this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/a-as78/DriverDrowsinessDetection.git
cd DriverDrowsinessDetection
```

Then, run the training script:

```bash
python train.py
```

## References
1. Volkswagen Emergency Assist Video
2. [Driver Drowsiness Dataset on Kaggle](https://www.kaggle.com/)

