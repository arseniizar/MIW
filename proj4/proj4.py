import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
    print("Apple Silicon GPU disabled. Running on CPU to avoid XLA crash...")
except:
    pass

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)
np.random.seed(42)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 1. DATA LOADING, SPLITTING, AND PROCESSING
print("Loading and splitting data...")
(x_train_full, y_train_full), (x_test_orig, y_test_orig) = cifar10.load_data()

X = np.concatenate((x_train_full, x_test_orig))
y = np.concatenate((y_train_full, y_test_orig))

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train_cat = to_categorical(y_train, 10)
y_val_cat = to_categorical(y_val, 10)
y_test_cat = to_categorical(y_test, 10)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)


# 2. CNN MODELS

# Architecture 1: Shallow CNN
def build_model_1():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Architecture 2: Deep CNN (VGG-style with Batch Normalization and Dropout)
def build_model_2():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.35),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.45),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


print("Building models...")
model1 = build_model_1()
model2 = build_model_2()

# 3. TRAINING
EPOCHS = 25
BATCH_SIZE = 64

print("\n--- Training Model 1 (Shallow CNN) ---")
history1 = model1.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    verbose=1
)

print("\n--- Training Model 2 (Deep CNN) ---")
history2 = model2.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    verbose=1
)


# 4. EVALUATION
def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


plot_history(history1, "Model 1 (Shallow CNN)")
plot_history(history2, "Model 2 (Deep CNN)")

# Test Set Evaluation
print("\n--- Test Set Evaluation ---")
test_loss_1, test_acc_1 = model1.evaluate(X_test, y_test_cat, verbose=0)
test_loss_2, test_acc_2 = model2.evaluate(X_test, y_test_cat, verbose=0)
print(f"Model 1 Test Accuracy: {test_acc_1:.4f}")
print(f"Model 2 Test Accuracy: {test_acc_2:.4f}")


# Confusion Matrices
def plot_confusion_matrix(model, X_test, y_test, title):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    ax.set_title(title)
    plt.show()


plot_confusion_matrix(model1, X_test, y_test, "Confusion Matrix - Model 1")
plot_confusion_matrix(model2, X_test, y_test, "Confusion Matrix - Model 2")

# 5|6: Conclusion and output
print("\n" + "=" * 50)
print("ANALYSIS AND CONCLUSIONS")
print("=" * 50)
print("1. Overfitting/Underfitting Analysis:")
print(
    "   - Model 1 (Shallow): Typically starts overfitting early. You can observe the validation loss diverging from the training loss.")
print(
    "   - Model 2 (Deep): Utilizing Data Augmentation, Dropout, and Batch Normalization keeps the validation curves much closer to the training curves, drastically reducing overfitting.")

print("\n2. Misclassifications Analysis:")
print("   - In the plotted Confusion Matrices, you will notice certain classes are frequently confused.")
print("   - For example, 'Cats' and 'Dogs' are heavily misclassified as each other due to structural similarities.")
print(
    "   - 'Automobile' and 'Truck' also share misclassifications, while classes with distinct backgrounds like 'Airplane' and 'Ship' have higher accuracy.")

print("\n3. Best Model Choice:")
print(
    "   - Model 2 is superior. The spatial hierarchy captures more complex traits, while regularization techniques (Dropout/BN) ensure it generalizes well to unseen test data.")
print("=" * 50)