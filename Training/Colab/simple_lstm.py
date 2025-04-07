import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = '/content/drive/MyDrive/Classified_S/'
SEQUENCE_LENGTH = 10
BATCH_SIZE = 512
EPOCHS = 100
MODEL_SAVE_PATH = 'cattle_movement_lstm.h5'
OVERLAP = SEQUENCE_LENGTH

def load_and_label_data(data_dir):
    """Load all CSV files and extract labels from filenames"""
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    sequences = []
    labels = []

    for filename in file_list:
        # Parse label from filename (format: {cattle_number}_{action_number}_{fragment_number}.csv)
        parts = filename.split('_')
        action_number = int(parts[1])

        # Load data
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)

        # Extract relevant columns (assuming they're in order: XA, YA, ZA, XG, YG, ZG)
        sensor_data = df[['XA', 'YA', 'ZA', 'XG', 'YG', 'ZG']].values

        # Split into sequences
        for i in range(0, len(sensor_data) - SEQUENCE_LENGTH + 1, OVERLAP):
            sequence = sensor_data[i:i + SEQUENCE_LENGTH]
            sequences.append(sequence)
            labels.append(action_number)

    return np.array(sequences), np.array(labels)

def create_model(input_shape, num_classes):
    """Create LSTM model architecture"""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():
    # Load and prepare data
    print("Loading data...")
    X, y = load_and_label_data(DATA_DIR)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    num_classes = len(label_encoder.classes_)

    print(f"Loaded {len(X)} sequences with {num_classes} classes")
    print("Class distribution:", np.unique(y, return_counts=True))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )

    # Create model
    input_shape = (SEQUENCE_LENGTH, X_train.shape[2])
    model = create_model(input_shape, num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')
    ]

    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    from sklearn.metrics import classification_report
    # Convert one-hot encoded labels back to class indices
    y_true = y_test.argmax(axis=1)
    y_pred = model.predict(X_test).argmax(axis=1)

    # Generate classification report
    class_report = classification_report(
        y_true, 
        y_pred, 
        target_names=[str(cls) for cls in label_encoder.classes_]
    )
    print("Classification Report:")
    print(class_report)

    # Calculate and display per-class accuracy
    class_accuracy = {}
    for class_label in label_encoder.classes_:
        class_idx = np.where(label_encoder.classes_ == class_label)[0][0]
        mask = y_true == class_idx
        class_correct = (y_true[mask] == y_pred[mask]).sum()
        class_total = mask.sum()
        class_accuracy[class_label] = class_correct / class_total

    print("\nPer-Class Accuracy:")
    for class_label, acc in class_accuracy.items():
        print(f"Class {class_label}: {acc:.2%}")

    # Optional: Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_accuracy)), list(class_accuracy.values()), align='center')
    plt.xticks(range(len(class_accuracy)), list(class_accuracy.keys()), rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.tight_layout()
    plt.show()

    # Save label encoder for future use
    np.save('label_encoder_classes.npy', label_encoder.classes_)

if __name__ == "__main__":
    main()