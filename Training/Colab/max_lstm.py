import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import Policy, set_global_policy
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from tensorflow.keras.layers import SimpleRNN, Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization

# Enable mixed precision for faster training on compatible hardware
set_global_policy(Policy('mixed_float16'))

# Configuration
DATA_DIR = '/content/drive/MyDrive/Lab/Sequenced_full/'
SEQUENCE_LENGTH = 10
BATCH_SIZE = 256
EPOCHS = 200
MODEL_SAVE_PATH = 'lstm_model.h5'
OVERLAP = SEQUENCE_LENGTH
NUM_WORKERS = os.cpu_count()

def load_and_label_data(data_dir):
    """Load all CSV files and extract labels from filenames"""

    def load_file(filename):
      """Helper function to load a single file"""
      filepath = os.path.join(DATA_DIR, filename)
      df = pd.read_csv(filepath, usecols=['XA', 'YA', 'ZA', 'XG', 'YG', 'ZG', 'action_id'])
      x = df[['XA', 'YA', 'ZA', 'XG', 'YG', 'ZG']].values
      y = df['action_id'].iloc[0]
      return x, y

    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    sequences = []
    labels = []

    # Use parallel processing to load files
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(load_file, file_list))

    # Process loaded data
    for sensor_data, action_number in results:
        # Split into sequences using numpy for better performance
        num_sequences = (len(sensor_data) - SEQUENCE_LENGTH) // OVERLAP + 1
        for i in range(0, num_sequences * OVERLAP, OVERLAP):
            sequence = sensor_data[i:i + SEQUENCE_LENGTH]
            sequences.append(sequence)
            labels.append(action_number)

    # Convert to numpy arrays in one operation
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels)

def create_model(input_shape, num_classes):
    """Create optimized LSTM model architecture"""
    inputs = Input(shape=input_shape)

    x = SimpleRNN(64, return_sequences=True, activation='tanh')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # 2. LSTM Layer(s)
    x = LSTM(256, input_shape=input_shape, return_sequences=True, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LSTM(256, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # 3. Output
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs, outputs)

    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def evaluate_model(model, history, X_test, y_test, label_encoder):
    """Comprehensive model evaluation with combined plots"""
    plt.figure(figsize=(20, 16))

    # Convert one-hot encoded labels back to class indices
    y_true = y_test.argmax(axis=1)
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE*2).argmax(axis=1)

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[str(cls) for cls in label_encoder.classes_]
    ))

    # Plot 1: Confusion matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)

    # Calculate per-class accuracy
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

    # Plot 2: Per-class accuracy
    plt.subplot(2, 2, 2)
    plt.bar(range(len(class_accuracy)), list(class_accuracy.values()), align='center')
    plt.xticks(range(len(class_accuracy)), list(class_accuracy.keys()), rotation=45, ha='right')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14)

    # Plot 3: Training accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(fontsize=12)

    # Plot 4: Training loss
    plt.subplot(2, 2, 4)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('Evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()



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

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )

    # Create model
    input_shape = (SEQUENCE_LENGTH, X_train.shape[2])
    model = create_model(input_shape, num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=100, mode='max',
                     restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True,
                       monitor='val_accuracy', mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                         min_lr=1e-6, mode='max', verbose=1)
    ]

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE*2, verbose=0)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    evaluate_model(model, history, X_test, y_test, label_encoder)

    # Save label encoder for future use
    np.save('label_encoder_classes.npy', label_encoder.classes_)

if __name__ == "__main__":
    main()