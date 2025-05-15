# train.py - Script for training the Deepfake Detection Model
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, LeakyReLU, BatchNormalization, Input
from sklearn.model_selection import train_test_split
import time

# Set parameters
IMG_SIZE = 128  # Image size
BATCH_SIZE = 32  # Batch size
EPOCHS = 20     # Total training epochs

# Set base path - update this to your folder path
base_path = 'deepfake_detection'
checkpoint_path = os.path.join(base_path, 'checkpoints')
results_path = os.path.join(base_path, 'results')

# Create necessary directories
os.makedirs(base_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# Define the paths to your real and fake image directories
MAIN_DIR = 'deepfake_detection\data\real_and_fake_face'
TRAINING_REAL_DIR = os.path.join(MAIN_DIR, 'training_real')
TRAINING_FAKE_DIR = os.path.join(MAIN_DIR, 'training_fake')

# Create validation and test directories if they don't exist
VALID_DIR = os.path.join(MAIN_DIR, 'validation')
TEST_DIR = os.path.join(MAIN_DIR, 'test')
VALID_REAL_DIR = os.path.join(VALID_DIR, 'real')
VALID_FAKE_DIR = os.path.join(VALID_DIR, 'fake')
TEST_REAL_DIR = os.path.join(TEST_DIR, 'real')
TEST_FAKE_DIR = os.path.join(TEST_DIR, 'fake')

for directory in [VALID_DIR, TEST_DIR, VALID_REAL_DIR, VALID_FAKE_DIR, TEST_REAL_DIR, TEST_FAKE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Function to split data into train, validation, and test sets
def prepare_data():
    # List all files
    real_files = [os.path.join(TRAINING_REAL_DIR, f) for f in os.listdir(TRAINING_REAL_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    fake_files = [os.path.join(TRAINING_FAKE_DIR, f) for f in os.listdir(TRAINING_FAKE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Split real images
    real_train, real_temp = train_test_split(real_files, test_size=0.3, random_state=42)
    real_valid, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)

    # Split fake images
    fake_train, fake_temp = train_test_split(fake_files, test_size=0.3, random_state=42)
    fake_valid, fake_test = train_test_split(fake_temp, test_size=0.5, random_state=42)

    # Copy files to their respective directories
    def copy_files(file_list, destination_dir):
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(destination_dir, file_name)
            if not os.path.exists(dest_path):
                import shutil
                shutil.copy(file_path, dest_path)

    # Copy validation and test files
    copy_files(real_valid, VALID_REAL_DIR)
    copy_files(fake_valid, VALID_FAKE_DIR)
    copy_files(real_test, TEST_REAL_DIR)
    copy_files(fake_test, TEST_FAKE_DIR)

    print(f"Training: {len(real_train)} real, {len(fake_train)} fake")
    print(f"Validation: {len(real_valid)} real, {len(fake_valid)} fake")
    print(f"Test: {len(real_test)} real, {len(fake_test)} fake")

# Data generators
def create_data_generators():
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation and test data generator - only rescaling
    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow from directory for all datasets
    train_flow = train_datagen.flow_from_directory(
        os.path.dirname(TRAINING_REAL_DIR),  # Use parent directory of training_real
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=['training_fake', 'training_real'],  # Specify class folder names
        class_mode='binary'
    )

    valid_flow = valid_test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_flow = valid_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        shuffle=False,
        class_mode='binary'
    )

    return train_flow, valid_flow, test_flow

# Attention block function
def attention_block(x, filters):
    # Compute attention weights
    attention = Conv2D(filters, (1, 1), padding='same')(x)
    attention = BatchNormalization()(attention)
    attention = tf.keras.activations.sigmoid(attention)

    # Apply attention weights to input feature map
    return x * attention

# Build enhanced discriminator model with attention mechanism
def build_enhanced_discriminator():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # First block
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Second block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Third block with attention
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = attention_block(x, 128)  # Apply attention
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Fourth block
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = attention_block(x, 256)  # Apply attention
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Fifth block
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Output layers
    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Enhanced_Discriminator")
    return model

# Main execution function
def main():
    print("Preparing data...")
    prepare_data()

    print("Setting up data generators...")
    train_flow, valid_flow, test_flow = create_data_generators()

    print("Building model...")
    model = build_enhanced_discriminator()

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    print("\nModel Summary:")
    model.summary()

    # Create checkpoint callback
    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_path, 'best_model.weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train the model
    print("\nStarting model training...")
    start_time = time.time()
    
    history = model.fit(
        train_flow,
        steps_per_epoch=train_flow.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=valid_flow,
        validation_steps=valid_flow.samples // BATCH_SIZE,
        callbacks=[checkpoint]
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Plot training history
    plt.figure(figsize=(12, 5))
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

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'training_history.png'))

    # Save final model
    model_save_path = os.path.join(base_path, 'final_deepfake_detector.h5')
    tf.keras.models.save_model(model, model_save_path)
    print(f"Final model saved to {model_save_path}")

if __name__ == "__main__":
    # Check for GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU is available. Using GPU for training.")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found. Using CPU for training.")
    
    main()