
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout, LayerNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras import __version__ as keras_version
from tensorflow.keras.models import load_model
from tensorflow.random import set_seed
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
from numpy.random import seed
import sys
from collections import Counter
LEARN_MODE = 'x'  # Options: 'x' or 'xy'



# Set image size
image_size = {"width": 200, "height": 66}


# Fix every random seed to make the training reproducible
seed(1)
set_seed(2)
random.seed(42)

sys.stdout.flush()
print("[INFO] Version:")
print("Tensorflow version: %s" % tf.__version__)
keras_version = str(keras_version).encode('utf8')
print("Keras version: %s" % keras_version)

# --- Check for GPU availability ---
print("[INFO] Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"[INFO] {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured.")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(f"[WARNING] Could not set memory growth: {e}")
    print("[INFO] GPUs found, but memory growth configuration failed (might be okay).")
else:
    print("[INFO] No GPU found. Training will fall back to CPU.")
# ---------------------------------


############ CNN construction ############


# Model structure taken from: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
def build_CNN(width, height, depth, activation='relu', dropout=0.25, output_size=2):

    # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)

    # After Keras 2.3 we need an Input layer instead of passing it as a parameter to the first layer
    model.add(Input(inputShape))

    # Adding normalization - applied per-sample across the feature dimension (channels)
    model.add(LayerNormalization())

    # Input: 3@66x200 (height=66, width=200, depth=3)

    # --- Convolutional Layers ---
    # Note: Output dimensions calculated based on input (66, 200)
    # Formula: floor((Input - Kernel + 2*Padding) / Stride) + 1
    # With padding='valid', Padding=0

    # Layer 1: Conv(5x5, S=2x2, P=valid) + Activation + Dropout -> Output: 24@31x98
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation=activation, padding="valid"))
    model.add(Dropout(dropout))

    # Layer 2: Conv(5x5, S=2x2, P=valid) + Activation + Dropout -> Output: 36@14x47
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activation, padding="valid"))
    model.add(Dropout(dropout))

    # Layer 3: Conv(5x5, S=2x2, P=valid) + Activation + Dropout -> Output: 48@5x22
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activation, padding="valid"))
    model.add(Dropout(dropout))

    # Layer 4: Conv(3x3, S=1x1, P=valid) + Activation + Dropout -> Output: 64@3x20
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation=activation, padding="valid"))
    model.add(Dropout(dropout))

    # Layer 5: Conv(3x3, S=1x1, P=valid) + Activation + Dropout -> Output: 64@1x18
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation=activation, padding="valid"))
    model.add(Dropout(dropout))


    # --- Fully Connected Layers ---
    model.add(Flatten())

    # 1st set of FC -> RELU layers + Dropout
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    # 2nd set of FC -> RELU layers + Dropout
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    # 3rd set of FC -> RELU layers + Dropout
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(output_size))

    # return the constructed network architecture
    return model



############# Training data preparation ############


dataset = os.path.join('..', 'labelled_data')
# initialize the data and labels
print("[INFO] loading images and labels...")
data = []
labels = [] # Will store lists of [x_val, y_val]

try:
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.shuffle(imagePaths)
except:
    print(f"[ERROR] Could not load images from {dataset}. Exiting.")
    exit()

# loop over the input images
processed_count = 0
skipped_count = 0
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Could not read image: {imagePath}. Skipping.")
        skipped_count += 1
        continue

    # Resize image - cv2.resize expects (width, height) -> (200, 66)
    try:
        image = cv2.resize(image, (image_size["width"], image_size["height"]))
    except Exception as e:
        print(f"[ERROR] Could not resize image: {imagePath}. Error: {e}. Skipping.")
        skipped_count += 1
        continue

    image = img_to_array(image) # Converts to numpy array (H, W, C) and dtype=float32 -> (66, 200, 3)


    # Extract the filename from the path
    filename = os.path.basename(imagePath)
    # Remove the file extension (.png, .jpg, etc.)
    filename_no_ext = os.path.splitext(filename)[0]
    # Split the filename by '_'
    parts = filename_no_ext.split('_')

    # Find the parts containing 'X' and 'Y' and extract the float values
    try:
        x_val = None
        y_val = None
        for part in parts:
            if part.startswith('X') and len(part) > 1:
                # Replace 'p' with '.' and 'n' with '-' before converting to float
                x_str = part[1:].replace('p', '.').replace('n', '-')
                x_val = float(x_str)
            elif part.startswith('Y') and len(part) > 1:
                y_str = part[1:].replace('p', '.').replace('n', '-')
                y_val = float(y_str)

        # Check if both values were found and are valid numbers
        if x_val is not None and y_val is not None:
            label = [x_val, y_val]
            labels.append(label)
            data.append(image) # Append image only if label is valid
            processed_count += 1
            # Optional: print progress less frequently for large datasets
            if processed_count % 500 == 0:
               print(f"Processed {processed_count} images...")
        else:
            # Raise error only if parsing seemed possible but failed, otherwise just warn
             print(f"[WARNING] Could not parse valid X/Y values from filename: {filename}. Skipping image.")
             skipped_count += 1
             continue # Skip to the next image

    except Exception as e: # Catch broader exceptions during parsing logic
        print(f"[ERROR] Unexpected error parsing filename: {filename}. Error: {e}. Skipping image.")
        skipped_count += 1
        continue # Skip to the next image

print(f"[INFO] Finished loading data. Processed: {processed_count}, Skipped: {skipped_count}")

if not data:
    print("[ERROR] No data loaded. Exiting.")
    exit()

# --- Convert lists to NumPy arrays ---
# Scale the raw pixel intensities to the range [0, 1]
# Perform scaling *after* splitting to prevent data leakage from test set statistics (though simple /255 is less sensitive)
data = np.array(data, dtype="float32")
labels = np.array(labels, dtype="float32")


# Define a flag to switch between 'x' and 'xy' learning
LEARN_MODE = 'x'  # Options: 'x' or 'xy'

# Adjust labels based on the learning mode
if LEARN_MODE == 'x':
    labels = labels[:, 0:1]  # Keep only the x values
elif LEARN_MODE == 'xy':
    labels = labels[:, 0:2]  # Keep both x and y values
else:
    raise ValueError("Invalid LEARN_MODE. Choose 'x' or 'xy'.")

# Modify the CNN construction function to handle the output size dynamically
def build_CNN(width, height, depth, activation='relu', dropout=0.25, output_size=2):
    # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)

    # After Keras 2.3 we need an Input layer instead of passing it as a parameter to the first layer
    model.add(Input(inputShape))

    # Adding normalization - applied per-sample across the feature dimension (channels)
    model.add(LayerNormalization())

    # Input: 3@66x200 (height=66, width=200, depth=3)

    # --- Convolutional Layers ---
    # Note: Output dimensions calculated based on input (66, 200)
    # Formula: floor((Input - Kernel + 2*Padding) / Stride) + 1
    # With padding='valid', Padding=0

    # Layer 1: Conv(5x5, S=2x2, P=valid) + Activation + Dropout -> Output: 24@31x98
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation=activation, padding="valid"))
    model.add(Dropout(dropout))

    # Layer 2: Conv(5x5, S=2x2, P=valid) + Activation + Dropout -> Output: 36@14x47
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activation, padding="valid"))
    model.add(Dropout(dropout))

    # Layer 3: Conv(5x5, S=2x2, P=valid) + Activation + Dropout -> Output: 48@5x22
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activation, padding="valid"))
    model.add(Dropout(dropout))

    # Layer 4: Conv(3x3, S=1x1, P=valid) + Activation + Dropout -> Output: 64@3x20
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation=activation, padding="valid"))
    model.add(Dropout(dropout))

    # Layer 5: Conv(3x3, S=1x1, P=valid) + Activation + Dropout -> Output: 64@1x18
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation=activation, padding="valid"))
    model.add(Dropout(dropout))


    # --- Fully Connected Layers ---
    model.add(Flatten())

    # 1st set of FC -> RELU layers + Dropout
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    # 2nd set of FC -> RELU layers + Dropout
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    # 3rd set of FC -> RELU layers + Dropout
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(output_size))

    return model

# Update the model instantiation to use the LEARN_MODE flag
output_size = 1 if LEARN_MODE == 'x' else 2
model = build_CNN(width=image_size["width"], height=image_size["height"], depth=3, dropout=DROPOUT_RATE, output_size=output_size)


# Extract the x values from the labels array
all_x = labels[:, 0]  # Assuming labels is a 2D array where the first column is `x` and the second is `y`

# Example: 21 equal-width bins in the interval [-1.0, +1.0]
bins   = np.linspace(-1, 1, 22)
bucket = np.digitize(all_x, bins)

# pick at most N images per bucket
balanced_idx = []
for b in np.unique(bucket):
    idx = np.where(bucket == b)[0]
    balanced_idx.extend( np.random.choice(idx, min(len(idx), 400)) )  # 400≈sqrt(2000)

data   = data[balanced_idx]
labels = labels[balanced_idx, 0:1]      # keep only x (see next point)


# --- Split data ---
# Partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, random_state=42) # Use the same random_state for reproducibility

# --- Scale data after splitting ---
trainX = trainX / 255.0
testX = testX / 255.0

print(f"[INFO] Data split complete. Training samples: {len(trainX)}, Test samples: {len(testX)}")




############### Training the model ###############

# --- Model Compilation and Training ---
print("[INFO] compiling model...")
# Define hyperparameters
LEARNING_RATE = 3e-4 #1e-3
EPOCHS = 50
BATCH_SIZE = 32
DROPOUT_RATE = 0.1

# Build the model (3 layer, RGB)
model = build_CNN(width=image_size["width"], height=image_size["height"], depth=3, dropout=DROPOUT_RATE)

# Compile the model for regression
opt = Adam(learning_rate=LEARNING_RATE)
# Using Mean Squared Error loss, add Mean Absolute Error for interpretability
model.compile(loss="mse", optimizer=opt, metrics=["mae", "mse"])

# Print model summary
model.summary()

# Reduce learning rate when a metric has stopped improving
# Monitor 'val_loss' which is generally preferred over training loss
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Save the best model based on validation loss
checkpoint_filepath = os.path.join("..", "network_model", "best_model.keras")
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, verbose=1)

# Add EarlyStopping to callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
callbacks_list=[lr_scheduler, checkpoint, early_stopping]

print("[INFO] training network...")
# Ensure data types are correct (TensorFlow prefers float32)
H = model.fit(trainX.astype('float32'), trainY.astype('float32'),
              validation_data=(testX.astype('float32'), testY.astype('float32')),
              batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
              callbacks=callbacks_list)

# Save the model after the last epoch
last_model_filepath = os.path.join("..", "network_model", "last_model.keras")
model.save(last_model_filepath)
print(f"[INFO] Last model saved to {last_model_filepath}")
print(f"[INFO] Best model saved to {checkpoint_filepath} (based on val_loss)")


print("[INFO] evaluating network using the best saved model...")
# Check if the best model file exists before loading
if os.path.exists(checkpoint_filepath):
    try:
        model = load_model(checkpoint_filepath)
        print(f"[INFO] Successfully loaded best model from {checkpoint_filepath}")

        predictions = model.predict(testX, batch_size=BATCH_SIZE)
        eval_results = model.evaluate(testX, testY, batch_size=BATCH_SIZE, verbose=0)
        print(f"[INFO] Evaluation on test set (Best Model) - Loss: {eval_results[0]:.4f}, MAE: {eval_results[1]:.4f}, MSE: {eval_results[2]:.4f}")

        # Plotting should ideally happen only if evaluation succeeded
        print("[INFO] plotting training history...")
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["mae"], label="train_mae")
        plt.plot(np.arange(0, N), H.history["val_mae"], label="val_mae")
        plt.title("Training Loss and MAE (200x66 Input)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/MAE")
        plt.legend(loc="best")
        plt.show()

    except Exception as e:
        print(f"[ERROR] Failed to load or evaluate the best model from {checkpoint_filepath}. Error: {e}")
else:
    print(f"[WARNING] Best model file not found at {checkpoint_filepath}. Skipping evaluation and plotting.")



print("[INFO] Script finished.")
