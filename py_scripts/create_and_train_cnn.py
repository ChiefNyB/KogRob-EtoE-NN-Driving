# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout, LayerNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import __version__ as keras_version
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
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

# Set image size
image_size = {"width": 200, "height": 66}


# Fix every random seed to make the training reproducible
seed(1)
set_seed(2)
random.seed(42)

print("[INFO] Version:")
print("Tensorflow version: %s" % tf.__version__)
keras_version = str(keras_version).encode('utf8')
print("Keras version: %s" % keras_version)



############ CNN construction ############


# Model structure taken from: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
def build_CNN(width, height, depth, activation='relu', dropout=0.25):

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

    # XY output (X:forward-backward, Y:left-right)
    model.add(Dense(2))

    # return the constructed network architecture
    return model



############# Training data preparation ############

dataset = '..//labelled_data' # Consider using os.path.join for cross-platform compatibility
# initialize the data and labels
print("[INFO] loading images and labels...")
data = []
labels = [] # Will store lists of [x_val, y_val]

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset)))
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Could not read image: {imagePath}. Skipping.")
        continue

    # Resize image - cv2.resize expects (width, height) -> (200, 66)
    try:
        image = cv2.resize(image, (image_size["width"], image_size["height"]))
    except Exception as e:
        print(f"[ERROR] Could not resize image: {imagePath}. Error: {e}. Skipping.")
        continue

    image = img_to_array(image) # Converts to numpy array (H, W, C) and dtype=float32 -> (66, 200, 3)
    data.append(image)

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
                x_val = float(part[1:]) # Get the value after 'X'
            elif part.startswith('Y') and len(part) > 1:
                y_val = float(part[1:]) # Get the value after 'Y'

        # Check if both values were found
        if x_val is not None and y_val is not None:
            label = [x_val, y_val]
            labels.append(label)
            # Optional: print progress less frequently for large datasets
            # if len(labels) % 100 == 0:
            #    print(f"Processed {len(labels)} images...")
        else:
            raise ValueError("X or Y value not found in filename parts")

    except (ValueError, IndexError, TypeError) as e:
        print(f"[WARNING] Could not parse X/Y values from filename: {filename}. Error: {e}. Skipping image.")
        # Remove the corresponding image data if label parsing failed
        data.pop() # Remove the image added just before the try block
        continue # Skip to the next image

# --- Convert lists to NumPy arrays ---
# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels, dtype="float32")



# --- Split data ---
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, random_state=42) # Use the same random_state for reproducibility



############### Training the model ###############

# --- Model Compilation and Training ---
print("[INFO] compiling model...")
# Define hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32
DROPOUT_RATE = 0.25

# Build the model (3 layer, RGB)
model = build_CNN(width=image_size["width"], height=image_size["height"], depth=3, dropout=DROPOUT_RATE)

# Compile the model for regression
opt = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mse", optimizer=opt, metrics=["mae", "mse"])

# Print model summary
model.summary()

# Reduce learning rate when a metric has stopped improving
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
# Save the best model based on validation loss
checkpoint_filepath = "..//network_model//best_model.keras"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, verbose=1)

# callbacks
callbacks_list=[lr_scheduler, checkpoint]

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
              callbacks=callbacks_list)

model.save("..//network_model//last_model.keras")

print("[INFO] evaluating network...")
model = load_model("..//network_model//best_model.keras")

predictions = model.predict(testX, batch_size=BATCH_SIZE)

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
plt.legend(loc="lower left")
plt.show()


print("[INFO] Script finished.")
