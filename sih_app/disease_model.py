import tensorflow as tf
from tensorflow import keras
import os
import json

# ===== Config =====
image_size = 224
target_size = (image_size, image_size)
input_shape = (image_size, image_size, 3)
batch_size = 32
epochs = 25

# ===== Dataset paths (commented out for now) =====
# base_dir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"
# train_dir = os.path.join(base_dir, "train")
# test_dir = os.path.join(base_dir, "valid")

# ===== Data generators (ready to use when dataset is downloaded) =====
# train_datagen = keras.preprocessing.image.ImageDataGenerator(
#     rescale=1/255.0,
#     shear_range=0.2,
#     zoom_range=0.2,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     fill_mode="nearest"
# )
# test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)

# ===== Placeholder predict_disease function =====
def predict_disease(image_file):
    """
    Temporary dummy function until the model is trained.
    Returns:
        disease_name (str): Predicted disease name (dummy)
        recommended_action (str): Recommended action (dummy)
    """
    return "Unknown Disease", "No recommended action yet"

# ===== Real training code (commented out until dataset is available) =====
"""
# Load data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical"
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical"
)

# Save class indices
categories = list(train_data.class_indices.keys())
with open('class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f)

# Build MobileNet model
base_model = tf.keras.applications.MobileNet(
    weights="imagenet",
    include_top=False,
    input_shape=input_shape
)
base_model.trainable = False

inputs = keras.Input(shape=input_shape)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(len(categories), activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=x, name="LeafDisease_MobileNet")
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(), 'accuracy']
)

# Train the model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    steps_per_epoch=150,
    validation_steps=100
)

# Save the trained model
model.save('plant_disease')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
"""
