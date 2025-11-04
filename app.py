# ============================================================
# 🦠 COVID-19 / Pneumonia / Normal Classifier - Transfer Learning
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# ============================================================
# ⚙️ DATASET PREPARATION
# ============================================================
train_path = "/content/dataset/train"
val_path = "/content/dataset/val"

# Augment + balance
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_path,
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical'
)

# ============================================================
# 🧩 MODEL
# ============================================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
for layer in base_model.layers:
    layer.trainable = False  # freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ============================================================
# 🚀 TRAIN
# ============================================================
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# ============================================================
# 💾 SAVE MODEL
# ============================================================
model.save("/content/final_model.keras")
