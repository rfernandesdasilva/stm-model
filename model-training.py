import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16

# setting up enviroment to speed up epoch learning
# took me ages without this, changed so i would use 100% of my cpu. need to be careful with this tho, not all cpus have 20 threads and 10 cores
# DO NOT UNCOMMENT THIS UNLESS YOU KNOW YOUR CPU CONFIGURATION.

# os.environ["OMP_NUM_THREADS"] = "20"  # number of threads should be equal to the number of logical processors
# os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
# os.environ["TF_NUM_INTEROP_THREADS"] = "2"  # typically set this to 2 or the number of physical cores

# configure TensorFlow to use multiple cores
# tf.config.threading.set_intra_op_parallelism_threads(20)
# tf.config.threading.set_inter_op_parallelism_threads(2)

# replace placeholds with your directories
train_dir = r"ADD-DIR-HERE"
validation_dir = r"ADD-DIR-HERE"
test_dir = r"ADD-DIR-HERE"
model_save_dir = r"ADD-DIR-HERE"

os.makedirs(model_save_dir, exist_ok=True)
version = 1.0 # version control
model_save_path = os.path.join(model_save_dir, f'stm-v.{version}.keras')

# image data generators with augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# validation and test data generators (without augmentation)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# data loaders
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# freeze the base model initially
for layer in base_model.layers:
    layer.trainable = False

# add custom layers on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 classes: Cilinder, Cone, Cube, Pyramid, Sphere
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# train the model with frozen base layers
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# unfreeze more layers in the base model for fine-tuning
for layer in base_model.layers[-8:]:  # Unfreezing the last 8 layers
    layer.trainable = True

# re compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# fine-tune
history_fine = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

model.save(model_save_path)
print(f'Model saved at: {model_save_path}')


