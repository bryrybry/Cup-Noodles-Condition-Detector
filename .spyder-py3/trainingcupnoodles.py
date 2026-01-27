import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_dir = r"C:\Users\eizin\.spyder-py3\rps\CupNoodles\train"
test_dir  = r"C:\Users\eizin\.spyder-py3\rps\CupNoodles\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ SAVE CLASS NAMES BEFORE prefetch/shuffle
class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

# ✅ use num_classes instead of train_ds.class_names
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(train_ds, validation_data=test_ds, epochs=10)

model.save("cupnoodles_model.keras")
print("Saved: cupnoodles_model.keras")
