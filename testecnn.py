

#import h5py
import tensorflow_datasets as tfds
import tensorflow as tf
import itertools
import numpy as np


_, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples # 3670
class_names = info.features["label"].names # ["dandelion", "daisy", ...]
n_classes = info.features["label"].num_classes # 5

train_set_raw = tfds.load(
 'tf_flowers',
 split='train',
 as_supervised=True)

print(type(train_set_raw))

'''test_set_raw = dict(itertools.islice(train.items(), int(len(train)*0.1)))

valid_set_raw= dict(itertools.islice(train.items(), int(len(train)*0.25)))
train_set_raw= {x:train[x] for x in train if x not in valid_set_raw}
valid_set_raw = {x:valid_set_raw[x] for x in valid_set_raw if x not in test_set_raw}

valid_set_raw= np.array(list(valid_set_raw.items()))
train_set_raw= np.array(list(train_set_raw.items()))
test_set_raw= np.array(list(test_set_raw.items()))'''


batch_size = 32
preprocess = tf.keras.Sequential([
 tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
 tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])

train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)

data_augmentation = tf.keras.Sequential([
 tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
 tf.keras.layers.RandomRotation(factor=0.05, seed=42),
 tf.keras.layers.RandomContrast(factor=0.2, seed=42)
])

base_model = tf.keras.applications.xception.Xception(weights="imagenet",
 include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
 layer.trainable = False

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=3)