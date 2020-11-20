import collections
import pathlib
import re
import string

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization







train_dir = 'testingoutput'
val_dir = 'expected_text_result'
test_dir = 'test_data'

batch_size = 32
seed = 42

labels =['home depot', 'walmart', 'Tea', 'safeway', 'walmart', 'walmart', 'walmart']
train_ds = preprocessing.text_dataset_from_directory(train_dir,
    batch_size=batch_size,
    seed=seed)
val_ds = preprocessing.text_dataset_from_directory(val_dir,
    batch_size=batch_size,
    seed=seed)

test_ds = preprocessing.text_dataset_from_directory(test_dir,
    batch_size=batch_size,
    seed=seed)

VOCAB_SIZE = 10000

binary_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary')

MAX_SEQUENCE_LENGTH = 250

int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

train_text = train_ds.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)

def binary_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return binary_vectorize_layer(text), label

def int_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return int_vectorize_layer(text), label

# Retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(train_ds))
first_question, first_label = text_batch[0], label_batch[0]
print("Question", first_question)
print("Label", first_label)

binary_train_ds = train_ds.map(binary_vectorize_text)
binary_val_ds = val_ds.map(binary_vectorize_text)
binary_test_ds = test_ds.map(binary_vectorize_text)

int_train_ds = train_ds.map(int_vectorize_text)
int_val_ds = val_ds.map(int_vectorize_text)
int_test_ds = test_ds.map(int_vectorize_text)

AUTOTUNE = tf.data.experimental.AUTOTUNE

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

binary_model = tf.keras.Sequential([tf.keras.layers.Dense(4)])
binary_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
history = binary_model.fit(
    binary_train_ds, validation_data=binary_val_ds, epochs=10)


# def create_model(vocab_size, num_labels):
#   model = tf.keras.Sequential([
#       layers.Embedding(vocab_size, 64, mask_zero=True),
#       layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
#       layers.GlobalMaxPooling1D(),
#       layers.Dense(num_labels)
#   ])
#   return model
#
#
# # vocab_size is VOCAB_SIZE + 1 since 0 is used additionally for padding.
# int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=4)
# int_model.compile(
#     loss=losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer='adam',
#     metrics=['accuracy'])
# history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=5)