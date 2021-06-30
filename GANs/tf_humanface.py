import tensorflow as tf

tf.__version__

import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time

from IPython import display

# Load dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
print(train_images.shape()[0])
print(train_images.shape()[1])
print(train_images.shape()[2])


# Reshape the image to [60000 28 28 1] in data type float32
train_images = train_images.reshape(train_images.shape()[0], train_images.shape()[1], train_images.shape()[2], 1).astype('float32')

# Convert the data range from image greyscale to normalized value (- 1 to 1)
train_images = train_images / 127.5 - 1.0

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#
# create GANs models
def make_discriminator_model():
    model = tf.keras.Sequential()

    # first layer
    # Add layers
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    # Add activation function
    model.add(layers.LeakyReLU())
    # Add Dropout
    model.add(layers.Dropout(0.2))


    # second layer
    # Add layers
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    # Add activation function
    model.add(layers.LeakyReLU())
    # Add Dropout
    model.add(layers.Dropout(0.2))


    # Add Flatten and Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    print(model.summary())

    return model



def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape(7, 7, 256))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization)
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization)
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model


discriminator = make_discriminator_model()
generator = make_generator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(fake_output, real_output):
    return cross_entropy(tf.zeros_like(fake_output), fake_output) \
           + cross_entropy(tf.ones_like(real_output), real_output)

EPOCHS = 50
seed = tf.random.normal([16, 100])

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discrminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# one epoch
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(fake_output, real_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discrminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_image(model, test_input): # test_input is noise
    prediction = model(test_input, training=False)

    for i in range(prediction.shape[0]):
        plt.imshow(prediction[i, :, :, 0] * 127.5 + 127.5, cmap='gray')


def train(dataset, epochs):
    for epochs in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        generate_image(generator, seed)
        print('Epoch {} is finished.'.format(epochs))

    display.clear_output(wait=True)
    generate_image(generator, seed)

train(train_dataset, EPOCHS)
