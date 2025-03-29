import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose
import matplotlib.pyplot as plt

mnist_path = "/content/mnist.npz"  
with np.load(mnist_path) as data:
    x_train = data['x_train']

x_train = (x_train.astype(np.float32) - 127.5) / 127.5  
x_train = np.expand_dims(x_train, axis=-1) 

latent_dim = 100 
img_shape = (28, 28, 1)

# Generator Model
def build_generator():
    model = keras.Sequential([
        Dense(7 * 7 * 256, input_dim=latent_dim),
        Reshape((7, 7, 256)),
        BatchNormalization(),
        Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(1, kernel_size=3, strides=1, padding="same", activation='tanh')
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = keras.Sequential([
        Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model


generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])


discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = keras.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(0.0002, 0.5))


batch_size = 128
epochs = 5000
sample_interval = 500

real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

 
        noise = np.random.normal(0, 1, (25, latent_dim))
        gen_imgs = generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5 

        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        plt.show()
