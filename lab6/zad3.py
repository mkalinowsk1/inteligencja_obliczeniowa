# gan7_fixed.py
import os
import numpy as np
from numpy.random import randn, randint
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets.mnist import load_data

# ========== helpers ==========
def save_models(g_model, d_model, epoch, folder="models"):
    os.makedirs(folder, exist_ok=True)
    g_model.save(os.path.join(folder, f"generator_e{epoch:03d}.h5"))
    d_model.save(os.path.join(folder, f"discriminator_e{epoch:03d}.h5"))
    print(f"Saved models at epoch {epoch}")

def load_models(g_path, d_path):
    g = load_model(g_path)
    d = load_model(d_path)
    return g, d

# ========== define models ==========
def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7,7,128)))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    # output 28x28x1, tanh for [-1,1]
    model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
    return model

def define_discriminator(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# ========== data ==========
def load_real_samples():
    (trainX, _), (testX, _) = load_data()
    X = np.expand_dims(trainX, axis=-1)  # <-- use train set
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X

# ========== sample generators ==========
def generate_real_samples(dataset, n_samples, smooth=0.9):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples,1)) * smooth  # label smoothing
    return X, y

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input, verbose=0)
    y = np.zeros((n_samples,1))
    return X, y

# ========== plotting ==========
def save_plot(examples, epoch, n=10, folder="plots"):
    os.makedirs(folder, exist_ok=True)
    examples = (examples + 1) / 2.0  # rescale to [0,1] for plotting
    fig = plt.figure(figsize=(6,6))
    for i in range(n*n):
        ax = fig.add_subplot(n, n, 1 + i)
        ax.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    filename = os.path.join(folder, f'generated_plot_e{epoch:03d}.png')
    plt.savefig(filename)
    plt.close(fig)

# ========== training ==========
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=128, save_every=5):
    bat_per_epo = max(1, int(dataset.shape[0] / n_batch))
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            # add small label noise (flip some labels)
            if np.random.rand() < 0.05:
                y[np.random.randint(0, y.shape[0], size=1)] = 1 - y[np.random.randint(0, y.shape[0], size=1)]

            d_loss, d_acc = d_model.train_on_batch(X, y)

            # prepare points for generator training (try to trick discriminator)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch,1))  # want generator to produce outputs discriminator labels as real
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            print(f"Epoch {i+1}/{n_epochs}, Batch {j+1}/{bat_per_epo}, d_loss={d_loss:.3f}, d_acc={d_acc:.3f}, g_loss={g_loss:.3f}")
        # end epoch
        if (i+1) % save_every == 0 or i == 0 or (i+1)==n_epochs:
            # evaluate and save
            X_real_eval, y_real_eval = generate_real_samples(dataset, 100)
            _, acc_real = d_model.evaluate(X_real_eval, y_real_eval, verbose=0)
            X_fake_eval, y_fake_eval = generate_fake_samples(g_model, latent_dim, 100)
            _, acc_fake = d_model.evaluate(X_fake_eval, y_fake_eval, verbose=0)
            print(f"> Accuracy real: {acc_real*100:.1f}%, fake: {acc_fake*100:.1f}%")
            save_plot(X_fake_eval, i+1)
            save_models(g_model, d_model, i+1)
    print("Training finished.")

# ========== main ==========
if __name__ == "__main__":
    latent_dim = 100
    g_model = define_generator(latent_dim)
    d_model = define_discriminator()
    gan_model = define_gan(g_model, d_model)
    dataset = load_real_samples()
    train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=3, n_batch=128, save_every=5)
