from numpy import expand_dims, zeros, ones, vstack
from numpy.random import randn, randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from matplotlib import pyplot as plt

# -----------------------------
# Discriminator
# -----------------------------
def define_discriminator(in_shape=(28,28,1)):
    model = Sequential([
        Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=in_shape),
        LeakyReLU(0.2),
        Dropout(0.4),
        Conv2D(64, (3,3), strides=(2,2), padding='same'),
        LeakyReLU(0.2),
        Dropout(0.4),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# -----------------------------
# Generator
# -----------------------------
def define_generator(latent_dim):
    model = Sequential([
        Dense(128*7*7, input_dim=latent_dim),
        LeakyReLU(0.2),
        Reshape((7,7,128)),
        Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        LeakyReLU(0.2),
        Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        LeakyReLU(0.2),
        Conv2D(1, (7,7), activation='tanh', padding='same')
    ])
    return model

# -----------------------------
# GAN
# -----------------------------
def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential([g_model, d_model])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# -----------------------------
# Load MNIST
# -----------------------------
def load_real_samples():
    (trainX, _), (_, _) = load_data()
    X = expand_dims(trainX, axis=-1)
    X = (X - 127.5) / 127.5  # scale to [-1,1]
    return X

# -----------------------------
# Sample real and fake images
# -----------------------------
def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples,1)) * 0.9  # label smoothing
    return X, y

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples).reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input, verbose=0)
    y = zeros((n_samples,1))
    return X, y

# -----------------------------
# Save images
# -----------------------------
def save_plot(examples, epoch, n=10):
    examples = (examples + 1) / 2.0  # rescale to [0,1] for plotting
    for i in range(n*n):
        plt.subplot(n,n,i+1)
        plt.axis('off')
        plt.imshow(examples[i,:,:,0], cmap='gray')
    plt.savefig(f'generated_plot_e{epoch+1:03d}.png')
    plt.close()

# -----------------------------
# Summarize performance
# -----------------------------
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
    print(f'>Accuracy real: {acc_real*100:.0f}%, fake: {acc_fake*100:.0f}%')
    save_plot(X_fake, epoch)
    g_model.save(f'generator_model_{epoch+1:03d}.h5')

# -----------------------------
# Train GAN
# -----------------------------
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
    bat_per_epo = dataset.shape[0] // n_batch
    half_batch = n_batch // 2
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch,1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print(f'>{i+1}, {j+1}/{bat_per_epo}, d={d_loss:.3f}, g={g_loss:.3f}')
        summarize_performance(i, g_model, d_model, dataset, latent_dim)

# -----------------------------
# Run GAN
# -----------------------------
latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
dataset = load_real_samples()
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64)
