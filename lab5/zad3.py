import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


DATA_PATH = '/home/kali/Desktop/code/dogs-cats-mini'   
IMG_SIZE = (150, 150)            
BATCH_SIZE = 32
EPOCHS = 20                      
VALIDATION_SPLIT = 0.2            
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

print("GPUs Available:", tf.config.list_physical_devices('GPU'))

cats_dir = os.path.join(DATA_PATH, 'cats')
dogs_dir = os.path.join(DATA_PATH, 'dogs')

if not os.path.exists(cats_dir) or not os.path.exists(dogs_dir):
    os.makedirs(cats_dir, exist_ok=True)
    os.makedirs(dogs_dir, exist_ok=True)
    for fname in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, fname)
        if os.path.isfile(path):
            low = fname.lower()
            if 'cat' in low:
                os.rename(path, os.path.join(cats_dir, fname))
            elif 'dog' in low:
                os.rename(path, os.path.join(dogs_dir, fname))

# 2) Generatory obrazów i data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT
)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',   
    subset='training',
    seed=SEED
)

validation_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED
)


# 3) Funkcja budująca model 
def build_model(input_shape=(*IMG_SIZE, 3), base_filters=32, dropout_rate=0.5):
    model = Sequential()
    model.add(Conv2D(base_filters, (3,3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(base_filters*2, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(base_filters*4, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4) Callbacks (zapisywanie najlepszego modelu .h5 co epokę tylko jeśli val_accuracy lepszy)
checkpoint_cb = ModelCheckpoint(
    'best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# 5) Trening
model = build_model(base_filters=32, dropout_rate=0.5)
model.summary()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

# 6) Krzywe uczenia (accuracy + loss)
def plot_history(history):
    plt.figure(figsize=(12,5))
    # accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("wykres1.png")

plot_history(history)

# 7) Ocena i macierz błędów 
y_true = []
y_pred = []
filenames = validation_generator.filenames

validation_generator.reset()
steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))
for i in range(steps):
    X_batch, y_batch = validation_generator.__next__()
    preds = model.predict(X_batch)
    preds_bin = (preds.ravel() >= 0.5).astype(int)
    y_true.extend(y_batch.astype(int))
    y_pred.extend(preds_bin)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=['cat','dog']))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('wykres2.png')

plot_confusion_matrix(cm, classes=['cat','dog'], normalize=False)

# 8) Wypisz i pokaż kilka błędnie sklasyfikowanych obrazków

errors_idx = np.where(y_true != y_pred)[0]
print(f"Liczba błędnych klasyfikacji w walidacji: {len(errors_idx)} / {len(y_true)}")


num_to_show = min(10, len(errors_idx))
if num_to_show > 0:
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(errors_idx[:num_to_show]):
        fname = filenames[idx]
        img_path = os.path.join(validation_generator.directory, fname)
        img = plt.imread(img_path)
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.axis('off')
        true_label = 'dog' if y_true[idx]==1 else 'cat'
        pred_label = 'dog' if y_pred[idx]==1 else 'cat'
        plt.title(f"true: {true_label}\npred: {pred_label}")
    plt.tight_layout()
    plt.savefig('wykres3.png')
else:
    print("Brak błędnych klasyfikacji do pokazania.")


