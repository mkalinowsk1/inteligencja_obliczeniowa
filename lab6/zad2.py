import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

# KONFIGURACJA
DATA_PATH = '/home/kali/Desktop/code/dogs-cats-mini'   
IMG_SIZE = (224, 224)  # Większy rozmiar dla pre-trained models
BATCH_SIZE = 32
EPOCHS = 15  # Możesz zmniejszyć jeśli chcesz szybszy trening
VALIDATION_SPLIT = 0.2            
SEED = 42

print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Przygotowanie struktury katalogów
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

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
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


# Załaduj pre-trained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

# Zamroź warstwy bazowego modelu
base_model.trainable = False


# Budowa pełnego modelu
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
checkpoint_cb = ModelCheckpoint(
    'best_model_transfer_learning.h5', 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)
earlystop_cb = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-7, 
    verbose=1
)

# TRENING
print(f"\n{'='*60}")
print(f"Rozpoczynam trening...")
print(f"{'='*60}\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
    verbose=1
)

# WYKRESY UCZENIA
def plot_history(history):
    plt.figure(figsize=(14, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy - Transfer Learning', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss - Transfer Learning', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('wykres1.png', dpi=300, bbox_inches='tight')

plot_history(history)

# OCENA MODELU
print(f"\n{'='*60}")
print(f"WYNIKI TRANSFER LEARNING")
print(f"{'='*60}\n")

best_train_acc = max(history.history['accuracy'])
best_val_acc = max(history.history['val_accuracy'])
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"Najlepsza dokładność treningowa:  {best_train_acc:.4f}")
print(f"Najlepsza dokładność walidacyjna: {best_val_acc:.4f}")
print(f"Finalna dokładność treningowa:    {final_train_acc:.4f}")
print(f"Finalna dokładność walidacyjna:   {final_val_acc:.4f}")

# PREDYKCJA I MACIERZ BŁĘDÓW
y_true = []
y_pred = []
filenames = validation_generator.filenames

validation_generator.reset()
steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))

print(f"\nGenerowanie predykcji dla {validation_generator.samples} obrazów walidacyjnych...")

for i in range(steps):
    X_batch, y_batch = validation_generator.__next__()
    preds = model.predict(X_batch, verbose=0)
    preds_bin = (preds.ravel() >= 0.5).astype(int)
    y_true.extend(y_batch.astype(int))
    y_pred.extend(preds_bin)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
print("\n" + "="*60)
print("CONFUSION MATRIX:")
print("="*60)
print(cm)
print("\n" + "="*60)
print("CLASSIFICATION REPORT:")
print("="*60)
print(classification_report(y_true, y_pred, target_names=['cat','dog']))

# Wizualizacja confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Transfer Learning (MobileNetV2)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20, fontweight='bold')
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('wykres2.png', dpi=300, bbox_inches='tight')

plot_confusion_matrix(cm, classes=['Cat','Dog'])

# BŁĘDNIE SKLASYFIKOWANE OBRAZKI
errors_idx = np.where(y_true != y_pred)[0]
accuracy = (len(y_true) - len(errors_idx)) / len(y_true) * 100


print(f"Liczba błędnych klasyfikacji: {len(errors_idx)} / {len(y_true)}")
print(f"Dokładność na zbiorze walidacyjnym: {accuracy:.2f}%")

num_to_show = min(10, len(errors_idx))
if num_to_show > 0:
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(errors_idx[:num_to_show]):
        fname = filenames[idx]
        img_path = os.path.join(validation_generator.directory, fname)
        img = plt.imread(img_path)
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.axis('off')
        true_label = 'Dog' if y_true[idx]==1 else 'Cat'
        pred_label = 'Dog' if y_pred[idx]==1 else 'Cat'
        plt.title(f"True: {true_label}\nPred: {pred_label}", 
                 fontsize=11, fontweight='bold',
                 color='red')
    plt.suptitle('Misclassified Examples - Transfer Learning', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('wykres3.png', dpi=300, bbox_inches='tight')
else:
    print("Brak błędnych klasyfikacji!")

