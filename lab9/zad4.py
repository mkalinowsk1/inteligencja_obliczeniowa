import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# --- Ustawienia ---
FILE_NAME = 'gutenberg_book.txt' # Własny plik z książką w formacie plaintext
MODEL_PATH = 'model_words.hdf5'
SEQUENCE_LENGTH = 50 # Długość sekwencji słów (wejście do LSTM)
# Epoki do testu (4a/4b)
INITIAL_EPOCHS = 5
# Epoki do dotrenowania (4d)
ADDITIONAL_EPOCHS = 10

# Wymagane funkcje do przygotowania danych (tokenizacja słowna)
def prepare_data(text, sequence_length):
    text = text.lower()
    # Prosta tokenizacja na słowa
    words = text.split()
    
    # Mapowanie słów na liczby (unikalne ID)
    unique_words = sorted(list(set(words)))
    word_to_int = dict((c, i) for i, c in enumerate(unique_words))
    int_to_word = dict((i, c) for i, c in enumerate(unique_words))
    
    n_words = len(words)
    vocab_size = len(unique_words)
    
    # Przygotowanie sekwencji wejściowych (X) i wyjściowych (Y)
    dataX = []
    dataY = []
    for i in range(0, n_words - sequence_length, 1):
        seq_in = words[i:i + sequence_length]
        seq_out = words[i + sequence_length]
        dataX.append([word_to_int[word] for word in seq_in])
        dataY.append(word_to_int[seq_out])
    
    n_patterns = len(dataX)
    print(f"Całkowita liczba wzorców: {n_patterns}")

    # Przekształcenie X do [wzorce, długość_sekwencji, cechy] (one-hot encoding dla słów)
    X = np.reshape(dataX, (n_patterns, sequence_length, 1))
    X = X / float(vocab_size) # Normalizacja
    
    # One-hot encoding dla Y
    y = tf.keras.utils.to_categorical(dataY)
    
    return X, y, word_to_int, int_to_word, vocab_size, n_patterns

# --- Trening ---
def train_model(X, y, vocab_size, initial_epochs, load_if_exists=False, retrain=False):
    # Załaduj dane z pliku (jeśli symulacja, utwórz plik)
    if not os.path.exists(FILE_NAME):
         with open(FILE_NAME, 'w', encoding='utf-8') as f:
            f.write("The quick brown fox jumps over the lazy dog. The lazy dog then barks loudly. The quick fox laughs and runs away. ")
            print(f"Utworzono przykładowy plik {FILE_NAME}. Zastąp go swoją książką.")

    with open(FILE_NAME, 'r', encoding='utf-8') as f:
        text = f.read()

    X, y, word_to_int, int_to_word, vocab_size, n_patterns = prepare_data(text, SEQUENCE_LENGTH)

    if load_if_exists and os.path.exists(MODEL_PATH):
        # 4d) Dotrenowanie - wczytanie istniejącego modelu
        print(f"Wczytywanie istniejącego modelu z {MODEL_PATH} do dotrenowania...")
        model = load_model(MODEL_PATH)
    else:
        # 4b) Trening początkowy - definicja modelu
        print("Tworzenie nowego modelu LSTM...")
        model = Sequential()
        model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Definicja Checkpointu do zapisywania modelu
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Trening/Dotrenowanie
    print(f"Rozpoczęcie treningu na {initial_epochs} epok...")
    model.fit(X, y, epochs=initial_epochs, batch_size=64, callbacks=callbacks_list)
    
    return word_to_int, int_to_word, vocab_size, n_patterns

# --- Uruchomienie treningu początkowego (4b) ---
print("\n[ZADANIE 4B] TRENING POCZĄTKOWY (Word-by-Word)")
word_to_int_init, int_to_word_init, vocab_size_init, n_patterns_init = train_model(None, None, None, INITIAL_EPOCHS, load_if_exists=False)

# --- Uruchomienie dotrenowania (4d) ---
print("\n[ZADANIE 4D] DOTRENOWANIE (Word-by-Word)")
word_to_int_retrain, int_to_word_retrain, vocab_size_retrain, n_patterns_retrain = train_model(None, None, None, ADDITIONAL_EPOCHS, load_if_exists=True, retrain=True)

print("\nZapisano wytrenowane modele do HDF5.")