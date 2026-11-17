"""Simple deep baseline: small CNN trained from scratch + LSTM decoder (M4 Section 4)."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model(vocab_size, seq_len, image_size=224, units=256, dropout=0.0):
    image = keras.Input(shape=(image_size, image_size, 3), name="image")
    caption_in = keras.Input(shape=(seq_len,), dtype="int32", name="caption_in")

    # Encoder: 3 convolution layers, trained from scratch
    x = layers.Rescaling(1 / 255.0)(image)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    h0 = layers.Dense(units, activation="tanh", name="init_h")(x)
    c0 = layers.Dense(units, activation="tanh", name="init_c")(x)

    # Decoder: word embedding + LSTM, started from the image vector
    words = layers.Embedding(vocab_size, units, mask_zero=True, name="embedding")(caption_in)
    words = layers.LSTM(units, return_sequences=True, name="lstm")(words, initial_state=[h0, c0])
    words = layers.Dropout(dropout)(words)
    logits = layers.Dense(vocab_size, name="word_scores")(words)
    return keras.Model([image, caption_in], logits, name="simple_cnn_lstm")


def greedy_captions(model, image_ds, id_to_word, seq_len):
    """Generate one caption per image: always take the most likely next word."""
    start_id = next(i for i, w in id_to_word.items() if w == "<start>")
    end_id = next(i for i, w in id_to_word.items() if w == "<end>")
    captions = []
    for images in image_ds:
        n = images.shape[0]
        tokens = np.zeros((n, seq_len), dtype=np.int32)
        tokens[:, 0] = start_id
        done = np.zeros(n, dtype=bool)
        for t in range(seq_len - 1):
            logits = model([images, tokens], training=False)
            next_ids = np.argmax(logits[:, t, :].numpy(), axis=-1)
            next_ids[done] = 0
            tokens[:, t + 1] = next_ids
            done |= next_ids == end_id
            if done.all():
                break
        for row in tokens:
            words = []
            for i in row[1:]:
                if i in (end_id, 0):
                    break
                if id_to_word[int(i)] != "<unk>":
                    words.append(id_to_word[int(i)])
            captions.append(" ".join(words))
    return captions


if __name__ == "__main__":
    model = build_model(vocab_size=2533, seq_len=23)
    model.summary()
    print(f"Trainable parameters: {model.count_params():,}")