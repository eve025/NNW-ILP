import os
from pathlib import Path

# Cambiar al directorio del notebook para dejar los artefactos allí
HERE = Path(__file__).parent
os.chdir(HERE)
ARTIFACT_DIR = HERE / 'artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as e:
    print('Error importing TensorFlow/Keras:', e)
    raise

import pickle
from sklearn.metrics import classification_report, accuracy_score


def build_model(max_vocab=5000, embed_dim=50, maxlen=20):
    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=embed_dim, input_length=maxlen),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    texts = [
        'win money now',
        'limited offer buy today',
        'meeting at 10am',
        'project deadline extended',
        'click here to claim prize',
        'please review attached document',
        'your account has been suspended',
        'lunch meeting tomorrow',
        'free trial offer',
        'classroom schedule update'
    ]
    labels = [1,1,0,0,1,0,1,0,1,0]

    MAX_VOCAB = 2000
    MAX_LEN = 20

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    import numpy as np
    y = np.array(labels)

    model = build_model(max_vocab=MAX_VOCAB, embed_dim=50, maxlen=MAX_LEN)
    print(model.summary())

    history = model.fit(X, y, epochs=5, batch_size=2, verbose=1)

    preds = (model.predict(X) > 0.5).astype(int).flatten()
    acc = accuracy_score(y, preds)
    print('Acc:', acc)
    print(classification_report(y, preds))

    MODEL_DEMO = ARTIFACT_DIR / 'spam_lstm_model_demo.h5'
    TOKENIZER_DEMO = ARTIFACT_DIR / 'tokenizer_demo.pkl'

    model.save(str(MODEL_DEMO))
    with open(TOKENIZER_DEMO, 'wb') as f:
        pickle.dump(tokenizer, f)

    with open(ARTIFACT_DIR / 'experiment_report.txt', 'w', encoding='utf8') as fo:
        fo.write(f'Acc_demo: {acc:.4f}\n')
        fo.write('Classification report:\n')
        fo.write(classification_report(y, preds))

    print('Demo guardada en:', ARTIFACT_DIR)


if __name__ == '__main__':
    main()
