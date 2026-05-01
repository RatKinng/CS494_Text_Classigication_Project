# import pandas as pd

# # reddit = load_dataset("json", data_files="data/reddit.json")
# from datasets import load_dataset

# reddit = load_dataset(
#     "roskoN/dstc8-reddit-corpus",
#     trust_remote_code=True
#
#commented lines above are to hold for loading in the corpuses from hugging face later on

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

filepath_dict = {
    'yelp':   'sentiment_analysis/yelp_labelled.txt',
    'amazon': 'sentiment_analysis/amazon_cells_labelled.txt',
    'imdb':   'sentiment_analysis/imdb_labelled.txt'
}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    df_list.append(df)

df = pd.concat(df_list)


def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers
    from tensorflow.keras import Input

    model = Sequential()

    model.add(Input(shape=(maxlen,)))

    model.add(layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    ))

    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


epochs = 20
embedding_dim = 50
maxlen = 100

for source, frame in df.groupby('source'):
    print("\n" + "="*50)
    print(f"Running dataset: {source}")
    print("="*50)

    sentences = frame['sentence'].values
    y = frame['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    # Tokenization
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = min(5000, len(tokenizer.word_index) + 1)

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # Grid params
    param_grid = {
        "model__num_filters": [32, 64, 128],
        "model__kernel_size": [3, 5, 7],
        "model__vocab_size": [vocab_size],
        "model__embedding_dim": [embedding_dim],
        "model__maxlen": [maxlen],
    }

    model = KerasClassifier(
        model=create_model,
        epochs=epochs,
        batch_size=10,
        verbose=0
    )

    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=4,
        verbose=1,
        n_iter=5
    )

    grid_result = grid.fit(X_train, y_train)

    test_accuracy = grid.score(X_test, y_test)

    print("\nRESULTS:")
    print("Best CV Score:", grid_result.best_score_)
    print("Best Params:", grid_result.best_params_)
    print("Test Accuracy:", test_accuracy)