# import pandas as pd

# # reddit = load_dataset("json", data_files="data/reddit.json")
# from datasets import load_dataset

# reddit = load_dataset(
#     "roskoN/dstc8-reddit-corpus",
#     trust_remote_code=True
# )

# filepath_dict = {'yelp':   'sentiment_analysis/yelp_labelled.txt',
#                  'amazon': 'sentiment_analysis/amazon_cells_labelled.txt',
#                  'imdb':   'sentiment_analysis/imdb_labelled.txt'}

# df_list = []
# for source, filepath in filepath_dict.items():
#     df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
#     df['source'] = source  # Add another column filled with the source name
#     df_list.append(df)

# df = pd.concat(df_list)
# print(df.iloc[0])

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression

# for source in df['source'].unique():
#     df_source = df[df['source'] == source]
#     sentences = df_source['sentence'].values
#     y = df_source['label'].values

#     sentences_train, sentences_test, y_train, y_test = train_test_split(
#         sentences, y, test_size=0.25, random_state=1000)

#     vectorizer = CountVectorizer()
#     vectorizer.fit(sentences_train)
#     X_train = vectorizer.transform(sentences_train)
#     X_test  = vectorizer.transform(sentences_test)

#     classifier = LogisticRegression()
#     classifier.fit(X_train, y_train)
#     score = classifier.score(X_test, y_test)
#     print('Accuracy for {} data: {:.4f}'.format(source, score))

# from keras.models import Sequential
# from keras import layers

# input_dim = X_train.shape[1]  # Number of features

# model = Sequential()
# model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

# history = model.fit(X_train, y_train, epochs=100, verbose=False, validation_data=(X_test, y_test), batch_size=10)

# from keras.backend import clear_session
# clear_session()

# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))

# import matplotlib.pyplot as plt
# plt.style.use('ggplot')

# def plot_history(history):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     x = range(1, len(acc) + 1)

#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, acc, 'b', label='Training acc')
#     plt.plot(x, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(x, loss, 'b', label='Training loss')
#     plt.plot(x, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()

# plot_history(history)

# from tensorflow.keras.preprocessing.text import Tokenizer

# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(sentences_train)

# X_train = tokenizer.texts_to_sequences(sentences_train)
# X_test = tokenizer.texts_to_sequences(sentences_test)

# vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

# print(sentences_train[2])
# print(X_train[2])

# from keras.preprocessing.sequence import pad_sequences

# maxlen = 100

# X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
# X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# print(X_train[0, :])


# from keras.models import Sequential
# from keras import layers

# from keras.preprocessing.sequence import pad_sequences

# data = pad_sequences(sequences, maxlen=maxlen)

# embedding_dim = 50

# model = Sequential()
# model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))
# # model.add(layers.Flatten())
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# print(vocab_size)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# -----------------------
# LOAD DATA
# -----------------------
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

# -----------------------
# PICK ONE DATASET (IMPORTANT)
# -----------------------
df = df[df['source'] == 'imdb']  # simplify debugging

sentences = df['sentence'].values
y = df['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000
)

# -----------------------
# TOKENIZATION
# -----------------------
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1

# -----------------------
# PADDING
# -----------------------
maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# -----------------------
# MODEL
# -----------------------
embedding_dim = 50

# model = Sequential()
# model.add(layers.Embedding(
#     input_dim=vocab_size,
#     output_dim=embedding_dim
# ))

# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()
# model = Sequential()

# model.add(input(shape=(maxlen)))

# model.add(layers.Embedding(
#     input_dim=vocab_size,
#     output_dim=embedding_dim
# ))

# model.add(layers.GlobalMaxPooling1D())

# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

model = Sequential()

model.add(layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_shape=(maxlen,)
))

model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------
# TRAIN
# -----------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# -----------------------
# EVALUATE
# -----------------------
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# -----------------------
# PLOT
# -----------------------
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, acc, label='Train')
    plt.plot(x, val_acc, label='Val')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, label='Train')
    plt.plot(x, val_loss, label='Val')
    plt.title('Loss')
    plt.legend()

    plt.show()

# plot_history(history)

import numpy as np

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix