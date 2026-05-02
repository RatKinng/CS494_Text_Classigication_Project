import pandas as pd
import numpy as np
import re

from datasets import load_dataset

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Input

from scikeras.wrappers import KerasClassifier

print("Loading datasets...")

#load reddit data. Twitter is much smaller, so take same amount of samples
reddit_ds = load_dataset("sentence-transformers/reddit-title-body", split="train[:5342]")
reddit_df = pd.DataFrame(reddit_ds)
reddit_df["sentence"] = reddit_df["body"].astype(str)
reddit_df = reddit_df[["sentence"]]
reddit_df["label"] = 0 

#load twitter data
twitter_ds = load_dataset("GateNLP/broad_twitter_corpus", split="train[:5342]")  # all available
twitter_df = pd.DataFrame(twitter_ds)

#twitter has weird names, so auto detect and finish loading
text_col = [c for c in twitter_df.columns if twitter_df[c].dtype == "object"][0]
twitter_df["sentence"] = twitter_df[text_col].astype(str)
twitter_df = twitter_df[["sentence"]]
twitter_df["label"] = 1  

print(f"Reddit samples: {len(reddit_df)}, Twitter samples: {len(twitter_df)}")

#Trying to make it harder to tell them apart by removing r/, u/, and @
def clean_text(text):
    text = str(text).lower()                         
    text = re.sub(r"http\S+", "", text)          
    text = re.sub(r"r/\w+", "", text)            
    text = re.sub(r"u/\w+", "", text)            
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"@\w+", "", text)             
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)   
    text = re.sub(r"\s+", " ", text).strip()     
    return text

#Actually clean up the sets
reddit_df["sentence"] = reddit_df["sentence"].apply(clean_text)
twitter_df["sentence"] = twitter_df["sentence"].apply(clean_text)

#make sure no duplicate sentences
reddit_df = reddit_df.drop_duplicates(subset=["sentence"])
twitter_df = twitter_df.drop_duplicates(subset=["sentence"])

#balancethe sets so there's not more of one than the other
min_size = min(len(reddit_df), len(twitter_df))
reddit_df = reddit_df.sample(min_size, random_state=42)
twitter_df = twitter_df.sample(min_size, random_state=42)

#combine datasets and mix them up so the model can't detect if one is real vs the other
df = pd.concat([reddit_df, twitter_df]).sample(frac=1, random_state=42).reset_index(drop=True)

sentences = df["sentence"].values
labels = df["label"].values

#keep balance, small test set
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.30, random_state=1000, stratify=labels)

max_words = 10000
maxlen = 50  
embedding_dim = 50

#tokenize the words
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences_train)

#create tokenized sets
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

#find vocab size
vocab_size = min(max_words, len(tokenizer.word_index) + 1)

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#creating the actual model
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential() #uses a sequential model
    #add the shape and layers of the model, set it up with all the values
    model.add(Input(shape=(maxlen,)))

    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))

    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

print("Running hyperparameter search...")

#different parameters for the model
param_grid = {
    "model__num_filters": [32, 64, 128],
    "model__kernel_size": [3, 5, 7],
    "model__vocab_size": [vocab_size],
    "model__embedding_dim": [embedding_dim],
    "model__maxlen": [maxlen],
}

#set the model
model = KerasClassifier(model=create_model, epochs=5,batch_size=32, verbose=0)

#set the grid
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, verbose=1, n_iter=5)

#get the result
grid_result = grid.fit(X_train, y_train)

print("\nRESULTS:")
print("Best CV Score:", grid_result.best_score_)
print("Best Params:", grid_result.best_params_)

#get test accuracy
test_accuracy = grid.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)
