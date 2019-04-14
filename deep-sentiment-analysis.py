import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant


EMBEDDING_DIM = 300

df = pd.DataFrame()
df = pd.read_csv('/media/aucan/Data1/Datasets/aclimdb/movie_data.csv', encoding='utf-8')

train_x = df.loc[:24999, 'review'].values
train_y = df.loc[:24999, 'sentiment'].values
test_x = df.loc[25000:, 'review'].values
test_y = df.loc[25000:, 'sentiment'].values

tknzr = Tokenizer()
total_reviews = train_x + test_x
tknzr.fit_on_texts(total_reviews)

max_length = max([len(s.split()) for s in total_reviews])

word_index = tknzr.word_index
vocab_size = len(word_index) + 1

tokens_train_x = tknzr.texts_to_sequences(train_x)
tokens_test_x = tknzr.texts_to_sequences(test_x)

pad_train_x = pad_sequences(tokens_train_x, maxlen=max_length, padding='post')
pad_test_x = pad_sequences(tokens_test_x, maxlen=max_length, padding='post')

embedding_index = {}
f = open('/media/aucan/Data1/embeddings/crawl-300d-2M-subword.vec', encoding='utf-8')
for line in f:
    values = line.split()
    if len(values) >= EMBEDDING_DIM:
        word = values[0]
        if word_index.get(word) is not None:
            coefs = np.asarray(values[1:])
            embedding_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > max_length:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_index = None

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),
                    input_length=max_length, trainable=False))
model.add(GRU(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(pad_train_x, train_y, batch_size=256, epochs=10, validation_data=(pad_test_x, test_y), verbose=1)
