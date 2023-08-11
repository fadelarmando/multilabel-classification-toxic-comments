import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

class Model:

    def __init__(self,dropout = 0.2, hidden_units = 128,  recurrent_dropout = 0.3,  epochs = 20,   batch_size = 64):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.epochs = epochs
        self.batch_size = batch_size
    
    def buildModel(self,data_preprocessed):
        # Membaca dataset dari file
        data = data_preprocessed
        
        # Mengambil data comment
        data_comments = data['original_text']
        
        # Daftar label
        labels = ['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']
        labels_value = data[labels].values

        # Split data menjadi data latih  dan data test 
        comments_train, comments_test, labels_train, labels_test = train_test_split(data_comments, labels_value, test_size=0.2, random_state=42)

        # Tokenisasi data train
        tokenizer = Tokenizer(oov_token='<UNK>', filters='')
        tokenizer.fit_on_texts(comments_train)
        print('-------Tokenizing selesai-------')
        
        # Save token
        # with open(f'model_data/token/tokenizer-{timestamp}.pickle', 'wb') as handle:
        #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Padding pada data
        def get_sequences(token, data, max_train=100):
            sequences = token.texts_to_sequences(data)
            padded_sequences = pad_sequences(sequences, maxlen=max_train)
            return padded_sequences

        padded_train_sequences = get_sequences(tokenizer, comments_train)
        padded_test_sequences = get_sequences(tokenizer, comments_test)
        print("-------Padding Selesai-------")
        
        # Melakukan word2vec
        model_path = 'model_data/word2vec/word2vec_400/word2vec_400dim.txt'
        word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

        vocab_size = len(tokenizer.word_index) + 1
        vector_size = word2vec_model.vector_size

        print(vocab_size)
        print(len(word2vec_model))
        
        # Membuat variable yang isinya vektor pada kata
        embedding_matrix = np.zeros((vocab_size, vector_size))
        for word, i in tokenizer.word_index.items():
            if word in word2vec_model:
                embedding_matrix[i] = word2vec_model[word]

        print("-------Embedding Matrix selesai dibuat-------")
        print('embedding matrix size',embedding_matrix.shape)

        print("-------Melakukan Training-------")
        # Membangun model LSTM dengan embedding layer
        model = Sequential()
        model.add(Embedding(vocab_size, vector_size, weights=[embedding_matrix], input_length=100, trainable=False))
        model.add(LSTM(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))
        model.add(Dense(len(labels), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Melatih model dengan data latih
        model.fit(padded_train_sequences, np.array(labels_train), validation_split=0.1, epochs=self.epochs, batch_size=self.batch_size) 
        
        # model.save(f"model_data/model/model-{timestamp}")            
        print("-------Training selesai-------")

        # Evaluasi model dengan data pengujian
        print("-------Melakukan Evaluasi-------")
        test_pred = model.predict(padded_test_sequences)
        hamming_loss_score = hamming_loss(np.array(labels_test), np.round(test_pred))

        # Menghitung nilai Hamming Loss untuk setiap label
        hamming_loss_per_label = []
        for i in range(len(labels)):
            label_true = labels_test[:, i]
            label_pred = np.round(test_pred[:, i])
            loss = hamming_loss(label_true, label_pred)
            hamming_loss_per_label.append(loss)

        # Menampilkan nilai Hamming Loss per label
        for i, label in enumerate(labels):
            print(f"Hamming Loss for {label}: {hamming_loss_per_label[i]}")
        
        print()
        print('Hamming Loss:', hamming_loss_score)
        print('-----------------------------------')
        print(labels_test)
        print(np.round(test_pred))
        print(np.sum(np.abs(labels_test - np.round(test_pred)), axis=0))
        # Menghitung jumlah data kesalahan per label
        error_per_column = np.sum(np.abs(labels_test - np.round(test_pred)), axis=0)
        print('-----------------------------------')
        
        # Menampilkan jumlah kesalahan per label
        for i in range(len(error_per_column)):
            print(f"Jumlah kesalahan pada {labels[i]} = {int(error_per_column[i])}")
        
        eval_results = []
        for i, label in enumerate(labels):
            eval_results.append(hamming_loss_per_label[i])
            
        eval_results.append(hamming_loss_score)
        
        # Create a DataFrame from the lists
        label_data = labels
        label_data.append('rata-rata')
        df = pd.DataFrame({'Kategori': label_data, 'Hamming Loss': eval_results})

        return df