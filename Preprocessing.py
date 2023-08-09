import pandas as pd
import re
from tkinter import filedialog

class Preprocessing:

    alay_dict = pd.read_csv('data/new_kamusalay.csv', encoding='latin-1', header=None)
    alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})
    alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
  
    def cleaning(self, text):
        text = re.sub(r'http\S+', ' ', text) # hapus link
        text = re.sub('\n', ' ', text)  # Remove every '\n'
        text = re.sub('\r', ' ', text)  # Remove every '\r'
        text = re.sub('(?i)rt', ' ', text)  # Remove every retweet symbol
        text = re.sub('@[^\s]+[ \t]', '', text)  # Remove every username
        text = re.sub('(?i)user', '', text)  # Remove every username
        text = re.sub(r'\\x..', ' ', text)  # Remove every emoji
        text = re.sub('(?i)url', ' ', text)  # Remove every url
        text = re.sub('  +', ' ', text)  # Remove extra spaces
        # Remove characters repeating more than twice
        text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text) #hapus semua karakter selain huruf dan spasi
        text = re.sub(r'[0-9]+', ' ', text) # hapus angka
        text = text.strip() #hapus spasi dari sisi kiri-kanan teks
        return text

    def case_folding(self, text):
        return text.lower()

    def normalize_alay(self, text):
        return ' '.join([self.alay_dict_map[word] if word in self.alay_dict_map else word for word in text.split(' ')])

    def preprocess(self, text):
        text = self.cleaning(text)  # 1
        text = self.case_folding(text)  # 2
        text = self.normalize_alay(text)  # 3
        return text

    def prosesData(self,file_input):
        # Membaca dataset dari file input        
        data = pd.read_csv(file_input)
        
        # Melakukan preprocessing
        data['original_text'] = data['original_text'].apply(self.preprocess)

        return data
    
    def inputDataset(self):
        file_input = filedialog.askopenfilename()
        return file_input