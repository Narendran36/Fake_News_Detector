import streamlit as st
import pandas as pd
import numpy as np
import nltk
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Bidirectional
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
nltk.download('punkt')                                                                                #(comment if installed)
nltk.download("stopwords")                                                                            #(comment if installed)


stop_words = stopwords.words("english")
stop_words.extend(['from','subject','re','edu','use'])

def preprocess(text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
      result.append(token)
  return result


def load_data():
	df_true = pd.read_csv("True.csv")
	df_fake = pd.read_csv("Fake.csv")
	df_true['isfake'] = 0
	df_fake['isfake'] = 1
	df = pd.concat([df_true,df_fake]).reset_index(drop = True)
	df.drop(columns = ['date'], inplace = True)
	df['original'] = df['title'] + ' ' + df['text']
	df['clean'] = df['original'].apply(preprocess)
	df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x)) #join the words into a string

	#Obtain all unique words present in dataset
	list_of_words = []
	for i in df['clean']:
	  for j in i:
	    list_of_words.append(j)
	total_words = len(list(set(list_of_words)))

	#length of maximum document (News) will be needed to create word embedding, ie the maximum number of words in a doc
	maxlength = -1
	for doc in df.clean_joined:
	  tokens = nltk.word_tokenize(doc)
	  if (maxlength < len(tokens)):
	    maxlength = len(tokens)

	#Data preparation tokenization and padding
	x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)
	tokenizer = Tokenizer(num_words = total_words)
	tokenizer.fit_on_texts(x_train)
	train_sequences = tokenizer.texts_to_sequences(x_train)
	test_sequences = tokenizer.texts_to_sequences(x_test)
	padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
	padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post')

	#Build and Train the model

	# Sequential Model
	model = Sequential()

	# embeddidng layer
	model.add(Embedding(total_words, output_dim = 128))



	# Bi-Directional RNN and LSTM
	model.add(Bidirectional(LSTM(128)))

	# Dense layers
	model.add(Dense(128, activation = 'relu'))
	model.add(Dense(1,activation= 'sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	y_train = np.asarray(y_train)

	# train the model
	model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)

	# make prediction
	pred = model.predict(padded_test)

	# if the predicted value is >0.5 it is fake else it is real
	prediction = []
	for i in range(len(pred)):
	    if pred[i].item() > 0.5:
	        prediction.append(1)
	    else:
	        prediction.append(0)

	accuracy = accuracy_score(list(y_test), prediction)


	cm = confusion_matrix(list(y_test), prediction)
	plt.figure(figsize = (25, 25))
	sns.heatmap(cm, annot = True)
	plt.savefig('cmatrix.png', bbox_inches='tight')                               # comment if cmatrix.png is generated
	st.pyplot()
	
	return df,accuracy,model,tokenizer,total_words,maxlength

df,accuracy,model,tokenizer,total_words,maxlength = load_data()

df.to_csv('DataFrame.csv')										                   # comment if DataFrame.csv is generated

st.write("Assign accuracy = ",accuracy)                             # Assign Values to these variables in next program
st.write("Assign total_words = ",total_words)
st.write("Assign maxlength = ",maxlength)


model.save('my_model.h5')										                   # comment if my_model.h5 is generated
