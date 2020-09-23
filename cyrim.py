import streamlit as st
import pandas as pd
import numpy as np
import nltk
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
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

st.title("😈CYRIM: Fake News Detector😈")
st.sidebar.title("News Visualization Tools")

stop_words = stopwords.words("english")
stop_words.extend(['from','subject','re','edu','use'])

def preprocess(text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
      result.append(token)
  return result

@st.cache(allow_output_mutation=True)
def fetch_data():
	return pd.read_csv('DataFrame.csv'),tf.keras.models.load_model('my_model.h5')  

df,model = fetch_data()

#Necessities
total_words = 108704                                                #paste outputs of previous file
maxlength = 4405													#paste outputs of previous file
temp_accuracy = 0.9973273942093541									#paste outputs of previous file

x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)

def get_accuracy():
	x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)
	tokenizer = Tokenizer(num_words = total_words)
	tokenizer.fit_on_texts(x_train)
	train_sequences = tokenizer.texts_to_sequences(x_train)
	test_sequences = tokenizer.texts_to_sequences(x_test)
	padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
	padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post')
	y_train = np.asarray(y_train)
	pred = model.predict(padded_test)
	prediction = []
	for i in range(len(pred)):
		if pred[i].item() > 0.5:
			prediction.append(1)
		else:
			prediction.append(0)

	accuracy = accuracy_score(list(y_test), prediction)
	return accuracy


def convert_input(user_input):
	processed_input = preprocess(user_input)
	processed_input = " ".join(processed_input)
	x = []
	x.append(processed_input)
	processed_input = pd.Series(x)
	processed_sequences = tokenizer.texts_to_sequences(processed_input)
	processed_test = pad_sequences(processed_sequences,maxlen = 40, truncating = 'post')
	return processed_test


user_input = st.text_area("Enter your News here")

if user_input !="":
	processed_input = convert_input(user_input)
	prediction = model.predict(processed_input)
	if prediction.item() > 0.5:                                     
		st.markdown("## Warning: Fake News Detected 👎")
		st.write("Your News: ")
		st.write(user_input)
	else:
		st.markdown("## Hurrah: Real News Detected 👍")
		st.markdown("Your News: ")
		st.write(user_input)


st.sidebar.subheader("Accuracy Score")
if st.sidebar.checkbox("Re-Run Accuracy",False,key=1):
	accuracy = get_accuracy()
	st.sidebar.markdown(accuracy)
else:
	st.sidebar.markdown(temp_accuracy)


st.sidebar.subheader("Show random News")

rn = -1
random_news = st.sidebar.radio('Type of News',('Real News','Fake News','Hide News'),index=2,key=2)
if random_news == 'Fake News':
	st.markdown("## Random Fake News 🧐")
	rn = 1
elif random_news == 'Real News':
	st.markdown("## Random Real News 🤗")
	rn = 0

if rn != -1:
	st.markdown(df.query('isfake == @rn')[["original"]].sample(n=1).iat[0,0])

def histogram():
	plt.figure(figsize=(8,8))
	sns.countplot(y = "isfake", data = df)
	st.pyplot()

@st.cache(suppress_st_warning=True)
def wordcloud():	
	wc1 = WordCloud(max_words = 2000, width = 1600, height = 800).generate(" ".join(df[df['isfake'] == 0].clean_joined))
	wc2 = WordCloud(max_words = 2000, width = 1600, height = 800).generate(" ".join(df[df['isfake'] == 1].clean_joined))
	return wc1,wc2

@st.cache(suppress_st_warning=True)
def piechart():
	fake_count = df['isfake'].value_counts()
	fake_count = pd.DataFrame({'isfake':fake_count.index,'Count':fake_count.values})
	fig = px.pie(fake_count,values='Count',names='isfake',title="Distribution of Fake News and Real News")
	return fig

def s_countanalysis():
	plt.figure(figsize=(8,8))
	sns.countplot(y = "subject", data = df)
	st.pyplot()

@st.cache(suppress_st_warning=True)
def ntword():
	fig = px.histogram(x = [len(nltk.word_tokenize(x)) for x in df.clean_joined], nbins = 100)
	return fig

st.sidebar.subheader("Real-vs-Fake")
news_type = st.sidebar.selectbox('Select Visualization',('Histogram','Word Cloud','Pie Chart'),key=3)
if st.sidebar.checkbox("Show",False,key=4):
	if news_type == 'Histogram':
		histogram()
	if news_type == 'Word Cloud':
		wc1,wc2 = wordcloud()
		st.markdown("Word Cloud: Real News")
		plt.figure(figsize = (20,20))
		plt.imshow(wc1, interpolation='bilinear')
		st.pyplot()
		plt.figure(figsize = (20,20))
		st.markdown("Word Cloud: Fake News")
		plt.imshow(wc2, interpolation='bilinear')
		st.pyplot()

	if news_type == 'Pie Chart':
		fig = piechart()
		st.plotly_chart(fig)

st.sidebar.subheader("Subject-Count Analysis")
if st.sidebar.checkbox("Show",False,key=5):
	st.subheader("Subject-Count Analysis")
	s_countanalysis()

st.sidebar.subheader("News-TotalWords Distribution")
if st.sidebar.checkbox("Show",False,key=6):
	st.subheader("News-TotalWords Distribution")
	fig = ntword()
	st.plotly_chart(fig)


st.sidebar.subheader("Confusion Matrix")
if st.sidebar.checkbox("Show",False,key=7):
	st.subheader("Confusion Matrix")
	st.image('cmatrix.png')
