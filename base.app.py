"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from streamlit_option_menu import option_menu


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
# Data dependencies
import pandas as pd
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import time
import requests



# Vectorizer
news_vectorizer = open("resources/vectorizer2.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Creates a main title and subheader on your page -
# these are static across all pages
st.title("DataLink Tweet Classifer APP")
st.subheader("Climate change tweet classification")
st.markdown("This application is a streamlit dashboard to analyze and classify predictions of Tweets sentiments")
# Creating sidebar with selection box -
# you can create multiple pages this way

with st.sidebar:
		#st.sidebar.image(Image.open('resources/imgs/Logo3.png'))
		selected = option_menu(
			menu_title = 'Navigation Menu',
			menu_icon="list", 
			options = ['Home', 'Data Analysis', 'Models', 'Contact Us'],
			icons = [ "house","bar-chart-line", "gear", "envelope"],
			)


# The main function where we will build the actual app
if selected == 'Home':
	st.subheader('Tweets Classifier App')
	image = Image.open('resources/imgs/Main_pic1.jpg')
	st.image(image, caption='TEAM JS6', use_column_width=True)
	# st.image(image name, width = None)
	st.write("Climate change refers to long-term shifts in temperatures and weather patterns."
		"In this APP we Can determine how people perceive climate change and whether or not a "
		"person believes in climate change based on their tweet data.")


if selected == 'Data Analysis':
	st.info("##### For this section we will explore the distribution of the data using different visualisation plots")
	st.write("---")
	bar_graph = Image.open('resources/imgs/Bar_Graph.png')
	st.write("##### Bar graph showing tweets per sentiment")
	st.image(bar_graph, width = None)

	st.write("---")

	pie_chart = Image.open('resources/imgs/Pie_chart.png')
	st.write("##### Pie Chart showing percentage tweets per sentiment")
	st.image(pie_chart, width = None)

	st.write("---")

	wordcloud = Image.open('resources/imgs/News.png')
	st.write("##### Wordcloud showing most popular tweets For News sentiment")
	st.image(wordcloud, width = None)

	st.write("---")

	wordcloud2 = Image.open('resources/imgs/Neutral.png')
	st.write("##### Wordcloud showing most popular tweets For Neutral sentiment")
	st.image(wordcloud2, width = None)

	st.write("---")

	wordcloud3 = Image.open('resources/imgs/Pro.png')
	st.write("##### Wordcloud showing most popular tweets For Pro sentiment")
	st.image(wordcloud3, width = None)

	st.write("---")

	wordcloud4 = Image.open('resources/imgs/Anti.png')
	st.write("##### Wordcloud showing most popular tweets For Anti sentiment")
	st.image(wordcloud4, width = None)

	st.write("---")
	st.write("The most popular words in all 4 classes are Climate Change, Global Warming."
			"There is some unnecessary data. The words http, https, website, co & RT are prominent, but will not assist us in our classification."
			"Thee top 5 words in all classes are the same, except for the News class, in which the word 'Trump' features prominently.")
	st.write("---")

	performance = Image.open('resources/imgs/Performance.png')
	st.write("##### Bar Chart showing Time taken to train and Performance of the Model")
	st.image(performance, width = None)
	st.write("From the results table & bar chart, we can see the models that"
	"perform best are the Logistic Regression & Linear Support Vector Machines(SVM). ")

	st.write("---")


if selected == "Models":
	left_column, right_column = st.columns(2)
	with left_column:
		st.subheader('Climate Change Tweet Classifier')

		st.info('Make Predictions of your Tweet(s) using our ML Model')

		tweet_text = st.text_area('Enter Text (max. 140 characters):') ##user entering a single text to classify and predict
		all_ml_models = ["Logistic Regression", "Random Forest Classifier", "MultinomialNB", "KNeighbors Classifier", "Linear SVC"]
		model_choice = st.selectbox("Choose ML Model",all_ml_models)

		prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}

		if st.button('Classify'):

			# Getting the predictions
			def get_keys(val,my_dict):
				for key,value in my_dict.items():
					if val == value:
						return key

			vect_text = tweet_cv.transform([tweet_text]).toarray()
			

			if model_choice == 'Logistic Regression':
				predictor = joblib.load(open(os.path.join("resources/lr.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice =='Random Forest Classifier':
				predictor = joblib.load(open(os.path.join("resources/rfc.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice == 'MultinomialNB':
				predictor = joblib.load(open(os.path.join("resources/mnb.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice == 'KNeighbors Classifier':
				predictor = joblib.load(open(os.path.join("resources/knn.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model_choice == 'Linear SVC':
				predictor = joblib.load(open(os.path.join("resources/svc.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			final_result = get_keys(prediction,prediction_labels)
			st.success("Tweets Categorized as: {}".format(final_result))
	
	with right_column:

		mail = Image.open('resources/imgs/Logo3.png')
		st.image(mail, width = None)

		def load_lottieurl(url: str):
			r = requests.get(url)
			if r.status_code != 200:
				return None
			return r.json()

		animation = "https://assets4.lottiefiles.com/packages/lf20_b88nh30c.json"
		loading = load_lottieurl(animation)
		st_lottie(loading, height =250, key = 'coding2')
		#load = Image.open('resources/imgs/load.jpg')
		#st.image(load, width = None)

		mail = Image.open('resources/imgs/tweeter.png')
		st.image(mail, width = None)



if selected == "Contact Us":
	left_column, right_column = st.columns(2)
	with left_column:
		st.info("### App Developers")
		st.info("Contact details in case you have any query")
		st.write("---")
		st.write("Moromo Mathobela : bonifasiusmathobela@gmail.com")
		st.write("Mbali Mnguni : mbali.mnguni6@gmail.com")
		st.write("Andy Sumo : andysithole87@gmail.com")
		st.write("Amantle Moepeng : amantlemoepeng5@gmail.com")
		st.write("Mengezi Sibeko : sibekomengezi5@gmail.com")
		st.write("Velly Shishoka : vellyseshoka04@gmail.com")


	with right_column:

		mail = Image.open('resources/imgs/Logo3.png')
		st.image(mail, width = None)
		mail2 = Image.open('resources/imgs/email2.png')
		st.image(mail2, width = None)

st.sidebar.title("About")
st.sidebar.info(
        	"This app enables you to predict tweets sentiments using our various Machine Learning Models. "

       		 "It is maintained by [DataLink Pty Ltd ](https://www.linkedin.com/in/aliavnicirik/). "
        	)
