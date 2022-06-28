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
from click import Option
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
from sympy import N

# Vectorizer
news_vectorizer = open("resources/TfidfVectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home","Prediction", "Information","Team"]
	with st.sidebar:
		st.title('Menu') #ctreating a menu function on the sidebar 
		selection = st.radio("Choose Option", options)#creating a radio button for  different models
	# Bulding the home page	
	if selection == "Home":
		st.info("Many companies are built around lessening one's environmental impact or carbon footprint.\n"
		"They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals.\n"
		"They would like to determine how people perceive climate change and whether or not they believe it is a real threat.\n"
		"This would add to their market research efforts in gauging how their product/service may be received")

		st.info ("Variable definitions.\n" 
		"sentiment: Sentiment of tweet\n"
		"message: Tweet body \n"
		"tweetid: Twitter unique id")

	
	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","enter your sentence / tweet here")
		st.info('Which classifier  to run to get the results')
		option = st.radio('ML models',('Logistic_Regression_classifier','KNNeighbor_sClassifier','RandomForestClassifier_Classifier'))
		# Transforming user input with vectorizer
		vect_text = tweet_cv.transform([tweet_text]).toarray()
		# Load your .pkl file with the model of your choice + make predictions
		# Try loading in multiple models to give the user a choice
		if option  =='Logistic_Regression_classifier':
			logistic_regression_classifier = joblib.load(open(os.path.join("resources/LogisticRegression_model.pkl"),"rb"))
			prediction = logistic_regression_classifier.predict(vect_text)
		elif option  =='KNNeighbors_Classifier':
			KNNeighbors_Classifier = joblib.load(open(os.path.join("resources/KNeighborsClassifier_model.pkl"),"rb"))
			prediction = KNNeighbors_Classifier.predict(vect_text)
		elif option  =='RandomForestClassifier_Classifier':
			RandomForestClassifier_Classifier= joblib.load(open(os.path.join("resources/RandomForestClassifier_model.pkl"),"rb"))
			prediction =RandomForestClassifier_Classifier.predict(vect_text)	
		# When model has successfully run, will print prediction
		# You can use a dictionary or similar structure to make this output
		# more human interpretable.
		st.success("Text Categorized as: {}".format(prediction))
		#creating a category markdowns
		st.info('Text Categories')
		#different category class
		st.markdown('**1** = **Pro/Posetive**.')
		st.markdown('**2**= **News**.')
		st.markdown('**-1** = **Anti/Negative**.')
		st.markdown('**0** = **Neutral**.')
	# buidling a team page 	
	if selection == "Team":
		#team  name 
		st.info("Team CBB3")
		#team mates names 
		st.markdown("Elewani Tshikovhi: Team Learder")
		st.markdown("Katlego Maponya: Team coordinater")
		st.markdown("Sinethemba Nongqoto")
		st.markdown("Musa Mashaba")
		st.markdown("Desree Maleka")
		st.markdown("Zothandwa Kunene")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
