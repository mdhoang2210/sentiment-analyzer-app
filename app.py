import streamlit as st
import pandas as pd 
import plotly.express as px
from langchain_ollama.llms import OllamaLLM



st.title("🥗 Customer Review Sentiment Analyzer")
st.markdown("This app analyzes the sentiment of customer reviews to gain insights into their opinions.")


# Import reviews.csv file
# import pandas as pd 
# df = pd.read_csv('reviews.csv')
# st.write(df)


# CSV file uploader
uploaded_file = st.file_uploader(
    'Upload a CSV file with restaurant reviews',
    type = ['csv']
)

# Once the user uploads a csv file:
if uploaded_file is not None:
    #Read the file
    reviews_df = pd.read_csv(uploaded_file)

    #Check if the data has a text column
    text_columns = reviews_df.select_dtypes(include='object').columns

    if len(text_columns) == 0:
        st.error('No text columns found in the uploaded file.')
    
    #Show a dropdown menu to select the review column
    review_column =  st.selectbox(
        'Select the column with the customer reviews',
        text_columns
    )


def classify_sentiment_ollama(review_text):

    # Load the model
    llm = OllamaLLM(model="llama3.2")  # Adjust the model name as needed, for example "deepseek-r1:7b" or "mistral"

    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    result = llm.invoke(prompt)

    return result

#Analyze the sentiment of the selected column
reviews_df['sentiment'] = reviews_df[review_column].apply(classify_sentiment_ollama)


#Display the sentiment distribution in metrics in 3 columns: Positive, Negative, Neutral
#Make the strings in the sentiment column title 
reviews_df['sentiment'] = reviews_df['sentiment'].str.title()
reviews_df['sentiment'] = reviews_df['sentiment'].str.replace('.', '')
sentiment_counts = reviews_df['sentiment'].value_counts()
st.write(reviews_df)
st.write(sentiment_counts)

#Create 3 columns to display the 3 metrics
col1, col2, col3 = st.columns(3)

with col1:
    #Show the number of positive reviews and the percentage
    positive_count = sentiment_counts.get('Positive', 0)
    st.metric('Positive', 
              positive_count, 
              f'{positive_count/len(reviews_df) * 100:.2f}%')
    
with col2:
    #Show the number of negative reviews and the percentage
    negative_count = sentiment_counts.get('Negative', 0)
    st.metric('Negative', 
              negative_count, 
              f'{negative_count/len(reviews_df) * 100:.2f}%')
    
with col3:
    #Show the number of neutral reviews and the percentage
    neutral_count = sentiment_counts.get('Neutral', 0)
    st.metric('Neutral', 
              neutral_count, 
              f'{neutral_count/len(reviews_df) * 100:.2f}%')


#Display pie chart
fig = px.pie(
    values = sentiment_counts.values,
    names = sentiment_counts.index, 
    title = 'Sentiment Distribution')
st.plotly_chart(fig)

# Example usage
# Write the results to the app
#st.write(classify_sentiment_ollama(review_column))