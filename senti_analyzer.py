import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
def senti_analyzer(user_id):    
    # Initialize the VADER sentiment analyzer
    

    # Read the cleaned data from 'cleanedDataWithoutURLs.csv' into a DataFrame
    df = pd.read_csv('cleanedData1.csv')

    # User ID for which you want to analyze sentiment
    # user_id = 1013187241  # Replace with the desired user_id

    # Filter the DataFrame for rows with the specified user_id
    user_df = df[df['user_id'] == user_id].copy()  # Make a copy to avoid the SettingWithCopyWarning
    # Apply sentiment analysis to the 'post_text' column for the user's tweets
    user_df['Sentiment Scores'] = user_df['post_text'].apply(get_sentiment_scores)

    # Extract sentiment scores
    user_df['Compound'] = user_df['Sentiment Scores'].apply(lambda x: x['compound'])
    user_df['Positive'] = user_df['Sentiment Scores'].apply(lambda x: x['pos'])
    user_df['Negative'] = user_df['Sentiment Scores'].apply(lambda x: x['neg'])
    user_df['Neutral'] = user_df['Sentiment Scores'].apply(lambda x: x['neu'])

    # Store the sentiment scores in an array
    sentiment_scores_array = user_df['Compound'].values

    # Display the DataFrame with sentiment scores for the specified user
    # print(user_df)

    # Display the array of sentiment scores
    sum_of_senti = sum(sentiment_scores_array)
    avg_sentiment = sum_of_senti/len(sentiment_scores_array)
    return avg_sentiment


# Function to get sentiment scores
def get_sentiment_scores(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment