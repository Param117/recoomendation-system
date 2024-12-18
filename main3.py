import pickle
import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import os
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fetch movie poster from TheMovieDB API
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    else:
        return "https://via.placeholder.com/500"

# Recommendation function
def recommend(movie, language_filter=None):
    index = movies[movies['original_title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:20]:  # Get top 20 to apply filters
        movie_data = movies.iloc[i[0]]
        
        # Language filter - handle missing 'original_language'
        if 'original_language' in movie_data and language_filter != 'All' and language_filter != movie_data['original_language']:
            continue

        movie_id = movie_data['id']
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movie_data['original_title'])
        if len(recommended_movie_names) == 7:  # Limit to 7 recommendations
            break
    return recommended_movie_names, recommended_movie_posters

# Fetch trending movies
def fetch_trending_movies():
    url = "https://api.themoviedb.org/3/trending/movie/week?api_key=8265bd1679663a7ea12ac168da84d2e8"
    data = requests.get(url).json()
    trending = [(movie['title'], fetch_poster(movie['id'])) for movie in data['results']]
    return trending

# Sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")

def fetch_reviews(movie_id):
    # Placeholder for reviews (fetch actual reviews using an API in real implementation)
    return ["Amazing storyline!", "Not great, but watchable", "Loved it!", "Could have been better"]

def analyze_sentiments(reviews):
    sentiments = sentiment_model(reviews)
    return sentiments

# Streamlit setup
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: white;
    }
    header, .css-1lcbmhc.e1fqkh3o3 {
        background-color: #000000;
        color: white;
    }
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 18px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 40px;
    }
    .movie-title {
        text-align: center;
        font-size: 16px;
        margin-top: 10px;
        color: white;
    }
    img {
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #E50914;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #f6121d;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Display headers
st.markdown("<div class='main-header'>Movie Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Select a movie you like, and we’ll recommend similar titles.</div>", unsafe_allow_html=True)

# Load the new movies and similarity matrix from pickle files
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Ensure 'release_year' is extracted safely
if 'release_date' in movies.columns:
    movies['release_year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
else:
    movies['release_year'] = None  # Set to None if 'release_date' doesn't exist

# Initialize session states for recommendations and sentiments
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'show_sentiments' not in st.session_state:
    st.session_state.show_sentiments = False

# Movie selection
movie_list = movies['original_title'].values
selected_movie = st.selectbox("Select a movie 🎮", movie_list)

# Filters
# Safely check for language filter (handle missing column)
if 'original_language' in movies.columns:
    filter_language = st.selectbox("Filter by Language", ['All'] + list(movies['original_language'].dropna().unique()))
else:
    filter_language = 'All'  # Default to 'All' if no language column exists

# Show recommendations
if st.button('Show Recommendations'):
    st.session_state.show_recommendations = True
if st.session_state.show_recommendations:
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie, filter_language)
    st.write("### Recommended for you:")
    cols = st.columns(7)
    for i in range(len(recommended_movie_names)):
        with cols[i % 7]:
            st.image(recommended_movie_posters[i], use_container_width=True)
            st.markdown(f"<div class='movie-title'>{recommended_movie_names[i]}</div>", unsafe_allow_html=True)

# Show sentiments
if st.button('Show Sentiments'):
    st.session_state.show_sentiments = True
if st.session_state.show_sentiments:
    reviews = fetch_reviews(selected_movie)
    sentiments = analyze_sentiments(reviews)

    # Check if the selected movie is "Green Lantern" and set sentiment to Neutral
    if selected_movie.lower() == "green lantern":
        sentiment_count = {
            "POSITIVE": 0,
            "NEGATIVE": 0,
            "NEUTRAL": len(reviews)  # All reviews will be considered neutral for Green Lantern
        }
    else:
        # Prepare sentiment distribution
        sentiment_labels = [sentiment['label'] for sentiment in sentiments]
        
        sentiment_count = {
            "POSITIVE": sentiment_labels.count("POSITIVE"),
            "NEGATIVE": sentiment_labels.count("NEGATIVE"),
            "NEUTRAL": sentiment_labels.count("NEUTRAL")  # You can add a neutral sentiment if the model provides it
        }
    
    # Determine overall sentiment based on the highest count
    overall_sentiment = max(sentiment_count, key=sentiment_count.get)

    # Sentiment color mapping
    sentiment_color = {
        "POSITIVE": "green",
        "NEGATIVE": "red",
        "NEUTRAL": "gray"
    }

    # Display the overall sentiment in a large text box
    st.markdown(f"<h2 style='color: {sentiment_color[overall_sentiment]}; text-align: center;'>Overall Sentiment: {overall_sentiment}</h2>", unsafe_allow_html=True)

    # Display the overall sentiment in a box
    st.markdown(f"<div style='font-size: 40px; color: white; background-color: {sentiment_color[overall_sentiment]}; padding: 20px; border-radius: 10px;'>{overall_sentiment}</div>", unsafe_allow_html=True)

# Trending movies
st.write("### Trending Movies:")
trending_movies = fetch_trending_movies()
cols = st.columns(5)
for i, (title, poster) in enumerate(trending_movies[:5]):
    with cols[i % 5]:
        st.image(poster, use_container_width=True)
        st.markdown(f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True)
