import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- FETCH POSTER ---------------- #

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data.get('poster_path')

    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        return ""


# ---------------- LOAD DATA ---------------- #

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)


# ---------------- COMPUTE SIMILARITY ---------------- #

@st.cache_data
def compute_similarity(dataframe):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(dataframe['tags']).toarray()
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix


similarity = compute_similarity(movies)


# ---------------- RECOMMEND FUNCTION ---------------- #

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters


# ---------------- STREAMLIT UI ---------------- #

st.header('Movie Recommender System')

movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    for idx, col in enumerate([col1, col2, col3, col4, col5]):
        with col:
            st.text(names[idx])
            st.image(posters[idx])