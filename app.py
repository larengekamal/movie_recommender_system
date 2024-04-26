# Importing Library
import pandas as pd
import numpy as np
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast # convert string dict to dict
import requests

@st.cache_resource
def load_data_frame():
    global df
    global similarity
    df_credits=pd.read_csv("tmdb_5000_credits.csv")
    df_movies=pd.read_csv("tmdb_5000_movies.csv")
    df=df_movies.merge(df_credits,on='title')
    pre_process_dataframe()
    # Stemming
    df['tags']=df['tags'].apply(do_stem)
    # Text Vectorization using Bag of Words
    cv=CountVectorizer(stop_words='english', max_features=5000)
    vectors=cv.fit_transform(df['tags']).toarray()
    # cosine similarity
    similarity=cosine_similarity(vectors)
    return df, similarity

# pre_process_dataframe
def pre_process_dataframe():
    # Dropping not used columns: 
    # budget, homepage, original_language, original_title,
    # popularity, production_companies, production_countries,
    # release_date, revenue, runtime, spoken_languages, status,
    # tagline, vote_average, vote_count, movie_id
    df.drop(columns=['budget', 'homepage', 'original_language', 'original_title', 'popularity',
                    'production_companies', 'production_countries', 'release_date', 'revenue',
                    'runtime', 'spoken_languages', 'status', 'tagline', 'vote_average', 'vote_count', 'movie_id'],
        inplace=True)
    df.head()
    # Dropping null values
    df.dropna(inplace=True)

    # Preprocess genres
    def convert_genres(obj_string_list):
        l=[]
        for i in ast.literal_eval(obj_string_list):
            l.append(i['name'])
        return l    
    df['genres']=df['genres'].apply(convert_genres)

    # Preprocess keywords
    df['keywords']=df['keywords'].apply(convert_genres)

    # Preprocess cast
    # Get 1st, 5 Cast
    def convert_cast(obj_string_list):
        l=[]
        counter = 0
        for i in ast.literal_eval(obj_string_list):
            if(counter != 5):
                l.append(i['name'])
                counter=counter+1
            else:
                break
        return l   
    df['cast']=df['cast'].apply(convert_cast)

    # Preprocess crew
    # Get Director name
    def convert_crew(obj_string_list):
        l=[]
        for i in ast.literal_eval(obj_string_list):
            if(i['job'] == 'Director'):
                l.append(i['name'])
                break
        return l  
    df['crew']=df['crew'].apply(convert_crew)

    # Preprocess overview
    # convert to list
    df['overview']=df['overview'].apply(lambda x:x.split())

    # Remove spaces between words
    df['genres']=df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
    df['keywords']=df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
    df['cast']=df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
    df['crew']=df['crew'].apply(lambda x:[i.replace(" ","") for i in x])

    df['tags']=df['overview']+df['genres']+df['keywords']+df['cast']+df['crew']
    df.drop(columns=['genres', 'keywords', 'overview', 'cast', 'crew'], inplace=True)

    # Convert list in tags to string
    df['tags']=df['tags'].apply(lambda x: " ".join(x))

    # convert to lower case
    df['tags']=df['tags'].apply(lambda x: x.lower())
    df['title']=df['title'].apply(lambda x: x.lower())

    
def do_stem(text):
    ps=PorterStemmer()
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)   

def recommend_movies(movie_name):
    recommended_movies=[]
    recommended_movies_id=[]
    try:
        movie_index=df[df['title']==movie_name.lower()].index[0]
        distances=similarity[movie_index]
        # sort in descending and get top 5 movies
        movie_list=sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

        for i in movie_list:
            print(f"Recommended Movie {df.iloc[i[0]].title} with ID={df.iloc[i[0]].id}")
            recommended_movies.append(df.iloc[i[0]].title)
            recommended_movies_id.append(df.iloc[i[0]].id)
    except IndexError:
        print("Not in Database")
    return recommended_movies, recommended_movies_id 

def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data=response.json()
    return "https://image.tmdb.org/t/p/w500"+data['poster_path']

df,similarity =load_data_frame()
st.title("Movie Recommender System")
option = st.selectbox(
    'Select a movie from list?',
    df['title'].values)
if st.button('Recommend'):
    st.write('You selected:', option)
    try:
        selected_movie_id=df[df['title'] == option]['id'].to_string(index=False)
        print(f"Selected movie ID={selected_movie_id}")
        st.image(fetch_poster(selected_movie_id))
    except IndexError:
        None    
    st.markdown("""---""")    
    st.header("Recommended Movies:")    
    movie_list, movie_id = recommend_movies(option)
    col1,col2,col3,col4,col5=st.columns(5, gap="small")
    with col1:
        st.caption(movie_list[0])
        st.image(fetch_poster(movie_id[0]))
    with col2:
        st.caption(movie_list[1])
        st.image(fetch_poster(movie_id[1]))
    with col3:
        st.caption(movie_list[2])
        st.image(fetch_poster(movie_id[2]))
    with col4:
        st.caption(movie_list[3])
        st.image(fetch_poster(movie_id[3]))
    with col5:
        st.caption(movie_list[4])
        st.image(fetch_poster(movie_id[4]))
    st.markdown("""---""")        
    # for i in movie_list:
    #     st.write('Recommended Movie is :', i)