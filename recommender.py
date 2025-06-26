import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load and Prepare Data ---
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')

df = movies.merge(credits, left_on='id', right_on='movie_id')

# Fill missing values
df['overview'] = df['overview'].fillna('')
df['cast'] = df['cast'].fillna('')
df['crew'] = df['crew'].fillna('')
df['genres'] = df['genres'].fillna('[]')

# --- Feature Extraction ---

def get_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        genre_names = [g['name'] for g in genres_list]
        return ' '.join(genre_names)
    except Exception:
        return ''

def get_top_cast(cast_str):
    try:
        cast_list = ast.literal_eval(cast_str)
        names = [x['name'] for x in cast_list[:3]]
        return ' '.join(names)
    except Exception:
        return ''

def get_director(crew_str):
    try:
        crew_list = ast.literal_eval(crew_str)
        directors = [x['name'] for x in crew_list if x['job'] == 'Director']
        return ' '.join(directors)
    except Exception:
        return ''

df['genre_names'] = df['genres'].apply(get_genres)
df['cast_names'] = df['cast'].apply(get_top_cast)
df['director_names'] = df['crew'].apply(get_director)

# --- Combine Features ---
df['combined_features'] = (
    df['overview'] + ' ' +
    df['cast_names'] + ' ' +
    df['director_names'] + ' ' +
    df['genre_names']
)

# --- TF-IDF Vectorization ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# --- Cosine Similarity Matrix ---
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- Title Index Mapping ---
indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()

# --- Recommend by Title ---
def get_recommendations(title, num=5):
    if not title or title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['original_title'].iloc[movie_indices].tolist()

# --- Recommend by Genres (or keywords) ---
def get_recommendations_by_genres(input_genres, num=5):
    if not input_genres or not input_genres.strip():
        return []
    input_vec = tfidf.transform([input_genres])
    sim_scores = list(enumerate(cosine_similarity(input_vec, tfidf_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:num]
    movie_indices = [i[0] for i in sim_scores]
    return df['original_title'].iloc[movie_indices].tolist()
