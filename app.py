import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the data
data = pd.read_csv('tmdb_5000_movies.csv')

# Function to safely parse JSON strings
def parse_json(x):
    try:
        return json.loads(x.replace("'", '"'))
    except json.JSONDecodeError:
        return []

# Process the genres column to convert JSON-like strings into a readable format
data['genres'] = data['genres'].apply(lambda x: ' '.join([i['name'] for i in parse_json(x)]))

# Process the keywords column to convert JSON-like strings into a readable format
data['keywords'] = data['keywords'].apply(lambda x: ' '.join([i['name'] for i in parse_json(x)]))

# Create a combined features column
data['combined_features'] = data['genres'] + ' ' + data['keywords']

# Create a TfidfVectorizer object
tfidf = TfidfVectorizer(stop_words='english')

# Apply the TF-IDF vectorizer to the combined features column
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on the title
def get_recommendations(title, cosine_sim=cosine_sim):
    # Check if the title exists in the data
    if title not in data['title'].values:
        return []

    # Get the index of the movie that matches the title
    idx = data[data['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices].tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        recommendations = get_recommendations(movie_title)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
