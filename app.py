from flask import Flask, request, jsonify, render_template  # <- added render_template here
from flask_cors import CORS
from recommender import get_recommendations, get_recommendations_by_genres

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is inside 'templates' folder

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('movie')
    if not title:
        return jsonify({'error': 'Movie title is required'}), 400

    try:
        result = get_recommendations(title)
        if not result:
            return jsonify({'movie': title, 'recommendations': [], 'message': 'Movie not found'}), 404
        return jsonify({'movie': title, 'recommendations': result})
    except Exception as e:
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/recommend_by_keywords', methods=['GET'])
def recommend_by_keywords():
    keywords = request.args.get('q')
    if not keywords:
        return jsonify({'error': 'Keywords parameter is required'}), 400

    try:
        result = get_recommendations_by_genres(keywords)  # Make sure this function exists in recommender.py
        if not result:
            return jsonify({'keywords': keywords, 'recommendations': [], 'message': 'No matching movies found'}), 404
        return jsonify({'keywords': keywords, 'recommendations': result})
    except Exception as e:
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
