import json
import plotly
import pandas as pd
import re

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.base import BaseEstimator,TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(texts):

    ## convert bytes to string format 
    normalised_text = re.sub(r'[^a-zA-Z0-9]'," ",str(texts).lower())
    #dataframe['column_name'] = dataframe['column_name'].fillna('').apply(str)
    
    tokens = word_tokenize(normalised_text)
    tokens = [token for token in tokens if token not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(token) for token in tokens]
    clean_tokens = [lemmatizer.lemmatize(w,pos='v') for w in lemmed]
    

    return clean_tokens


class VerbCounter(BaseEstimator,TransformerMixin):

    def counter(self, corpus):
        for sentence in corpus:
            count = 0
            token = tokenize(sentence)
            pos = pos_tag(token)
            for word,tag in pos:
                if tag in ['VB', 'VBP','VBZ']:
                    count += 1
            return count
            
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        verb_count = pd.Series(X).apply(self.counter)
        return pd.DataFrame(verb_count)

# load data
engine = create_engine('sqlite:///../data/Disaster_Response.db')
df = pd.read_sql_table('Disaster_Response_Table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()