import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import re

from nltk import pos_tag
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
#nltk.download('stopwords')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    '''
    Load the data from db file. Perform data preprocessing and train-test split on the data

    Args:
        database_filepath: Your database file location

    Returns:
        X: 'message' column in array
        Y: all the categorical column data
        category_names: categories column name
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df_ori = pd.read_sql_table('Disaster_Response_Table',engine)

    df = df_ori.drop(['original'],axis = 1)
    df.drop('child_alone',axis=1,inplace = True)
    df['related']=df['related'].apply(lambda x : 1 if x == 2 else x)
    df.dropna(inplace=True)
    
    X = df['message'].values
    Y = df[df.columns[3:]]

    category_names = Y.columns

    return X,Y,category_names


def tokenize(texts):
    '''
    This function performs text preprocessing on the message data

    Args:
        texts:byte object. To be converted to string object within teh function
    
    Returns:
        clean_tokens: Tokenized texts
    '''

    ## convert bytes to string format 
    normalised_text = re.sub(r'[^a-zA-Z0-9]'," ",str(texts).lower())
    
    tokens = word_tokenize(normalised_text)
    tokens = [token for token in tokens if token not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(token) for token in tokens]
    clean_tokens = [lemmatizer.lemmatize(w,pos='v') for w in lemmed]
    

    return clean_tokens


class VerbCounter(BaseEstimator,TransformerMixin):
    '''
    This transformer counts the number of Verb occurrence within a sentence from the message corpus.
    '''

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


def build_model(clf=RandomForestClassifier()):
    '''
    Load the transformers and classifier model into a pipeline

    Args:
        classifier model

    Returns:
        pipeline

    '''
    pipeline = Pipeline([
    ('features',FeatureUnion([
        ('text_pipeline',Pipeline([
            ('vect',CountVectorizer(tokenizer = tokenize)),
            ('tfidf',TfidfTransformer())
        ])),
            ('verb_counter',VerbCounter())
        ])),
        ('clf',MultiOutputClassifier(clf))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performance

    Args:
        model: pipeline loaded with transformers and model
        X_test: Test features data
        Y_test: Test labels data
        category_names: Different categories name
    
    Returns:
        None
    '''

    y_pred=model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Serialize the trained model to be saved and stored locally

    Args:
        model: pipeline object
        model_filepath: Location to store the trained model in .pkl file
    
    Returns:
        None

    '''
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/Disaster_Response.db classifier.pkl')


if __name__ == '__main__':
    main()