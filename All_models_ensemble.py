import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import VotingClassifier
from textblob import TextBlob
from gensim.models import Word2Vec


def clean_text(text, domain_specific_words=None):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join(text.split())
        stop_words = set(stopwords.words('english'))
        if domain_specific_words:
            stop_words = stop_words.union(set(domain_specific_words))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
        return text
    else:
        return text

def lexicon_sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def train_word2vec(corpus):
    tokenized_corpus = [sentence.split() for sentence in corpus]
    word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=8, min_count=1, workers=4)
    return word2vec_model

def vectorize_text(text, model, vector_size):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(vector_size)

def evaluate_ensemble(models, X_train, y_train, X_test, y_test, test_name, prediction_output, metrics_output):
    # Creating a Voting Classifier
    voting_classifier = VotingClassifier(estimators=[
        (name, model) for name, model, _, _ in models
    ], voting='soft')  # 'soft' voting averages class probabilities

    # Fit the ensemble model on the centralized feature space
    print(f"\nTraining ensemble on all feature spaces for {test_name}...")
    voting_classifier.fit(np.hstack([X_train_feature for _, _, X_train_feature, _ in models]), y_train)
    
    # Make predictions on the test set
    predictions = voting_classifier.predict(np.hstack([X_test_feature for _, _, _, X_test_feature in models]))
    
    # Evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
    print(f"\nEnsemble Results: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    # Save predictions to a DataFrame
    prediction_df = pd.DataFrame({
        'True Label': y_test,
        'Ensemble Prediction': predictions
    })
    
    metrics_dict = {
        'Model': 'Ensemble',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }
    
    # Save results to CSV
    prediction_df.to_csv(prediction_output, index=False)
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(metrics_output, index=False)

def main():
    datasets = {
        'IMDB': './IMDB_Dataset.csv',
        'Spotify': './Spotify_app.csv',
        'Amazon': './Amazon.csv',
        'RT': './Rotten_tomatoes.csv',
        'Twitter': './twitter_training.csv'
    }

    # Prepare all datasets
    dfs = {}
    for name, path in datasets.items():
        if name == 'Amazon':
            df = pd.read_csv(path, header=None, nrows=55000)
            df['review_cleaned'] = df[2].apply(clean_text).fillna('')
            df[0] = df[0].map({1: 0, 2: 1})
            df['sentiment_label'] = df[0]
        elif name == 'Twitter':
            df = pd.read_csv(path, header=None)
            df = df[df[2].isin(['Positive', 'Negative'])]
            df['review_cleaned'] = df[3].apply(clean_text).fillna('')
            le = LabelEncoder()
            df['sentiment_label'] = le.fit_transform(df[2])
        elif name == 'IMDB':
            df = pd.read_csv(path)
            df['review_cleaned'] = df['review'].apply(clean_text).fillna('')
            le = LabelEncoder()
            df['sentiment_label'] = le.fit_transform(df['sentiment'])
        elif name == 'Spotify':
            df = pd.read_csv(path)
            df['review_cleaned'] = df['Review'].apply(clean_text).fillna('')
            le = LabelEncoder()
            df['sentiment_label'] = le.fit_transform(df['label'])
        else:  # Rotten tomatoes
            df = pd.read_csv(path)
            df['review_cleaned'] = df['text'].apply(clean_text).fillna('')
            df['sentiment_label'] = df['label']

        dfs[name] = df

    for test_name in datasets.keys():
        print(f"\nRunning ensemble evaluation with {test_name} as the test set...")

        # Create test set
        df_test = dfs[test_name]
        X_test = df_test['review_cleaned']
        y_test = df_test['sentiment_label']

        # Create training set by combining other datasets
        train_dfs = [df for key, df in dfs.items() if key != test_name]
        df_train = pd.concat(train_dfs)

        X_train = df_train['review_cleaned']
        y_train = df_train['sentiment_label']

        # Vectorization
        count_vectorizer = CountVectorizer(max_features=1000)
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)

        X_train_bow = count_vectorizer.fit_transform(X_train)
        X_test_bow = count_vectorizer.transform(X_test)
        
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        word2vec_model = train_word2vec(X_train)
        vector_size = 100

 # Convert 1D Word2Vec vectors to 2D
        X_train_vec = np.array([vectorize_text(text, word2vec_model, vector_size) for text in X_train])
        X_test_vec = np.array([vectorize_text(text, word2vec_model, vector_size) for text in X_test])
        
        # Reshape to 2D arrays: (n_samples, n_features)
        X_train_vec = X_train_vec.reshape(-1, vector_size)
        X_test_vec = X_test_vec.reshape(-1, vector_size)

        # Define models
        models = [
            ("SVM (BOW)", CalibratedClassifierCV(LinearSVC()), X_train_bow),
            ("Naive Bayes (BOW)", MultinomialNB(), X_train_bow),
            ("Logistic Regression (BOW)", LogisticRegression(max_iter=1000), X_train_bow),
            ("SVM (TF-IDF)", CalibratedClassifierCV(LinearSVC()), X_train_tfidf),
            ("Naive Bayes (TF-IDF)", MultinomialNB(), X_train_tfidf),
            ("Logistic Regression (TF-IDF)", LogisticRegression(max_iter=1000), X_train_tfidf),
            ("SVM (Word2Vec)", CalibratedClassifierCV(LinearSVC()), X_train_vec),
            ("Logistic Regression (Word2Vec)", LogisticRegression(max_iter=1000), X_train_vec),
        ]

        # Combine models and test features
        models_eval = [
            (name, model, train_feature, test_feature)
            for (name, model, train_feature), test_feature in zip(models, [X_test_bow, X_test_bow, X_test_bow,
                                                                             X_test_tfidf, X_test_tfidf, X_test_tfidf,
                                                                             X_test_vec, X_test_vec])
        ]

        # File paths for output
        prediction_output = f"{test_name}_predictions_ensemble.csv"
        metrics_output = f"{test_name}_metrics_ensemble.csv"

        # Evaluate ensemble
        evaluate_ensemble(models_eval, X_train, y_train, X_test, y_test, test_name, prediction_output, metrics_output)

if __name__ == "__main__":
    main()