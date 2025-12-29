import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support,mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from textblob import TextBlob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
    if len(word_vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

def get_most_influential_words(model, vectorizer, n=20, positive=True):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_.flatten()

    if positive:
        top_indices = np.argsort(coefficients)[-n:]
        top_scores = coefficients[top_indices]
    else:
        top_indices = np.argsort(coefficients)[:n]
        top_scores = coefficients[top_indices]

    top_features = feature_names[top_indices]

    sentiment = "positive" if positive else "negative"
    print(f"\nTop words for {sentiment} sentiment:\n")
    for feature, score in zip(top_features, top_scores):
        print(f"{feature}: {score:.4f}")

def save_sample_predictions(original_texts, sample_indices, y_true, predictions, model_names):
    sample_df = pd.DataFrame({
        'Text': original_texts.iloc[sample_indices].to_list(),
        'True Sentiment': y_true.iloc[sample_indices].to_list()
    })
    for model_name, preds in zip(model_names, predictions):
        sample_df[f'Predicted Sentiment ({model_name})'] = preds[sample_indices]

    
    
    sample_df.to_csv('model_sample_predictions_Spotify_IMDB.csv', index=False)

def main():

    domain_specific_words_spotify = [
    # General App and Music Terms
    "app", "application", "spotify",
    "music", "song", "songs", "playlist", "playlists",
    "album", "albums", "artist", "artists",
    
    # Features and Functionality
    "stream", "streaming", 
    "download", "downloads", "downloading", "downloaded",
    "shuffle", "repeat", "playlist", 
    "offline", "online", "library",

    # User Interface and Experience
    "interface", "ui", "ux", "layout",
    "search", "searched", "searching",
    "recommendation", "recommendations", "suggest", "suggests", "suggested",
    
    # Technical and Performance Terms
    "buffer", "buffering",
    "crash", "crashed", "crashing",
    "bug", "bugs", "glitch", "glitches",

    # Subscription and Account Management
    "premium", "subscription", "subscribe", "subscribed",
    "account", "login", "logged", "logging",
    
    # Social and Sharing Features
    "share", "shared", "sharing",
    "follow", "following", "followers"
]
    
    domain_specific_words_imdb = [
    # Genres
    "action", "comedy", "drama", "horror", "thriller", "romance", "sci-fi", "documentary", "animation",
    
    # Roles
    "director", "actor", "actress", "producer", "writer", "filmmaker", "cast", "crew",
    
    # Cinematic Elements
    "screenplay", "dialogue", "plot", "character", "scene", "climax", "soundtrack", "special effects", "cinematography",
    
    # Releases and Viewings
    "premiere", "release", "screening", "sequel", "prequel", "trilogy", "adaptation",
    
    # Technical Terms
    "editing", "score", "sound design", "visual effects", "production", "lighting",
    
    # Audience and Viewing Context
    "viewer", "audience", "fans", "fandom", "blockbuster", "box office", "flop",
    "theater", "cinema", "home viewing", "matinee", "midnight screening"
]
    df_IMDB = pd.read_csv('./IMDB_Dataset.csv')

    # df_IMDB['review_cleaned'] = df_IMDB['review'].apply(clean_text)
    df_IMDB['review_cleaned'] = df_IMDB['review'].apply(lambda x: clean_text(x, domain_specific_words_imdb)).fillna('')
    df_IMDB['polarity_score'] = df_IMDB['review_cleaned'].apply(lexicon_sentiment_analysis)

    le = LabelEncoder()
    df_IMDB['sentiment_label'] = le.fit_transform(df_IMDB['sentiment'])

    df_Spotify = pd.read_csv('./Spotify_app.csv')

    # df_Spotify['review_cleaned'] = df_Spotify['Review'].apply(clean_text).fillna('')
    df_Spotify['review_cleaned'] = df_Spotify['Review'].apply(lambda x: clean_text(x, domain_specific_words_spotify)).fillna('')
    df_Spotify['polarity_score'] = df_Spotify['review_cleaned'].apply(lexicon_sentiment_analysis)

    le = LabelEncoder()
    df_Spotify['sentiment_label'] = le.fit_transform(df_Spotify['label'])

    X_test = df_IMDB['review_cleaned']
    y_test = df_IMDB['sentiment_label']

    X_train = df_Spotify['review_cleaned']
    y_train = df_Spotify['sentiment_label']

    # Polarized sentiment as regression target
    y_train_polarity = df_Spotify['polarity_score']
    y_test_polarity = df_IMDB['polarity_score']

    # Vectorization
    count_vectorizer = CountVectorizer(max_features=1000)
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)

    X_train_bow = count_vectorizer.fit_transform(X_train)
    X_test_bow = count_vectorizer.transform(X_test)
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    word2vec_model = train_word2vec(X_train)
    vector_size = 100

    X_train_vec = np.array([vectorize_text(text, word2vec_model, vector_size) for text in X_train])
    X_test_vec = np.array([vectorize_text(text, word2vec_model, vector_size) for text in X_test])


    def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, regression=False):
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if regression:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R^2: {r2:.2f}\n")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            print(f"Confusion Matrix:\n{cm}")
            print("\nPrecision Recall F1-Score")
            for i in range(len(precision)):
                print(f"{i} {precision[i]:.2f} {recall[i]:.2f} {f1[i]:.2f} {support[i]}")
        
        return y_pred

    svm = CalibratedClassifierCV(LinearSVC())
    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    svr = SVR()

    predictions = []
    model_names = ["SVM (BOW)","Naive Bayes (BOW)", "Logistic Regression (BOW)", 
                   "SVM (TF-IDF)", "Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)", 
                   "SVM (Word2Vec)", "Logistic Regression (Word2Vec)", "RNN","SVR (BOW)","SVR (TF-IDF)", "SVR (Word2Vec)"]

    predictions.append(evaluate_model(svm, X_train_bow, X_test_bow, y_train, y_test, model_names[0]))
    predictions.append(evaluate_model(nb, X_train_bow, X_test_bow, y_train, y_test, model_names[1]))
    predictions.append(evaluate_model(lr, X_train_bow, X_test_bow, y_train, y_test, model_names[2]))
    predictions.append(evaluate_model(svm, X_train_tfidf, X_test_tfidf, y_train, y_test, model_names[3]))
    predictions.append(evaluate_model(nb, X_train_tfidf, X_test_tfidf, y_train, y_test, model_names[4]))
    predictions.append(evaluate_model(lr, X_train_tfidf, X_test_tfidf, y_train, y_test, model_names[5]))
    predictions.append(evaluate_model(svm, X_train_vec, X_test_vec, y_train, y_test, model_names[6]))
    predictions.append(evaluate_model(lr, X_train_vec, X_test_vec, y_train, y_test, model_names[7]))
    predictions.append(evaluate_model(svr, X_train_bow, X_test_bow, y_train_polarity, y_test_polarity, model_names[9], regression=True))
    predictions.append(evaluate_model(svr, X_train_tfidf, X_test_tfidf, y_train_polarity, y_test_polarity, model_names[10], regression=True))
    predictions.append(evaluate_model(svr, X_train_vec, X_test_vec, y_train_polarity, y_test_polarity, model_names[11], regression=True))

    # Get most influential words for positive sentiment from Logistic Regression using TF-IDF
    print("\nLogistic Regression (TF-IDF): Most influential words for positive sentiment")
    get_most_influential_words(lr, tfidf_vectorizer, n=10, positive=True)

    # Get most influential words for negative sentiment from Logistic Regression using TF-IDF
    print("\nLogistic Regression (TF-IDF): Most influential words for negative sentiment")
    get_most_influential_words(lr, tfidf_vectorizer, n=10, positive=False)

    # Get most influential words for positive sentiment from Logistic Regression using BOW
    print("\nLogistic Regression (BOW): Most influential words for positive sentiment")
    get_most_influential_words(lr, count_vectorizer, n=10, positive=True)

    # Get most influential words for negative sentiment from Logistic Regression using BOW
    print("\nLogistic Regression (BOW): Most influential words for negative sentiment")
    get_most_influential_words(lr, count_vectorizer, n=10, positive=False)
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    save_sample_predictions(X_test, sample_indices, y_test, predictions, model_names)

    # RNN model
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = pad_sequences(X_test_seq, maxlen=100)

    model_rnn = Sequential([
    Embedding(input_dim=10000, output_dim=256),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dense(1, activation='sigmoid')
    ])
    model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nTraining RNN...")
    model_rnn.fit(X_train_pad, y_train, epochs=3, batch_size=64, validation_split=0.2)

    rnn_pred = model_rnn.predict(X_test_pad).flatten()
    rnn_pred_label = (rnn_pred > 0.5).astype(int)
    predictions.append(rnn_pred_label)

    rnn_accuracy = accuracy_score(y_test, rnn_pred_label)
    rnn_precision, rnn_recall, rnn_f1, _ = precision_recall_fscore_support(y_test, rnn_pred_label, average='binary')
    rnn_auc = roc_auc_score(y_test, rnn_pred)

    print(f"RNN model accuracy: {rnn_accuracy:.2f}")
    print(f"RNN model precision: {rnn_precision:.2f}")
    print(f"RNN model recall: {rnn_recall:.2f}")
    print(f"RNN model F1 Score: {rnn_f1:.2f}")
    print(f"RNN model ROC AUC Score: {rnn_auc:.2f}")

if __name__ == "__main__":
    main()