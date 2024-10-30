import pandas as pd
import re
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_tweets():
    df = pd.read_csv('Datasets/compiled_tweets.csv')

    #df.drop(['in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo', 'coordinates', ])
    df = df.loc[:, df.count() >= 2131]
    df = df.drop(columns=['id', 'id_str', 'source', 'truncated', 'user', 'retweet_count', 'favorite_count', 'favorited', 'retweeted', 'entities'], axis=1)
    df = df[df['lang'] == 'en']
    df = df.drop(columns=['lang'])

    df['created_at'] = pd.to_datetime(df['created_at'])

    df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+|www.\S+|@\w+|#w+', '', x))
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
    df['text'] = df['text'].str.replace(r'\d+', '', regex=True)

    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))
    
    df.to_csv('Datasets/preprocessed_tweets.csv', index=False)

def preprocess_sentiment():
    df = pd.read_csv('Datasets/stock_tweet_sentiment.csv')
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(columns=['timestamp', 'source', 'symbols', 'company_names'], axis=1)

    df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+|www.\S+|@\w+|#w+', '', x))
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
    df['text'] = df['text'].str.replace(r'\d+', '', regex=True)

    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))

    df.to_csv('Datasets/preprocessed_sentiment.csv', index=False)

def preprocess_price():
    df = pd.read_csv('Datasets/JPM_price.csv')
    pca = PCA(n_components=0.90)

    df['date'] = pd.to_datetime(df['Date'])
    df['return'] = df['Adj Close'].pct_change()
    df['high_low'] = df['High'] - df['Low']
    df['avg_price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    df['vol_change'] = df['Volume'].pct_change()

    df['avg_price_5d'] = df['avg_price'].rolling(window=5).mean()
    df['return_5d'] = df['return'].rolling(window=5).mean()

    #df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    features = df[['return', 'high_low', 'avg_price', 'vol_change', 'avg_price_5d', 'return_5d']]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca_features = pca.fit_transform(scaled_features)
    pc_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca_features.shape[1])])

    df = pd.concat([df[['date']], pc_df], axis=1)

    df.to_csv('Datasets/preprocessed_price.csv', index=False)

def preprocess():
    preprocess_sentiment()
    preprocess_tweets()
    preprocess_price()

preprocess()