import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_data(clean_corpus_train, clean_corpus_test, min_df=1, max_df=1, ngram_range=(1, 1)):
    """
    Creates a count vectorizer and a tfidf vectorizer
    :param clean_corpus_train: The corpus we want to train on (e.g. the year-end hits)
    :param clean_corpus_test: The corpus we want to test on (e.g. the one-hit wonders)
    :param min_df: The minimum number of times a word needs to appear in the corpus to be included in the count
    vector (default 1)
    :param max_df: The maximum % of documents a word can appear in to be included in the vector (e.g. 1.0 means 100%
    of documents, default 1)
    :param ngram_range: The range of n_grams to be constructed by the vectorizer, as a tuple (default is (1,1))
    :return: The tdidf vectors for both the training and test corpuses
    """
    cv_tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    X_train = cv_tfidf.fit_transform(clean_corpus_train)
    X_test = cv_tfidf.transform(clean_corpus_test)
    X_train = pd.DataFrame(X_train.toarray(), columns=cv_tfidf.get_feature_names())
    X_test = pd.DataFrame(X_test.toarray(), columns=cv_tfidf.get_feature_names())
    return X_train, X_test

