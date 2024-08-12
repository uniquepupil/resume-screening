from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(texts):
    """
    Extract TF-IDF features from the given texts.
    
    :param texts: List of texts (e.g., resume and job description).
    :return: TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix
