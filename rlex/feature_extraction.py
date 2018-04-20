import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_sentence_doc_features(
        articles,
        tfidf_norm='l2',
        tfidf_max_features=100,
        pca_features=100,
        verbose=True,):
    """
    :param articles: list of Article
    :param tfidf_norm: 'l1', 'l2', or None, normalizes tfidf vectors
    :param tfidf_max_features: maximum number of features in the vector
    :param pca_features: numer of features we PCA down to
    :param verbose: boolean
    :return: numpy array of features of size A, A articles,
             and at each article idx there's the number of sents
    """
    all_sents = []
    max_len = 0
    for a in articles:
        max_len = max(max_len, len(a))
        for i in range(len(a)):
            all_sents.append(a.get_doc_sent_string(i))

    if verbose: print('TFIDF is vectorizing...')
    tfidf = TfidfVectorizer(norm=tfidf_norm, max_features=tfidf_max_features)
    X = np.array(tfidf.fit_transform(all_sents).todense())

    if pca_features < tfidf_max_features:
        X = PCA(n_components=min(pca_features, X.shape[1])).fit_transform(X)

    # now this is where the fun begins. first get mean doc feature vectors
    doc_feats = []
    aidx = 0
    for a in articles:
        doc_feats.append(np.mean(X[aidx: len(a)+aidx], axis=0))
        aidx += len(a)

    # now we need to relate sentence vectors to their doc vectors
    # for now, just concatenate them, but also add distance features
    position_feats = np.identity(max_len) # one-hot-encoding of position
    aidx = 0
    crt_art_idx = 0
    arts_sent_feats = [[]]

    if verbose: print('Extracting sentence-level features...')
    for i in range(len(X)):
        if i >= len(articles[crt_art_idx]) + aidx:
            aidx += len(articles[crt_art_idx])
            crt_art_idx += 1
            arts_sent_feats.append([])

        # distance features
        l1_d = np.linalg.norm(X[i] - doc_feats[crt_art_idx], ord=1)
        l2_d = np.linalg.norm(X[i] - doc_feats[crt_art_idx], ord=2)
        cos_d = cosine(X[i], doc_feats[crt_art_idx])
        dist_feats = np.array([l1_d, l2_d, cos_d])

        # combine them all uppp: that is
        # -- tfidf features of sentence
        # -- the total document's features
        # -- the distance features
        # -- the position features
        x_feat = np.array(X[i]).flatten()
        arts_sent_feats[-1].append(
            np.concatenate((x_feat, dist_feats, position_feats[i - aidx],))
        )

    if verbose: print('Feature extraction complete.')
    return list(map(np.array, arts_sent_feats)), doc_feats

# def combine
