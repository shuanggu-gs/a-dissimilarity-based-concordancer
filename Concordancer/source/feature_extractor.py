import numpy as np
from sentence_transformers import SentenceTransformer
import sister

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import pickle


def extract_features(instances, lang_id, model_path=None, feature_type='sbert'):
    """
    Given n instances, return features of dim (n_samples * feature_dim)

    :param instances, a list of triples, [left context], target word, [right context])
    :param lang_id, str, used for fasttext feature
    :param model_path, str, used for word2vec, glove features
    :param feature_type: str, 'word2vec', 'glove', 'fasttext' and 'sbert', default is 'sbert'
    :return: 2-D array, size is n_samples * feature_dim
    """

    if feature_type == 'word2vec':
        left_features, right_features = word2vec(instances, model_path)
    elif feature_type == 'glove':
        left_features, right_features = glove_vec(instances, model_path)
    elif feature_type == 'fasttext':
        left_features, right_features = fasttext(instances, lang_code=lang_id)
    else:
        left_features, right_features = sbert(instances)

    features = np.concatenate((left_features, right_features), axis=1)
    return features

def dimension_reduction(features, reduction_type='pca', n_components=10):
    """
    Applied dimensionality reduction and normalization

    :param features: 2-D array of size n_samples * feature_dim
    :param reduction_type: str, 'pca' or 'tsne', default is 'pca'
    :param n_components: feature dimension
    :return: 2-D array, reduced feature dimension with size n_samples * n_components
    """

    if reduction_type == 'tsne':
        tsne = TSNE(n_components=n_components, method='exact')
        features = normalize(features, norm='l2')
        features = tsne.fit_transform(features)
    else:
        reduction = PCA(n_components=n_components)
        # normalization
        features = normalize(features, norm='l2')
        features = reduction.fit_transform(features)
        # print(reduction.explained_variance_ratio_)
        # print("sum of pca variance ratio:", sum(reduction.explained_variance_ratio_))

    return features

def glove_vec(instances, model_path):
    """
    Finds instances of a target word and retrieves surrounding context vectors with a given vectorizer model.
    Returns vect_to_context (list of tuples of vectors with the source context) and contexts (list of context vecs)

    :param instances: a list of triples, [left context], target word, [right context])
    :param model_path: str, the path of pre-trained model
    :return: a pair of 2-D array: features of left and right contexts
    """
    vec_model = pickle.load(open(model_path, 'rb'))
    left_feats = []
    right_feats = []
    # Obtaining average vector of all vectors within vocab of vec_model
    average_vec = np.mean([vec_model[token] for token in vec_model.keys()], axis=0)
    
    for left_context, target, right_context in instances:
        left = []
        for word in left_context:
            token = word.lower()
            if token in vec_model:
                left.append(vec_model[token])
            else:
                # OOV tokens use a vector which is an average of all vectors in vocab
                left.append(average_vec)
        if left == []:
            left = np.zeros(average_vec.size)
        else:
            left = np.mean(np.array(left), axis=0)

        right = []
        for word in right_context:
            token = word.lower()
            if token in vec_model:
                right.append(vec_model[token])
            else:
                # OOV tokens use a vector which is an average of all vectors in vocab
                right.append(average_vec)
        if right == []:
            right = np.zeros(average_vec.size)
        else:
            right = np.mean(np.array(right), axis=0)

        assert right.size == left.size, "Size mismatch between left and right contexts"
        left_feats.append(left)
        right_feats.append(right)

    return left_feats, right_feats


def sbert(instances):
    """
    take a list of strings and return the embeddings of left/right context
    https://github.com/UKPLab/sentence-transformers

    @article{reimers-2020-multilingual-sentence-bert,
        title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
        author = "Reimers, Nils and Gurevych, Iryna",
        journal= "arXiv preprint arXiv:2004.09813",
        month = "04",
        year = "2020",
        url = "http://arxiv.org/abs/2004.09813",
    }

    :param instances: list of triple ([left context], target word, [right context])
    :return: a pair of 2-D array: features of left and right contexts
    """
    model = SentenceTransformer('distiluse-base-multilingual-cased')
    left_contexts = [' '.join(left_context) for left_context, _, _ in instances]
    right_contexts = [' '.join(right_context) for _, _, right_context in instances]
    left_feats = model.encode(left_contexts)
    right_feats = model.encode(right_contexts)

    return left_feats, right_feats


def word2vec(instances, model_path):
    """
    Finds instances of a target word and retrieves surrounding context vectors with a given vectorizer model.
    Returns vect_to_context (list of tuples of vectors with the source context) and contexts (list of context vecs)

    :param instances: a list of triples, [left context], target word, [right context])
    :param model_path: str, the path of pre-trained model
    :return: a pair of 2-D array: features of left and right contexts
    """
    vec_model = pickle.load(open(model_path, 'rb'))
    
    left_feats = []
    right_feats = []
    # Obtaining average vector of all vectors within vocab of vec_model
    average_vec = np.mean([vec_model.wv[token] for token in vec_model.wv.vocab.keys()], axis=0)
    for left_context, target, right_context in instances:
        left = []
        for word in left_context:
            token = word.lower()
            if token in vec_model.wv.vocab:
                left.append(vec_model.wv[token])
            else:
                # OOV tokens use a vector which is an average of all vectors in vocab
                left.append(average_vec)
        if left == []:
            left = np.zeros(average_vec.size)
        else:
            left = np.mean(np.array(left), axis=0)

        right = []
        for word in right_context:
            token = word.lower()
            if token in vec_model.wv.vocab:
                right.append(vec_model.wv[token])
            else:
                # OOV tokens use a vector which is an average of all vectors in vocab
                right.append(average_vec)
        if right == []:
            right = np.zeros(average_vec.size)
        else:
            right = np.mean(np.array(right), axis=0)
        

        assert right.size == left.size, "Size mismatch between left and right contexts"
        left_feats.append(left)
        right_feats.append(right)

    return left_feats, right_feats


def fasttext(instances, lang_code='en'):
    """
    take a list of strings and lang code and return the fasttext embeddings of sentences

    :param instances: a list of triples, [left context], target word, [right context])
    :param lang_code: str
    :return: a pair of 2-D array: features of left and right contexts
    """
    embedder = sister.MeanEmbedding(lang=lang_code)
    left_feats = []
    right_feats = []
    for left_context, _, right_context in instances:
        if left_context == []:
            left_feats.append(np.zeros(300))
        else:
            left_feats.append(embedder(' '.join(left_context)))
        if right_context == []:
            right_feats.append(np.zeros(300))
        else:
            right_feats.append(embedder(' '.join(right_context)))
    return left_feats, right_feats
