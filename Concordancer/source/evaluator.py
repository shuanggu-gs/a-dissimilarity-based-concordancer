import numpy as np
from rouge import Rouge
from scipy.spatial.distance import cdist
from feature_extractor import *

def cosine_calc(features):
    """
    Given features of instances, return the average cosine distances of each pair of instances
    :param features: 2-D array, features of instances
    :return: float, average cosine score of those features
    """
    Y = cdist(features, features, 'cosine')
    score = 0
    count = 0
    for i in range(len(features)-1):
        for j in range(i+1, len(features)):
            score += Y[i][j]
            count += 1
    if count == 0:
        return 0
    return score/count

def rouge_calc(context_list, rouge_type='rouge-l'):
    """
    Takes a given rouge type and list of contexts (list of strings) to calculate and returns the fscore, precision, recall and other statisitics of the distribution
    :param context_list: a list of triples
    :param rouge_type: str
    :return: dict, mean f1-score and standard deviation
    """
    sentences = [' '.join(left_context+right_context) for left_context, _, right_context in context_list]
    # creating a list of tuples where each tuple contains a pair of sentences to be compared
    sent_tuple_list = []
    for i in range(len(sentences)-1):
        for j in range(i + 1, len(sentences)):
            sent_tuple_list.append((sentences[i], sentences[j]))

    # instantiating a rouge object to compare each pair and take average of the fscore of each pair
    rouge = Rouge()
    fscore_list = []
    precision_list = []
    recall_list = []
    result_dict = dict()
    
    for tup in sent_tuple_list:
        tup_1, tup_2 = tup
        f_score = rouge.get_scores(tup_1, tup_2)[0][rouge_type]['f']
        precision = rouge.get_scores(tup_1, tup_2)[0][rouge_type]['p']
        recall = rouge.get_scores(tup_1, tup_2)[0][rouge_type]['r']
        
        # print(score)
        fscore_list.append(f_score)
        precision_list.append(precision)
        recall_list.append(recall)
    
    result_dict['mean_fscore'] = np.mean(fscore_list)
    # result_dict['mean_precision'] = np.mean(precision_list)
    # result_dict['mean_recall'] = np.mean(recall_list)
    # result_dict['min_fscore'] = np.min(fscore_list)
    # result_dict['max_fscore'] = np.max(fscore_list)
    result_dict['std_fscore'] = np.std(fscore_list)

    return result_dict