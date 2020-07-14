import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan, KMeans
from evaluator import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.gridspec as gridspec
import os

from random import sample

# cnames is a set of color names which is used for showing different clusters
cnames = {
            'aliceblue': '#F0F8FF',
            'antiquewhite': '#FAEBD7',
            'aqua': '#00FFFF',
            'aquamarine': '#7FFFD4',
            'azure': '#F0FFFF',
            'beige': '#F5F5DC',
            'bisque': '#FFE4C4',
            'blanchedalmond': '#FFEBCD',
            'blue': '#0000FF',
            'blueviolet': '#8A2BE2',
            'brown': '#A52A2A',
            'burlywood': '#DEB887',
            'cadetblue': '#5F9EA0',
            'chartreuse': '#7FFF00',
            'chocolate': '#D2691E',
            'coral': '#FF7F50',
            'cornflowerblue': '#6495ED',
            'cornsilk': '#FFF8DC',
            'crimson': '#DC143C',
            'cyan': '#00FFFF',
            'darkblue': '#00008B',
            'darkcyan': '#008B8B',
            'darkgoldenrod': '#B8860B',
            'darkgreen': '#006400',
            'darkkhaki': '#BDB76B',
            'darkmagenta': '#8B008B',
            'darkolivegreen': '#556B2F',
            'darkorange': '#FF8C00',
            'darkorchid': '#9932CC',
            'darkred': '#8B0000',
            'darksalmon': '#E9967A',
            'darkseagreen': '#8FBC8F',
            'darkslateblue': '#483D8B',
            'darkslategray': '#2F4F4F',
            'darkturquoise': '#00CED1',
            'darkviolet': '#9400D3',
            'deeppink': '#FF1493',
            'deepskyblue': '#00BFFF',
            'dimgray': '#696969',
            'dodgerblue': '#1E90FF',
            'firebrick': '#B22222',
            'floralwhite': '#FFFAF0',
            'forestgreen': '#228B22',
            'fuchsia': '#FF00FF',
            'gainsboro': '#DCDCDC',
            'ghostwhite': '#F8F8FF',
            'goldenrod': '#DAA520',
            'green': '#008000',
            'greenyellow': '#ADFF2F',
            'honeydew': '#F0FFF0',
            'hotpink': '#FF69B4',
            'indianred': '#CD5C5C',
            'indigo': '#4B0082',
            'ivory': '#FFFFF0',
            'khaki': '#F0E68C',
            'lavender': '#E6E6FA',
            'lavenderblush': '#FFF0F5',
            'lawngreen': '#7CFC00',
            'lemonchiffon': '#FFFACD',
            'lime': '#00FF00',
            'limegreen': '#32CD32',
            'linen': '#FAF0E6',
            'magenta': '#FF00FF',
            'maroon': '#800000',
            'mintcream': '#F5FFFA',
            'mistyrose': '#FFE4E1',
            'moccasin': '#FFE4B5',
            'navajowhite': '#FFDEAD',
            'navy': '#000080',
            'oldlace': '#FDF5E6',
            'olive': '#808000',
            'olivedrab': '#6B8E23',
            'orange': '#FFA500',
            'orangered': '#FF4500',
            'orchid': '#DA70D6',
            'palegoldenrod': '#EEE8AA',
            'palegreen': '#98FB98',
            'paleturquoise': '#AFEEEE',
            'palevioletred': '#DB7093',
            'papayawhip': '#FFEFD5',
            'peachpuff': '#FFDAB9',
            'peru': '#CD853F',
            'pink': '#FFC0CB',
            'plum': '#DDA0DD',
            'powderblue': '#B0E0E6',
            'purple': '#800080',
            'rosybrown': '#BC8F8F',
            'royalblue': '#4169E1',
            'saddlebrown': '#8B4513',
            'salmon': '#FA8072',
            'sandybrown': '#FAA460',
            'seagreen': '#2E8B57',
            'seashell': '#FFF5EE',
            'sienna': '#A0522D',
            'skyblue': '#87CEEB',
            'slateblue': '#6A5ACD',
            'slategray': '#708090',
            'snow': '#FFFAFA',
            'springgreen': '#00FF7F',
            'steelblue': '#4682B4',
            'tan': '#D2B48C',
            'teal': '#008080',
            'thistle': '#D8BFD8',
            'tomato': '#FF6347',
            'turquoise': '#40E0D0',
            'violet': '#EE82EE',
            'wheat': '#F5DEB3',
            'white': '#FFFFFF',
            'whitesmoke': '#F5F5F5',
            'yellow': '#FFFF00',
            'yellowgreen': '#9ACD32'}

def vis_optics_dbscan(features):
    """
    Visualize optics and dbscan cluster results
    :param features: 2-D array
    """
    X = features
    clust = OPTICS(min_samples=2, xi=.05)

    # Run the fit
    clust.fit(features)

    labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=0.5)
    labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=1.0)

    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = X[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 0.5
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = X[labels_050 == klass]
        ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
    ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = X[labels_200 == klass]
        ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
    ax4.set_title('Clustering at 1.0 epsilon cut\nDBSCAN')

    plt.tight_layout()
    plt.show()

def concordancer(instances, cluster_method='kmeans', feature_type='sbert', lang_id='en', reduction_type='pca', n_components=50, model_path=None, debug=False):
    """
    Main function of a concordancer including extracting features, reduction, clustering similar samples and visualization
    :param instances: a list of triples, [left context], target word, [right context])
    :param cluster_method: str, 'kmeans', 'optics', 'dbscan', default is kmeans
    :param feature_type: str, 'sbert', 'word2vec', 'fasttext', 'glove', default is 'sbert'
    :param lang_id: str, 'en', 'fr'
    :param reduction_type: str, 'pca', 'tsne'
    :param n_components: int, reduced dimension
    :param model_path: str, paths of pre-trained models that used when you setting 'glove' or 'word2vec' as feature
    :param debug: boolean, if true, do visualization; if false, will not visualize
    :return: results, random scores and cluster scores
    """
    target_word = instances[0][1].upper()

    features = extract_features(instances,
                                lang_id=lang_id,
                                model_path=model_path,
                                feature_type=feature_type)
    full_dim_features = features
    if reduction_type:
        features = dimension_reduction(features,
                                       reduction_type=reduction_type,
                                       n_components=n_components)

    if cluster_method == 'dbscan':
        labels, centroids = dbscan(features)
    elif cluster_method == 'optics':
        labels, centroids = optics(features)
    else:
        labels, centroids = kmeans(features)

    # randomly picked n points
    random_centroids = sample(range(len(instances)), len(centroids))
    random_results = [instances[i] for i in random_centroids]
    results = [instances[i] for i in centroids]

    # rouge score
    random_rouge_score = rouge_calc(random_results)
    rouge_score = rouge_calc(results)

    # coinse distance with reduced dimension
    random_features = np.array([features[i] for i in random_centroids])
    centroid_features = np.array([features[i] for i in centroids])
    random_cosine_score_reduced = cosine_calc(random_features)
    cosine_score_reduced = cosine_calc(centroid_features)

    # coinse distance with full dimension
    random_features = np.array([full_dim_features[i] for i in random_centroids])
    centroid_features = np.array([full_dim_features[i] for i in centroids])
    random_cosine_score_full = cosine_calc(random_features)
    cosine_score_full = cosine_calc(centroid_features)

    random_scores = {'rouge': random_rouge_score, 'cosine_full': random_cosine_score_full, 'cosine_reduced': random_cosine_score_reduced}
    cluster_scores = {'rouge': rouge_score, 'cosine_full': cosine_score_full, 'cosine_reduced': cosine_score_reduced}

    if debug:
        print('Rouge Score(random):', random_rouge_score)
        print('Rouge Score({}):'.format(cluster_method), rouge_score)
        print('Cosine Score(random)-reduced:', random_cosine_score_reduced)
        print('Cosine Score({})-reduced:'.format(cluster_method), cosine_score_reduced)
        print('Cosine Score(random)-full dimension:', random_cosine_score_full)
        print('Cosine Score({})-full dimension:'.format(cluster_method), cosine_score_full)

        plt.figure(figsize=(5, 5))
        colors = sample(list(cnames.keys()), min(len(cnames.keys()), len(centroids)))
        for klass, color in zip(range(0, min(len(cnames.keys()), len(centroids))), colors):
            Xk = features[labels == klass]
            plt.scatter(Xk[:, 0], Xk[:, 1], marker='.', color=color, alpha=0.3)
        plt.scatter(features[centroids, 0], features[centroids, 1], marker='+', color='red', label='centriods', alpha=0.7)
        plt.scatter(features[random_centroids, 0], features[random_centroids, 1], marker='*', color='black', label='random points', alpha=0.6)
        plt.scatter(features[labels == -1, 0], features[labels == -1, 1], marker='x', color='lightgray', label='noise', alpha=0.1)
        plt.legend()
        plt.title(cluster_method.upper())
        plt.xlabel(target_word)
        if not os.path.exists('../results'):
            os.mkdir('../results')
        plt.savefig('../results/' + cluster_method + '_' + target_word + '.png')
        # plt.show()

    return results, random_scores, cluster_scores

def dbscan(features, eps=1, min_samples=2):
    """
    sklearn dbscan clustering algorithm

    :param features: 2-D narray, features of a list of instances
    :param eps: epsilon, https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    :param min_samples: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    :return: labels, indices of centroids
    """
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = model.labels_
    centroids = []
    label_set = set()
    for i in range(len(labels)):
        if labels[i] == -1:
            continue
        if labels[i] not in label_set:
            label_set.add(labels[i])
            centroids.append(i)
    return labels, centroids

def optics(features, min_samples=2):
    """
    sklearn optics clustering algorithm

    :param features: 2-D narray, features of a list of instances
    :param min_samples: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS
    :return: labels, indices of centroids
    """
    assert min_samples <= len(features), "Number of instances is less than min_samples!"

    model = OPTICS(min_samples=min_samples).fit(features)
    labels = model.labels_
    core_distances_ = model.core_distances_
    label_lookup = {}
    for i in range(len(labels)):
        if labels[i] == -1:
            continue
        if labels[i] not in label_lookup:
            label_lookup[labels[i]] = i
        else:
            # for points in each cluster, pick the point which has the minimum core distance as a centroid
            if core_distances_[i] < core_distances_[label_lookup[labels[i]]]:
                label_lookup[labels[i]] = i

    return labels, list(label_lookup.values())

def kmeans(features, cv=1, start=2, end=50):
    """
    Takes in a list of context vectors, creates many kmeans clustering models (specified by cv), and uses
    avg_inertia_elbow to establish the optimal value of k clusters. Plot determines whether to plot the distribution
    of inertias with the optimal value specified by a vertical line

    :param features: 2-D narray, features of a list of instances
    :param cv: int, cross validation
    :param start: int, thr minimum number of clusters
    :param end: int, the maximum number of clusters
    :return: labels, indices of centroids
    """
    assert start > 0 and type(start) == int, "You must give a starting value greater than zero that is an integer"
    assert end > start and type(end) == int, "You must give an ending value greater than start that is an integer"
    assert cv > 0 and type(cv) == int, "You must give a cv value greater than zero that is an integer"
    end = min(len(features), end)

    avg_inertias = []
    for k in range(start, end):
        inertias = []
        for c in range(cv):
            inertias.append(KMeans(n_clusters=k).fit(features).inertia_)
        avg_inertias.append(np.mean(inertias))
    k = avg_inertia_elbow(avg_inertias, start, end, plot=False)
    model = KMeans(n_clusters=k).fit(features)

    closest_to_centroids, _ = pairwise_distances_argmin_min(model.cluster_centers_, features)
    labels = model.labels_

    return labels, closest_to_centroids


def avg_inertia_elbow(avg_inertias, start, end, plot=False):
    """
    Takes distribution of average interitas in accordance to models using k clusters. Returns the determined
    elbow of said distribution by finding the distribution of all second order derivatives, finding the minimum
    and using this minimum in a calculation for a threshold of determining the distribution's linear trend.
    Setting argument plot to True will plot the distribution of inertias

    :param avg_inertias: array, inertias returned by kmeans model
    :param start: int, the minimum number of clusters
    :param end: int, the maximum number of clusters
    :param plot: boolean, if true, do visualization; if not, will not show
    :return: int, optimized number of clusters
    """

    assert start > 0 and type(start) == int, "You must give a starting value greater than zero that is an integer"
    assert end > start and type(end) == int, "You must give an ending value greater than start that is an integer"
    
    deriv_dist = []
    optimal_k = 2
    for j, k in enumerate(avg_inertias):
        if j >= 2:
            xi_1 = avg_inertias[j-1]
            xi_2 = avg_inertias[j-2]
            xi = k
            second_derivative = xi - xi_1 - (xi_1 - xi_2)
            deriv_dist.append(abs(second_derivative))


    mean_deriv = np.mean(deriv_dist)
    smallest_gap = 999
    for i, deriv in enumerate(deriv_dist):
        if deriv < mean_deriv and mean_deriv - deriv < smallest_gap: 

            smallest_gap = mean_deriv - deriv
            optimal_k = i
        if deriv > mean_deriv and deriv - mean_deriv < smallest_gap:

            smallest_gap = deriv - mean_deriv
            optimal_k = i
            
    if plot:             
        w = 15
        h = 17
        plt.figure(figsize = (w,h))
        plt.plot(range(start, end),[k for k in avg_inertias], '-o')
        plt.axvline(x=optimal_k)
        ax = plt.gca()
        ax.tick_params('both', labelsize=(w+h)/2)
        ax.set_xlabel('K', fontsize=w)
        ax.set_ylabel("Inertia", fontsize=w)

    return optimal_k


def distance_distribution(features, plot=False):
    """
    Used for picking optimal eps for DBSCAN and OPTICS

    :param features: 2-D narray, features of a list of instances
    :param plot: boolean, if true, do visualization; if not, will not show
    """
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(features)
    distances, indices = nbrs.kneighbors(features)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    if plot:
        w = 12
        h = 14
        plt.figure(figsize=(w, h))
        plt.plot(distances, '-o')
        # plt.axvline(x=optimal_eps);
        ax = plt.gca()
        ax.tick_params('both', labelsize=(w + h) / 2)
        ax.set_xlabel('Example', fontsize=w)
        ax.set_ylabel("Epsilon", fontsize=w)
        plt.show()
