import keywords as kw
import pca_tsne as pt
import math
import re
import numpy as np
import pandas as pd
import pdb

from collections import Counter
from sklearn.cluster import KMeans


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
def conventional_kmeans(data, tfidf, kmeans_size_keywords, k):
    # mix_diff = 9999
    # best_random = 0
    # for random in range(200, 1000):
    matrix = tfidf.fit_transform(data.setting_value)
    #melhor random_state entre 0 e 1000 = 560, menor diferença entre min e max ficou 252
    fit = KMeans(n_clusters=k, random_state=560).fit(matrix)
    means_clusters = fit.predict(matrix)
    distances = fit.transform(matrix)

    sse = 0
    i = 0
    for cluster in means_clusters:
        sse = sse + distances[i][cluster]
        i = i + 1
    print("\nSSE = {}".format(sse))

    ssd = 0
    i = 0
    for cluster in means_clusters:
        ssd = ssd + math.pow(distances[i][cluster] - (sse/2702), 2)
        i = i + 1
    ssd = math.sqrt(ssd/2702)
    print("\nSSD = {}".format(ssd))

    sizes = np.bincount(means_clusters)
    top_keywords = kw.get_top_keywords(matrix, means_clusters, tfidf.get_feature_names(), 10)

    index = 1
    for ind, size in enumerate(sizes):
        regex = "{}(.*)".format(index)
        cluster_keywords = re.search(regex, top_keywords).group(1)

        print("Cluster {}".format(index))
        print(cluster_keywords)

        element_positions = [i for i, value in enumerate(means_clusters) if value == ind]
        kmeans_size_keywords.append([ind,size, cluster_keywords,
                                     list(data.iloc[element_positions]['submission_id'].values),
                                     list(fit.cluster_centers_[ind])])
        index += 1



    print("\nClusters Size")
    print(sizes)
    #
    # diff = sizes.max() - sizes.min()
    # if diff < mix_diff:
    #     best_random = random
    #     mix_diff = diff
    #
    # # pdb.set_trace()
    # print("\nDiff: : {}".format(diff))
    # print("\nRandom: {}".format(random))
    # print("\nMin Diff: {}".format(mix_diff))
    # print("\nBest Random State: {}".format(best_random))

    return means_clusters, fit.cluster_centers_


def iteractive_kmeans(data, tfidf, clusters_size_keywords, t):
    sse = 0
    dists_array = []
    original_data = data.copy()
    original_matrix = tfidf.fit_transform(original_data.setting_value)
    while (data.size > 0):
        print("Applying TFIDF...\n")
        matrix = tfidf.fit_transform(data.setting_value)
        
        k = 1
        found = False
        while (found == False):
            print("Clustering with k = {}...".format(k))
            fit = KMeans(n_clusters=k, random_state=20).fit(matrix)
            means_clusters = fit.predict(matrix)
            distances = fit.transform(matrix)
            
            cluster_size = np.bincount(means_clusters)
            print("Clusters sizes = {}".format(cluster_size))
            
            min_sizes = sorted(i for i in cluster_size if i <= t)
            print("Min cluster sizes = {}\n".format(min_sizes))
            
            if min_sizes:
                rows_removal = []
                counts = Counter(means_clusters)
                print("Counter occurrences = {}\n".format(counts))
                
                for min_size in min_sizes:
                    min_element = list(counts.keys())[list(counts.values()).index(min_size)]
                    print("Current min_element = {}".format(min_element))
                    del counts[min_element]                
                    print("Removed min_element {} from counter {}\n".format(min_element, counts))

                    print("Removing smallest cluster elements = {} with occurrences = {}".format(min_element, min_size))
                    print("Getting element indexes...")
                    min_element_positions = [index for index, value in enumerate(means_clusters) if value == min_element]
                    rows_removal.extend(min_element_positions)

                    original_mean_elements = np.mean(original_matrix[rows_removal], axis=0, dtype=np.float64)
                    original_cluster_center = np.asarray(original_mean_elements).reshape(-1)

                    min_size_position = list(cluster_size).index(min_size) + 1
                    print("Getting cluster {} size and top keywords...".format(min_size_position))
                    top_keywords = kw.get_top_keywords(matrix, means_clusters, tfidf.get_feature_names(), 10)
                    regex = "{}(.*)".format(min_size_position)
                    min_cluster_keywords = re.search(regex, top_keywords).group(1)
                    print("Top keywords are {}\n".format(min_cluster_keywords))
                    
                    # clusters_size_keywords.append([min_size, min_cluster_keywords, data.iloc[min_element_positions]])
                    clusters_size_keywords.append([min_size, min_cluster_keywords,
                                                   list(data.iloc[min_element_positions]['submission_id'].values),
                                                   list(original_cluster_center)])

                print("Being removed {} elements...".format(len(rows_removal)))
                
                print("Old data size = {}".format(data.index))
                data = data.drop(data.index[rows_removal]).reset_index(drop=True)
                print("New data size = {}\n".format(data.index))
                
                for position in rows_removal:
                    print("Adding sse of element {} of cluster {}".format(position, means_clusters[position]))
                    sse = sse + distances[position][means_clusters[position]]
                    dists_array.append(distances[position][means_clusters[position]])
                print("Current sse = {}".format(sse))
                
                found = True
            else:    
                print("k = {} failed\n".format(k))
                k = k + 2
    ssd = 0
    for dist in dists_array:
        ssd = ssd + math.pow(dist - (sse/2702),2)
    ssd = math.sqrt(ssd/2702)
    print("Total ssd = {}".format(ssd))

    #creating labels
    labels = np.full(original_data.index.size, -1)
    for num, item in enumerate(clusters_size_keywords):
        item.insert(0, num)
    df = pd.DataFrame(clusters_size_keywords)
    df.columns = ["cluster_id", "size", "top_keywords", "cluster_elements", "cluster_center"]
    for i, row in df[['cluster_id', 'cluster_elements']].iterrows():
        index = original_data.index[original_data['submission_id'].isin(row['cluster_elements'])].tolist()
        labels[index] = row['cluster_id']
    #pt.plot_tsne_pca(tfidf.fit_transform(original_data.setting_value), labels)
    return labels
