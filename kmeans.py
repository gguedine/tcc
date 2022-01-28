import lemmatize as lem
import clustering as cl
import os
import pca_tsne as pt
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from ast import literal_eval
import loaded_tfidf as l_tfidf
import importlib
import pdb
# >>> import some_module as sm
# ...
# >>> import importlib
# >>> importlib.reload(pt)
# >>> importlib.reload(some_module) # raises "NameError: name 'some_module' is not defined"
# >>> importlib.reload(sm) # works
BASE_DIR = os.getcwd()
clusters_size_keywords = []
kmeans_size_keywords = []

data = pd.read_csv('select.csv', sep=';', quotechar='"')


# new_dataset = data.iloc[:5]
examples = data.sample(frac=0.2, random_state=1)

# pdb.set_trace()
# data = data.iloc[5:].reset_index(drop=True)
data = data.drop(examples.index).reset_index(drop=True)
new_dataset = examples.reset_index(drop=True)

new_dataset = lem.perform_lemmatize_dataset(new_dataset)
data = lem.perform_lemmatize_dataset(data)

# examples = data.iloc[:5]
# examples = lem.perform_lemmatize_dataset(examples)
# data = data.iloc[5:].reset_index(drop=True)
# data = lem.perform_lemmatize_dataset(data)

general_words = ["data_", "use", "using", "used", "paper", "method", "analysis", "area",
                 "proper", "total", "different", "based", "result", "problem", "furthermore",
                 "propose", "important", "general", "approach", "present", "aim", "work",
                 "make", "goal", "exist", "like", "new", "12", "xxxix", "rio", "grande", "nbsp"]

my_stop_words = text.ENGLISH_STOP_WORDS.union(general_words)

tfidf = TfidfVectorizer(
    min_df=5,
    max_df=0.95,
    max_features=8000,
    stop_words=my_stop_words
)


def conventinal(file_name ="classic_kmeans"):
    conventional_labels, conventional_kmeans_centroids = cl.conventional_kmeans(data, tfidf, kmeans_size_keywords, 11)
    array = np.array(kmeans_size_keywords)
    np.savetxt(os.path.join(BASE_DIR, "dados_salvos", file_name+"_data.csv"), array, delimiter=";",
               fmt='% s', comments='',
               header="cluster_id;size;top_keywords;cluster_elements;cluster_center")
    datf = pd.read_csv(os.path.join(BASE_DIR, "dados_salvos", file_name+"_data.csv"),
                     converters={'cluster_elements': literal_eval, 'cluster_center': literal_eval},
                     sep=';')

    np.savetxt(os.path.join(BASE_DIR, "dados_salvos", file_name+"_labels.txt"), np.atleast_2d(conventional_labels), delimiter=";", fmt='%d')
    return datf, conventional_labels


def iteractive(file_name ="interactive_kmeans"):
    iteractive_labels = cl.iteractive_kmeans(data, tfidf, clusters_size_keywords, 250)
    array = np.array(clusters_size_keywords)
    np.savetxt(os.path.join(BASE_DIR, "dados_salvos", file_name+"_data.csv"), array, delimiter=";",
               fmt='% s', comments='',
               header="cluster_id;size;top_keywords;cluster_elements;cluster_center")
    datf = pd.read_csv(os.path.join(BASE_DIR, "dados_salvos", file_name+"_data.csv"),
                     converters={'cluster_elements': literal_eval, 'cluster_center': literal_eval},
                     sep=';')
    # labels = np.full(data.index.size, -1)
    # for i, row in df[['cluster_id', 'cluster_elements']].iterrows():
    #     index = data.index[data['submission_id'].isin(row['cluster_elements'])].tolist()
    #     labels[index] = row['cluster_id']

    np.savetxt(os.path.join(BASE_DIR, "dados_salvos", file_name+"_labels.txt"), np.atleast_2d(iteractive_labels), delimiter=";", fmt='%d')
    return datf, iteractive_labels


# pt.plot_tsne_pca_with_centroids(tfidf.fit_transform(data.setting_value),
#                                 conventional_labels,
#                                 conventional_kmeans_centroids)
#
# pt.plot_tsne_pca_with_centroids_and_newdata(tfidf.fit_transform(data.setting_value),
#                                 conventional_labels,
#                                 conventional_kmeans_centroids,
#                                 tfidf.transform(examples.setting_value))


# saving original tfidf
# reset original tfidf
df, labels = conventinal(file_name="classic_kmeans_test_k_11_random_1")
# cl.elbow_method(data, tfidf)
# pdb.set_trace()
tfidf.fit_transform(data.setting_value)
l_tfidf.save(tfidf, data, "tfidf_values_k_11_random_1")

centroids = np.array(df['cluster_center'].values.tolist())
# pt.plot_tsne_pca_with_centroids_and_newdata(tfidf.fit_transform(data.setting_value),
#                                             labels,
#                                             centroids,
#                                             tfidf.transform(examples.setting_value))
##READS SAVED JSON##
# tfidf_values = json.load(open("tfidf_values.json"))
# tfidf_values['idf'] = np.asarray(tfidf_values['idf'], dtype=np.float64)
# tfidf_values['vocabulary'] = {k: np.int64(v) for k, v in tfidf_values['vocabulary'].items()}


### READS SAVED RESULT FROM ITERACTIVE CLUSTERIZATION SAVING TIME
# buffered_iteractive_kmeans = open(os.path.join(BASE_DIR, "dados_salvos","iteractive_kmeans_data.txt").read().splitlines()
# iteractive_kmeans_kw = []
# for bik in buffered_iteractive_kmeans:
#    iteractive_kmeans_kw.append(bik[bik.find("'"):-1])

# conventional_labels = cl.conventional_kmeans(data, tfidf, kmeans_size_keywords, 10)
