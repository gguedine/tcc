import matplotlib.pyplot as plt
import lemmatize as lem
import os
import pca_tsne as pt
import pandas as pd
import numpy as np
import loaded_tfidf as l_tfidf
from ast import literal_eval
from scipy.spatial import distance_matrix
import word_cloud as wc
import pdb

BASE_DIR = os.getcwd()
csv_file_name = os.path.join(BASE_DIR,"dados_salvos","classic_kmeans_test_2_data.csv")
txt_file_name = os.path.join(BASE_DIR,"dados_salvos","'classic_kmeans_test_2_labels.txt")

loaded_tfidf = l_tfidf.load()
data = pd.read_csv('select.csv', sep=';', quotechar='"')
# new_dataset = data.iloc[:5]
# new_dataset = lem.perform_lemmatize_dataset(new_dataset)
#
# data = data.iloc[5:].reset_index(drop=True)
# data = lem.perform_lemmatize_dataset(data)

new_dataset = data.sample(frac=0.2, random_state=0)
data = data.drop(new_dataset.index).reset_index(drop=True)
new_dataset = new_dataset.reset_index(drop=True)

new_dataset = lem.perform_lemmatize_dataset(new_dataset)
data = lem.perform_lemmatize_dataset(data)

new_dataset_matrix = np.array(loaded_tfidf.transform(new_dataset.setting_value).todense())
# new_dataset_matrix = np.array(loaded_tfidf.transform([new_data.setting_value]).todense())
df = pd.read_csv(csv_file_name,
                 converters={'cluster_elements': literal_eval, 'cluster_center': literal_eval},
                 sep=';')



def sort_similars_by_distance_matrix(new_data, similar_data_slice, loaded_tfidf, size=None):
    if (size is None) or (size > len(similar_data_slice)):
        size = len(similar_data_slice)
    sim_elements = similar_data_slice.setting_value
    similar_elements_matrix = np.array(loaded_tfidf.transform(sim_elements).todense())
    new_element_matrix = new_data
    distance_array = distance_matrix(new_element_matrix, similar_elements_matrix)[0]
    sorted_index = np.argsort(distance_array)
    sorted_similar_elements = similar_data_slice.iloc[sorted_index][:size]
    return sorted_similar_elements


# def predicted_labels():
#     centroids = np.array(df['cluster_center'].values.tolist())
#     distances = distance_matrix(new_dataset_matrix, centroids)
#     predict = distances.argmin(axis=1)
#     return predict
#
#
# # select needed clusters and data
# predict = predicted_labels()
# pdb.set_trace()
centroids = np.array(df['cluster_center'].values.tolist())

distances = distance_matrix(new_dataset_matrix, centroids)
predict = distances.argmin(axis=1)
# select needed clusters and data
needed_clusters = np.unique(predict)
needed_data = pd.DataFrame({})
for i in needed_clusters:
    needed_data = needed_data.append(data.loc[data['submission_id'].isin(df['cluster_elements'][i])], ignore_index=True)

needed_data = lem.perform_lemmatize_dataset(needed_data)


def build_recommendation(size=None):
    recommendation = pd.DataFrame({})
    for ind in range(len(new_dataset)):
        closest_cluster = distances[ind].argmin()
        similar_elements = needed_data.loc[needed_data['submission_id'].isin(df['cluster_elements'][closest_cluster])]
        sorted_similart_elements = sort_similars_by_distance_matrix([new_dataset_matrix[ind]], similar_elements,
                                                                   loaded_tfidf, size=size)
        new_data_dat = new_dataset.iloc[ind]
        ex = {'new_data': pd.DataFrame(new_data_dat).T,
              'closest_cluster': closest_cluster,
              # 'similar_elements': sort_similars_by_distance_matrix([new_dataset_matrix[ind]], similar_elements, loaded_tfidf)
              'similar_elements': sorted_similart_elements,
              'similar_elements_submission_id': sorted_similart_elements['submission_id'].array
              }
        recommendation = recommendation.append(ex, ignore_index=True)

    return recommendation


recommend = build_recommendation(size=50)
print(recommend)

BASE_DIR = os.getcwd()
file_name = os.path.join(BASE_DIR,"dados_salvos","recommmendation_data.pkl")
recommend.to_pickle(file_name)
load_recommend = pd.read_pickle(file_name)



def word_cloud_plot():
    f, ax = plt.subplots(1, 5, figsize=(20, 10))
    # ind = 0
    # for ind in range(len(new_dataset)):
    #     # for j in  len(recommendation):
    #     wc.plot_word_cloud_only_dataset(recommendation.iloc[0].new_data, recommendation.iloc[ind].similar_elements,
    #                                         all_new_features=True,plt=ax[ind])
    #ficou legal usando recommend.iloc[1 ou 4 ou 7].new_data,
    wc.plot_word_cloud_only_dataset(recommend.iloc[7].new_data, recommend.iloc[0].similar_elements,
                                    all_new_features=True, plt=ax[0])
    wc.plot_word_cloud_only_dataset(recommend.iloc[7].new_data, recommend.iloc[1].similar_elements,
                                    all_new_features=True, plt=ax[1])
    wc.plot_word_cloud_only_dataset(recommend.iloc[7].new_data, recommend.iloc[2].similar_elements,
                                    all_new_features=True, plt=ax[2])
    wc.plot_word_cloud_only_dataset(recommend.iloc[7].new_data, recommend.iloc[7].similar_elements,
                                    all_new_features=True, plt=ax[3])
    wc.plot_word_cloud_only_dataset(recommend.iloc[7].new_data, recommend.iloc[4].similar_elements,
                                    all_new_features=True, plt=ax[4])
    plt.subplots_adjust(wspace=.1, left=.01, right=.99)
    plt.show()

# word_cloud_plot()