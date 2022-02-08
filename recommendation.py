import matplotlib.pyplot as plt
import pca_tsne as pt
import lemmatize as lem
import os
import pandas as pd
import numpy as np
import loaded_tfidf as l_tfidf
from ast import literal_eval
from scipy.spatial import distance_matrix
import word_cloud as wc
import pdb

BASE_DIR = os.getcwd()
csv_file_name = os.path.join(BASE_DIR,"dados_salvos","classic_kmeans_test_k_11_random_1_data.csv")
txt_file_name = os.path.join(BASE_DIR,"dados_salvos","'classic_kmeans_test_k_11_random_1_labels.txt")

loaded_tfidf = l_tfidf.load("tfidf_values_k_11_random_1")
data = pd.read_csv('select.csv', sep=';', quotechar='"')
# new_dataset = data.iloc[:5]
# new_dataset = lem.perform_lemmatize_dataset(new_dataset)
#
# data = data.iloc[5:].reset_index(drop=True)
# data = lem.perform_lemmatize_dataset(data)

new_dataset = data.sample(frac=0.2, random_state=1)
data = data.drop(new_dataset.index).reset_index(drop=True)
new_dataset = new_dataset.reset_index(drop=True)

new_dataset = lem.perform_lemmatize_dataset(new_dataset)
# data = lem.perform_lemmatize_dataset(data)

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
        distance = distances[ind].min()
        similar_elements = needed_data.loc[needed_data['submission_id'].isin(df['cluster_elements'][closest_cluster])]
        sorted_similart_elements = sort_similars_by_distance_matrix([new_dataset_matrix[ind]], similar_elements,
                                                                   loaded_tfidf, size=size)
        new_data_dat = new_dataset.iloc[ind]
        ex = {'submission_id': new_data_dat.submission_id,
              'new_data': pd.DataFrame(new_data_dat).T,
              'closest_cluster': closest_cluster,
              # 'distance': distance,
              # 'similar_elements': sort_similars_by_distance_matrix([new_dataset_matrix[ind]], similar_elements, loaded_tfidf)
              'similar_elements': sorted_similart_elements,
              'similar_elements_submission_id': sorted_similart_elements['submission_id'].array
              }
        recommendation = recommendation.append(ex, ignore_index=True)

    return recommendation


recommend = build_recommendation(size=50)
pd.set_option("min_rows", 50)
print(recommend)

BASE_DIR = os.getcwd()
file_name = os.path.join(BASE_DIR,"dados_salvos","recommendation_data_k11_random_1.pkl")
recommend.to_pickle(file_name)
load_recommendation = pd.read_pickle(file_name)
recommend = load_recommendation


def word_cloud_plot(submission_id, highlight=True):
    f, ax = plt.subplots(2, 2, figsize=(20, 10))
    # ind = 0
    # for ind in range(len(new_dataset)):
    #     # for j in  len(recommendation):
    #     wc.plot_word_cloud_only_dataset(recommendation.iloc[0].new_data, recommendation.iloc[ind].similar_elements,
    #                                         all_new_features=True,plt=ax[ind])
    # pdb.set_trace()
    example_index = recommend.loc[recommend['submission_id'] == submission_id].index[0]
    if highlight == True:
        example_new_data = recommend.iloc[example_index].new_data
    else:
        example_new_data = None
    used_clusters = [recommend.iloc[example_index].closest_cluster]
    others_index = []
    for ix in range(4):
        ex = recommend.loc[~recommend.closest_cluster.isin(used_clusters)].sample(n=1, random_state=1)
        others_index.append(ex.index[0])
        used_clusters.append(ex.closest_cluster.array[0])

    print("\nCurrent Example: ")
    print(recommend.iloc[[example_index]])
    wc.plot_word_cloud_only_dataset(recommend.iloc[example_index].similar_elements,
                                    new_data=example_new_data,
                                    all_new_features=True, plt=ax[0][0])

    print("\nCurrent Example: ")
    print(recommend.iloc[[others_index[0]]])
    wc.plot_word_cloud_only_dataset(recommend.iloc[others_index[0]].similar_elements,
                                    new_data=example_new_data,
                                    all_new_features=True, plt=ax[0][1])

    print("\nCurrent Example: ")
    print(recommend.iloc[[others_index[1]]])
    wc.plot_word_cloud_only_dataset(recommend.iloc[others_index[1]].similar_elements,
                                    new_data=example_new_data,
                                    all_new_features=True, plt=ax[1][0])

    print("\nCurrent Example: ")
    print(recommend.iloc[[others_index[2]]])
    wc.plot_word_cloud_only_dataset(recommend.iloc[others_index[2]].similar_elements,
                                    new_data=example_new_data,
                                    all_new_features=True, plt=ax[1][1])

    # print("\nCurrent Example: ")
    # print(recommend.iloc[[others_index[3]]])
    # wc.plot_word_cloud_only_dataset(example_new_data,
    #                                 recommend.iloc[others_index[3]].similar_elements,
    #                                 all_new_features=True, plt=ax[4])
    # plt.subplots_adjust(wspace=.1, left=.01, right=.99)
    plt.subplots_adjust(wspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

def word_cloud_plot_example_own_cluster(submission_ids):
    if len(submission_ids) > 5:
        return print("Deve ser passado no máximo 5 ids!")
    f, ax = plt.subplots(len(submission_ids),1, figsize=(20, 10))
    # ind = 0
    # for ind in range(len(new_dataset)):
    #     # for j in  len(recommendation):
    #     wc.plot_word_cloud_only_dataset(recommendation.iloc[0].new_data, recommendation.iloc[ind].similar_elements,
    #                                         all_new_features=True,plt=ax[ind])

    for idx, val in enumerate(submission_ids):
        example_index = recommend.loc[recommend['submission_id'] == val].index[0]
        print("\nCurrent Example: ")
        print(recommend.iloc[[example_index]])
        wc.plot_word_cloud_only_dataset(recommend.iloc[example_index].similar_elements,
                                        new_data=recommend.iloc[example_index].new_data,
                                        all_new_features=True, plt=ax[idx])


    plt.subplots_adjust(wspace=.1, left=.01, right=.99)
    plt.show()

def word_cloud_plot_example_own_cluster_no_highligth(submission_ids):
    if len(submission_ids) > 5:
        return print("Deve ser passado no máximo 5 ids!")
    f, ax = plt.subplots(len(submission_ids),1, figsize=(20, 10))
    # ind = 0
    # for ind in range(len(new_dataset)):
    #     # for j in  len(recommendation):
    #     wc.plot_word_cloud_only_dataset(recommendation.iloc[0].new_data, recommendation.iloc[ind].similar_elements,
    #                                         all_new_features=True,plt=ax[ind])

    for idx, val in enumerate(submission_ids):
        example_index = recommend.loc[recommend['submission_id'] == val].index[0]
        print("\nCurrent Example: ")
        print(recommend.iloc[[example_index]])
        wc.plot_word_cloud_only_dataset(recommend.iloc[example_index].similar_elements,
                                        new_data=None,
                                        all_new_features=True, plt=ax[idx])


    plt.subplots_adjust(wspace=.1, left=.01, right=.99)
    plt.show()

def word_cloud_plot_1():
    f, ax = plt.subplots(2, 2, figsize=(20, 10))
    #ficou legal usando recommend.iloc[1 ou 4 ou 7].new_data,
    wc.plot_word_cloud_only_dataset(recommend.iloc[526].similar_elements,
                                    new_data=recommend.iloc[526].new_data,
                                    all_new_features=True, plt=ax[0][0])
    wc.plot_word_cloud_only_dataset(recommend.iloc[0].similar_elements,
                                    new_data=recommend.iloc[526].new_data,
                                    all_new_features=True, plt=ax[0][1])
    wc.plot_word_cloud_only_dataset(recommend.iloc[4].similar_elements,
                                    new_data=recommend.iloc[526].new_data,
                                    all_new_features=True, plt=ax[1][0])
    wc.plot_word_cloud_only_dataset(recommend.iloc[1].similar_elements,
                                    new_data=recommend.iloc[526].new_data,
                                    all_new_features=True, plt=ax[1][1])
    plt.subplots_adjust(wspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

def word_cloud_plot_2():
    f, ax = plt.subplots(2, 2, figsize=(20, 10))
    #ficou legal usando recommend.iloc[1 ou 4 ou 7].new_data,
    wc.plot_word_cloud_only_dataset(recommend.iloc[526].similar_elements,
                                    new_data=recommend.iloc[1].new_data,
                                    all_new_features=True, plt=ax[0][0])
    wc.plot_word_cloud_only_dataset(recommend.iloc[0].similar_elements,
                                    new_data=recommend.iloc[1].new_data,
                                    all_new_features=True, plt=ax[0][1])
    wc.plot_word_cloud_only_dataset(recommend.iloc[4].similar_elements,
                                    new_data=recommend.iloc[1].new_data,
                                    all_new_features=True, plt=ax[1][0])
    wc.plot_word_cloud_only_dataset(recommend.iloc[1].similar_elements,
                                    new_data=recommend.iloc[1].new_data,
                                    all_new_features=True, plt=ax[1][1])
    plt.subplots_adjust(wspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# pdb.set_trace()
# word_cloud_plot_1()#sub_id:6095 cluster 8
# word_cloud_plot_2()#sub_id:5972 cluster 5
# word_cloud_plot(7431,highlight=False)
# word_cloud_plot(5689,highlight=False)
# word_cloud_plot(6496)
# word_cloud_plot(5906)
# word_cloud_plot(5972)
# word_cloud_plot(5689)
# word_cloud_plot_example_own_cluster([7431,5906,6095,5972,5689])
#recommend.loc[recommend['submission_id'].isin([7431,6095,5906,5972,5689])]
#      closest_cluster submission_id
# 0                0.0    5906
# 1                5.0    5972
# 4                1.0    5689
# 5                6.0    7431
# 526              8.0      6095
