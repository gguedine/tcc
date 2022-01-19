import matplotlib.pyplot as plt
import lemmatize as lem
import os
import pandas as pd
import numpy as np
import loaded_tfidf as l_tfidf
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn import metrics
import word_cloud as wc
import pdb

BASE_DIR = os.getcwd()
csv_file_name = os.path.join(BASE_DIR,"dados_salvos","classic_kmeans_test_2_data.csv")
txt_file_name = os.path.join(BASE_DIR,"dados_salvos","'classic_kmeans_test_2_labels.txt")

loaded_tfidf = l_tfidf.load()
data = pd.read_csv('select.csv', sep=';', quotechar='"')
# new_dataset = data.iloc[:5]
new_dataset = data.sample(frac=0.2, random_state=0)

# pdb.set_trace()
# data = data.iloc[5:].reset_index(drop=True)
data = data.drop(new_dataset.index).reset_index(drop=True)
new_dataset = new_dataset.reset_index(drop=True)

new_dataset = lem.perform_lemmatize_dataset(new_dataset)
data = lem.perform_lemmatize_dataset(data)

new_dataset_matrix = np.array(loaded_tfidf.transform(new_dataset.setting_value).todense())
dataset_matrix = np.array(loaded_tfidf.transform(data.setting_value).todense())

df = pd.read_csv(csv_file_name,
                 converters={'cluster_elements': literal_eval, 'cluster_center': literal_eval},
                 sep=';')
labels = np.full(data.index.size, -1)
for i, row in df[['cluster_id', 'cluster_elements']].iterrows():
    index = data.index[data['submission_id'].isin(row['cluster_elements'])].tolist()
    labels[index] = row['cluster_id']


def predicted_labels():
    # Maximum accuracy: 0.8308641975308642 at K = 48, RANGE(1,50)
    # Maximum accuracy:- 0.8221709006928406 at K = 46, RANGE(1,50)
    knn = KNeighborsClassifier(n_neighbors=46).fit(dataset_matrix, labels)
    predicted_labels = knn.predict(new_dataset_matrix)
    return predicted_labels


predict = predicted_labels()


def build_recommendation(size=None):  # for all cluster size
    if size is None:
        recommendation = pd.DataFrame({})
        for ind in range(len(new_dataset)):
            size = df[df['cluster_id'] == ind]['size'].values[0]
            nn = NearestNeighbors(n_neighbors=size).fit(dataset_matrix)
            similar_elements_ind = nn.kneighbors([new_dataset_matrix[ind]], return_distance=False)[0]
            closest_cluster = predict[ind]
            similar_elements = data.iloc[similar_elements_ind]
            new_data_dat = new_dataset.iloc[ind]
            ex = {'new_data': pd.DataFrame(new_data_dat).T,
                  'closest_cluster': closest_cluster,
                  'similar_elements': similar_elements,
                  'similar_elements_submission_id': similar_elements['submission_id'].array
                  }
            recommendation = recommendation.append(ex, ignore_index=True)

        return recommendation
    else:
        recommendation = pd.DataFrame({})
        nn = NearestNeighbors(n_neighbors=size).fit(dataset_matrix)
        similar_elements_ind = nn.kneighbors(new_dataset_matrix, return_distance=False)
        for ind in range(len(new_dataset)):
            closest_cluster = predict[ind]
            similar_elements = data.iloc[similar_elements_ind[ind]]
            new_data_dat = new_dataset.iloc[ind]
            ex = {'new_data': pd.DataFrame(new_data_dat).T,
                  'closest_cluster': closest_cluster,
                  'similar_elements': similar_elements,
                  'similar_elements_submission_id': similar_elements['submission_id'].array
                  }
            recommendation = recommendation.append(ex, ignore_index=True)

        return recommendation


recommend = build_recommendation(size=50)
print(recommend)

file_name = os.path.join(BASE_DIR,"dados_salvos","knn_data.pkl")
recommend.to_pickle(file_name)
load_recommend = pd.read_pickle(file_name)

def get_best_accuracy(dataset_matrix, labels, n=50): #calculate better k(n_neighbors)
    X_train, X_test, y_train, y_test = train_test_split(dataset_matrix, labels, test_size=0.2)
    acc = []
    for i in range(1, n):
        neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        yhat = neigh.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, yhat))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n), acc, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', markersize=10)
    plt.title('accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))
    plt.show()
    return acc.index(max(acc))

# get_best_accuracy(dataset_matrix, labels)