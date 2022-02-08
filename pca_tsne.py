import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=labels.size, replace=False)

    pca = PCA(n_components=2).fit_transform(data[max_items, :].todense())
    tsne = TSNE().fit_transform(PCA(n_components=72).fit_transform(data[max_items, :].todense()))

    idx = np.random.choice(range(pca.shape[0]), size=labels.size, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    plt.show()


def plot_tsne_pca2(data, labels):
    max_label = max(labels)
    matrix = data[:, :].todense()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(matrix)
    print('Total variation of components: {}'.format(pca.explained_variance_ratio_.cumsum()))

    tsne = TSNE()
    tsne_result = tsne.fit_transform(PCA(n_components=100).fit_transform(matrix))

    label_subset = labels[:]
    label_subset = [cm.hsv(i / max_label) for i in label_subset]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].scatter(pca_result[:, 0], pca[:, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(xs=tsne_result[:, 0], ys=tsne_result[:, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')

    plt.show()


# example
# pt.plot_tsne_pca_with_centroids(tfidf.fit_transform(data.setting_value),
# conventional_labels,
# conventional_kmeans_centroids
# )
def plot_tsne_pca_with_centroids(data, labels, centroids):
    max_label = max(labels)
    label_subset = labels[:]
    label_subset = [cm.hsv(i / max_label) for i in label_subset]

    matrix = np.array(data[:, :].todense())
    merged_matrix = np.r_[matrix, centroids]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(merged_matrix)
    pca_centroids = pca_result[-centroids.shape[0]:, :]
    pca_result = pca_result[:-centroids.shape[0], :]
    print('Total variation of components: {}'.format(pca.explained_variance_ratio_.cumsum()))

    pcs_t = PCA(n_components=100)
    tsne = TSNE()
    tsne_result = tsne.fit_transform(pcs_t.fit_transform(merged_matrix))
    tsne_centroids = tsne_result[-centroids.shape[0]:, :]
    tsne_result = tsne_result[:-centroids.shape[0], :]

    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].scatter(pca_result[:, 0], pca_result[:, 1], c=label_subset)
    ax[0].scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='x', s=80, color='k')
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], c=label_subset)
    ax[1].scatter(x=tsne_centroids[:, 0], y=tsne_centroids[:, 1], marker='x', s=80, color='k')
    ax[1].set_title('TSNE Cluster Plot')
    plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.show()


# example
# pt.plot_tsne_pca_with_centroids_and_newdata(tfidf.fit_transform(data.setting_value),
#                                 conventional_labels,
#                                 conventional_kmeans_centroids,
#                                 tfidf.transform(examples.setting_value))
def plot_tsne_pca_with_centroids_and_newdata(data, labels, centroids, newdata):
    max_label = max(labels)
    label_subset = labels[:]
    label_subset = [cm.hsv(i / max_label) for i in label_subset]

    matrix = np.array(data[:, :].todense())
    newdata = np.array(newdata.todense())
    merged_matrix = np.r_[newdata, matrix]
    merged_matrix = np.r_[merged_matrix, centroids]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(merged_matrix)
    pca_newdata = pca_result[:newdata.shape[0], :]
    pca_centroids = pca_result[-centroids.shape[0]:, :]
    pca_result = pca_result[newdata.shape[0]:-centroids.shape[0], :]
    print('Total variation of components: {}'.format(pca.explained_variance_ratio_.cumsum()))

    pcs_t = PCA(n_components=100)
    tsne = TSNE()
    tsne_result = tsne.fit_transform(pcs_t.fit_transform(merged_matrix))
    tsne_newdata = tsne_result[:newdata.shape[0], :]
    tsne_centroids = tsne_result[-centroids.shape[0]:, :]
    tsne_result = tsne_result[newdata.shape[0]:-centroids.shape[0], :]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    u_labels = np.unique(labels)
    for i in u_labels:
        ax[0].scatter(pca_result[np.where(labels == i), 0], pca_result[np.where(labels == i), 1], label=i)
    # ax[0].scatter(pca_result[:, 0], pca_result[:, 1], c=label_subset)
    ax[0].scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='x', s=80, color='k')
    ax[0].scatter(pca_newdata[:, 0], pca_newdata[:, 1], marker='*', s=80, color='k')
    ax[0].legend()
    ax[0].set_title('PCA Cluster Plot')

    for i in u_labels:
        ax[1].scatter(tsne_result[np.where(labels == i), 0], tsne_result[np.where(labels == i), 1], label=i)
    # ax[1].scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], c=label_subset)
    ax[1].scatter(x=tsne_centroids[:, 0], y=tsne_centroids[:, 1], marker='x', s=80, color='k')
    ax[1].scatter(x=tsne_newdata[:, 0], y=tsne_newdata[:, 1], marker='*', s=80, color='k')
    ax[1].legend()
    ax[1].set_title('TSNE Cluster Plot')
    plt.show()
