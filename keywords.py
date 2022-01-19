import numpy as np
import pandas as pd


def get_top_keywords(dissim_matrix_data, clusters_labels, features, n_terms):
    df = pd.DataFrame(dissim_matrix_data.todense()).groupby(clusters_labels).mean()
    top_keywords = ""

    for i,r in df.iterrows():
        top_keywords += '\nCluster {}'.format(i+1)
        top_keywords += ','.join([features[t] for t in np.argsort(r)[-n_terms:]])

    return top_keywords


def get_top_keywords_by_data(dissim_matrix_data, features, n_terms):
    if n_terms > len(features):
        n_terms = len(features)

    df = pd.DataFrame(dissim_matrix_data.todense()).mean()
    #remove as features que terão valor zero dentro dos dados
    valid_df = df[df>0]
    valid_features = np.array(features)[valid_df.keys()]
    valid_top_keywords = [valid_features[t] for t in np.argsort(valid_df)[-n_terms:]]
    valid_top_keywords.reverse()
    return valid_top_keywords
    # top_keywords = [features[t] for t in np.argsort(df)[-n_terms:]]
    # top_keywords.reverse()
    # return top_keywords

