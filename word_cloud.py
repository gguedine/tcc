import pandas as pd
import numpy as np
import keywords as kw
import matplotlib.pyplot as plt
import loaded_tfidf as l_tfidf
from wordcloud import WordCloud
import pdb

def scale_number(unscaled, to_min, to_max, from_min, from_max):
    return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min


def scale_list(l, to_min, to_max):
    return [scale_number(i, to_min, to_max, min(l), max(l)) for i in l]


def plot_word_cloud(new_data, dataset, n_terms=50, all_new_features=False):
    loaded_tfidf = l_tfidf.load()
    matrix_dataset = loaded_tfidf.transform(dataset.setting_value)
    matrix_data = loaded_tfidf.transform([new_data.setting_value])

    # filter only used features
    top_dataset_keywords = kw.get_top_keywords_by_data(matrix_dataset, loaded_tfidf.get_feature_names(),
                                                       n_terms=n_terms)
    if all_new_features:
        arr = np.array(matrix_data.todense())[0]
        ind = np.where(arr != 0)
        features_arr = np.array(loaded_tfidf.get_feature_names())
        top_data_keywords = features_arr[ind]
    else:
        top_data_keywords = kw.get_top_keywords_by_data(matrix_data, loaded_tfidf.get_feature_names(),
                                                        n_terms=n_terms)

    top_keywords = np.sort(np.unique(top_dataset_keywords + top_data_keywords))
    top_index = np.where(np.in1d(loaded_tfidf.get_feature_names(), top_keywords))[0]

    matrix_dataset = np.array(matrix_dataset.todense())[:, top_index]
    matrix_data = np.array(matrix_data.todense())[:, top_index]

    merged_matrix = np.r_[matrix_data, matrix_dataset]
    df = pd.DataFrame(merged_matrix, columns=top_keywords)

    frequency = df.T.sum(axis=1)
    wordcloud = WordCloud(  # font_path='fonts/Lato/Lato-Regular.ttf',
        background_color="white",
        width=3000, height=2000
    ).generate_from_frequencies(frequency)

    similars = list(set(top_data_keywords) & set(top_dataset_keywords))
    print("Similars words = [{}]".format(frequency[similars]))
    print("Similars words total frequency = {}...".format(frequency[similars].sum()))
    # word_to_color = dict()
    # for word in top_data_keywords:
    #     word_to_color[word] = "hsl(0, %d%%, %d%%)" % (random.randint(68, 75), random.randint(50, 70))

    top_data_keywords_set = set(top_data_keywords)

    # frequency = frequency*1000
    def color_function(word, *args, **kwargs):
        size = frequency[word]
        if word in similars:
            color = "hsl(0, 95%%, %d%%)" % (  # np.rint(scale_number(size, 68, 75, frequency.min(), frequency.max())),
                scale_number(size, 45, 50, frequency.min(), frequency.max()))
            # color = "hsl(0, %d%%, %d%%)" % (random.randint(68, 75), random.randint(50, 70))
        # try:
        #     color = word_to_color[word]
        #     "hsl(0, %d%%, %d%%)" % (random.randint(68, 75), random.randint(50, 70))
        # except KeyError:
        else:
            # color = "#000000"
            color = "hsl(0, 0%%, %d%%)" % (scale_number(size, 0, 1, frequency.min(), frequency.max()))
            # color = "hsl(0, 0%%, %d%%)" % (random.randint(0, 40))
        return color

    wordcloud.recolor(color_func=color_function)
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.tight_layout(pad=0)
    plt.axis("off")
    plt.show()
    return


def plot_word_cloud_only_dataset(dataset, new_data=None, n_terms=40, all_new_features=False, plt=plt):
    loaded_tfidf = l_tfidf.load("tfidf_values_k_11_random_1")
    matrix_dataset = loaded_tfidf.transform(dataset.setting_value)
    # filter only used features
    top_dataset_keywords = kw.get_top_keywords_by_data(matrix_dataset, loaded_tfidf.get_feature_names(),
                                                       n_terms=n_terms)
    if new_data is not None:
        matrix_data = loaded_tfidf.transform(new_data.setting_value)
        if all_new_features:
            arr = np.array(matrix_data.todense())[0]
            ind = np.where(arr != 0)
            features_arr = np.array(loaded_tfidf.get_feature_names())
            top_data_keywords = features_arr[ind]
        else:
            top_data_keywords = kw.get_top_keywords_by_data(matrix_data, loaded_tfidf.get_feature_names(), n_terms=n_terms)


    top_keywords = np.sort(np.unique(top_dataset_keywords))
    top_index = np.where(np.in1d(loaded_tfidf.get_feature_names(), top_keywords))[0]

    matrix_dataset = np.array(matrix_dataset.todense())[:, top_index]

    df = pd.DataFrame(matrix_dataset, columns=top_keywords)

    # pdb.set_trace()
    frequency = df.T.sum(axis=1)
    wordcloud = WordCloud(  # font_path='fonts/Lato/Lato-Regular.ttf',
        background_color="white",
        width=3000, height=2000
    ).generate_from_frequencies(frequency)

    print("\n**********\n")
    print("Top Keywords = [\n{}\n]".format(top_keywords))
    similars = None
    if new_data is not None:
        similars = list(set(top_data_keywords) & set(top_dataset_keywords))
        print("Similars words = [\n{}\n]".format(frequency[similars].to_string()))
        print("Similars words total frequency = {}...".format(frequency[similars].sum()))

    def color_function(word, *args, **kwargs):
        size = frequency[word]
        if (similars is not None) and (word in similars):
            color = "hsl(0, 95%%, %d%%)" % (  # np.rint(scale_number(size, 68, 75, frequency.min(), frequency.max())),
                scale_number(size, 40, 50, frequency.min(), frequency.max()))
                # scale_number(-size, 20, 40, -frequency.max(), -frequency.min()))
        else:
            color = "hsl(0, 0%%, %d%%)" % (scale_number(size, 0, 10, frequency.min(), frequency.max()))
            # color = "hsl(0, 0%%, %d%%)" % (scale_number(-size, 0, 25, -frequency.max(), -frequency.min()))
        return color


    wordcloud.recolor(color_func=color_function)


    # plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    # plt.tight_layout(pad=0)
    plt.axis("off")
    # plt.show(block=False)
    return plt
