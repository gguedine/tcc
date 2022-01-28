import numpy as np
import json
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text


def save(old_tfidf, dataset, file_name="tfidf_values"):
    #reset original_tfidf
    old_tfidf.fit_transform(dataset.setting_value)
    vocabulary = {k: str(v) for k, v in old_tfidf.vocabulary_.items()}
    ids = [str(v) for v in old_tfidf.idf_]
    tfidf_json = {'idf': ids, 'vocabulary': vocabulary}
    json.dump(tfidf_json, open(file_name+".json", 'w'))
    return


def load( file_name="tfidf_values.json"):
    tfidf_values = json.load(open(file_name+".json"))
    tfidf_values['idf'] = np.asarray(tfidf_values['idf'], dtype=np.float64)
    tfidf_values['vocabulary'] = {k: np.int64(v) for k, v in tfidf_values['vocabulary'].items()}

    class LoadedTfidfVectorizer(TfidfVectorizer):
        # plug our pre-computed IDFs
        TfidfVectorizer.idf_ = tfidf_values['idf']

    general_words = ["data", "use", "using", "used", "paper", "method", "analysis", "area",
                     "proper", "total", "different", "based", "result", "problem", "furthermore",
                     "propose", "important", "general", "approach", "present", "aim", "work",
                     "make", "goal", "exist", "like", "new", "12", "xxxix", "rio", "grande", "nbsp"]

    my_stop_words = text.ENGLISH_STOP_WORDS.union(general_words)
    loaded_tfidf = LoadedTfidfVectorizer(
        min_df=5,
        max_df=0.95,
        max_features=8000,
        stop_words=my_stop_words
    )
    loaded_tfidf._tfidf._idf_diag = sp.spdiags(tfidf_values['idf'], diags=0,
                                               m=len(tfidf_values['idf']), n=len(tfidf_values['idf']))
    loaded_tfidf.vocabulary_ = tfidf_values['vocabulary']
    return loaded_tfidf


