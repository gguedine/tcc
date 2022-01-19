import os
import pandas as pd
import numpy as np
from ast import literal_eval
import pdb


BASE_DIR = os.getcwd()
recm_file_name = os.path.join(BASE_DIR,"dados_salvos","recommmendation_data.pkl")
knn_file_name = os.path.join(BASE_DIR,"dados_salvos","knn_data.pkl")

load_recommendation = pd.read_pickle(recm_file_name)
recm_new_data = pd.concat(load_recommendation['new_data'].array)
recm_similar_elements = pd.DataFrame(load_recommendation['similar_elements'].array)
recm_similar_elements_id = pd.DataFrame(load_recommendation['similar_elements_submission_id'].array)
recm_clusters = pd.DataFrame(load_recommendation['closest_cluster'].array)

load_knn = pd.read_pickle(knn_file_name)
knn_new_data = pd.concat(load_knn['new_data'].array)
knn_similar_elements = pd.DataFrame(load_knn['similar_elements'].array)
knn_similar_elements_id = pd.DataFrame(load_knn['similar_elements_submission_id'].array)
knn_clusters = pd.DataFrame(load_knn['closest_cluster'].array)

d = {
    'example_submission_id': np.atleast_2d(knn_new_data['submission_id'].values).T,
    'recm_sub_ids' : recm_similar_elements_id.values,
    'knn_sub_ids' : knn_similar_elements_id.values
}

arr = np.concatenate((d['example_submission_id'], d['recm_sub_ids'],d['knn_sub_ids']), axis=1)
recm_size = (arr.shape[1] - 1)/2
dict = {
    'example_submission_id': arr[:,0],
    'recm_sub_ids' : arr[:, 1:int(recm_size+1)].tolist(),
    'knn_sub_ids' : arr[:, int(recm_size+1):].tolist()
}

df = pd.DataFrame(dict)
df['matchs'] = [list(set(a).intersection(set(b))) for a, b in zip(df.knn_sub_ids, df.recm_sub_ids)]
df['erros'] = [list(set(a).difference(set(b))) for a, b in zip(df.knn_sub_ids, df.recm_sub_ids)]
df['matchs_count'] = df.matchs.apply(lambda x: len(x))
df['matchs_rate'] = df.matchs_count.apply(lambda x: x*100/50)
df['erros_count'] = df.erros.apply(lambda x: len(x))
df['errors_rate'] = df.erros_count.apply(lambda x: x*100/50)

file_name='compare_result'
result_array = np.array(df[['example_submission_id','matchs_count','matchs_rate','erros_count','errors_rate','matchs','erros']])
np.savetxt(os.path.join(BASE_DIR, "dados_salvos",file_name +"_.csv"), result_array, delimiter=";",
               fmt='% s', comments='',
               header="submission_id;matchs_count;matchs_rate;erros_count;errors_rate;matchs;erros")

datf = pd.read_csv( os.path.join(BASE_DIR, "dados_salvos", file_name+"_.csv"),
                    converters={'matchs': literal_eval, 'erros': literal_eval},
                    sep=';')
print(datf)
# pdb.set_trace()