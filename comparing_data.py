import os
import pandas as pd
import numpy as np
import evaluating as ev
from ast import literal_eval
import pdb


BASE_DIR = os.getcwd()
recm_file_name = os.path.join(BASE_DIR,"dados_salvos","recommendation_data_k11_random_1.pkl")
knn_file_name = os.path.join(BASE_DIR,"dados_salvos","knn_data_k11_random_1.pkl")

load_recommendation = pd.read_pickle(recm_file_name)
recm_new_data = pd.concat(load_recommendation['new_data'].array)
recm_similar_elements = pd.DataFrame(load_recommendation['similar_elements'].array)
recm_similar_elements_id = pd.DataFrame(load_recommendation['similar_elements_submission_id'].array)
recm_clusters = load_recommendation['closest_cluster']

load_knn = pd.read_pickle(knn_file_name)
knn_new_data = pd.concat(load_knn['new_data'].array)
knn_similar_elements_id = pd.DataFrame(load_knn['knn_similar_elements_submission_id'].array)
nn_similar_elements = pd.DataFrame(load_knn['similar_elements'].array)
nn_similar_elements_id = pd.DataFrame(load_knn['similar_elements_submission_id'].array)
knn_clusters = load_knn['closest_cluster']

d = {
    'submission_id': np.atleast_2d(knn_new_data['submission_id'].values).T,
    'recm_sub_ids': recm_similar_elements_id.values,
    'knn_sub_ids': knn_similar_elements_id.values,
    'nn_sub_ids': nn_similar_elements_id.values,
    'knn_clusters': knn_clusters.values
}

arr = np.concatenate((d['submission_id'], d['recm_sub_ids'],d['nn_sub_ids'],d['knn_sub_ids']), axis=1)
recm_size = (arr.shape[1] - 1)/3
dicionario = {
    'submission_id': arr[:,0],
    'recm_clusters': recm_clusters,
    'knn_clusters': knn_clusters,
    'recm_sub_ids': arr[:, 1:int(recm_size+1)].tolist(),
    'nn_sub_ids': arr[:, int(recm_size+1):2*int(recm_size)+1].tolist(),
    'knn_sub_ids': arr[:, 2*int(recm_size)+1:].tolist()
}
pd.options.display.float_format = '{:.2f}'.format
df = pd.DataFrame(dicionario)
#TODOS QUE não estão em nenhum dos dois # VERDADEIRO NEGATIVO
df['same_cluster'] = np.where(df["recm_clusters"] == df["knn_clusters"], True, False)
df['matchs(VP)-REC'] = [list(set(a).intersection(set(b))) for a, b in zip(df.nn_sub_ids, df.recm_sub_ids)] #VP #SÓ USE esse
df['matchs(VP)-KNN'] = [list(set(a).intersection(set(b))) for a, b in zip(df.nn_sub_ids, df.knn_sub_ids)] #VP #SÓ USE esse
df['erros(FP)-REC'] = [list(set(a).difference(set(b))) for a, b in zip(df.nn_sub_ids, df.recm_sub_ids)] #FP
df['erros(FP)-KNN'] = [list(set(a).difference(set(b))) for a, b in zip(df.nn_sub_ids, df.knn_sub_ids)] #FP
df['erros(FN)-REC'] = [list(set(a).difference(set(b))) for a, b in zip(df.recm_sub_ids, df.nn_sub_ids)] #FN # E esse
df['erros(FN)-KNN'] = [list(set(a).difference(set(b))) for a, b in zip(df.knn_sub_ids, df.nn_sub_ids)] #FN # E esse
df['matchs_count(VP)-REC'] = df['matchs(VP)-REC'].apply(lambda x: len(x))
df['matchs_count(VP)-KNN'] = df['matchs(VP)-KNN'].apply(lambda x: len(x))
df['erros_count(FP)-REC'] = df['erros(FP)-REC'].apply(lambda x: len(x))
df['erros_count(FP)-KNN'] = df['erros(FP)-KNN'].apply(lambda x: len(x))
df['erros_count(FN)-REC'] = df['erros(FN)-REC'].apply(lambda x: len(x))
df['erros_count(FN)-KNN'] = df['erros(FN)-KNN'].apply(lambda x: len(x))
df['precision-REC'] = df[['matchs_count(VP)-REC','erros_count(FP)-REC']].apply(
    lambda x: ev.precision(x['matchs_count(VP)-REC'], x['erros_count(FP)-REC'])*100, axis=1
)
df['precision-KNN'] = df[['matchs_count(VP)-KNN','erros_count(FP)-KNN']].apply(
    lambda x: ev.precision(x['matchs_count(VP)-KNN'], x['erros_count(FP)-KNN'])*100, axis=1
)
# df['recall-REC'] = df[['matchs_count(VP)-REC','erros_count(FN)-REC']].apply(
#     lambda x:  ev.recall(x['matchs_count(VP)-REC'], x['erros_count(FN)-REC'])*100, axis=1
# )
# df['f1score-REC'] = df[['precision-REC', 'recall-REC']].apply(
#     lambda x: ev.recall(x['precision-REC'], x['recall-REC'])*100, axis=1
# )
# df['matchs_rate(VP)'] = df['matchs_count(VP)-REC'].apply(lambda x: x*100/50)
# df['matchs_rate(VP)'] = df['matchs_count(VP)-REC'].apply(lambda x: x*100/50)
# df['errors_rate'] = df.erros_count.apply(lambda x: x*100/50)
file_name='compare_result_k_11_random_1'
needed_columns = [
    'submission_id',
    'recm_clusters',
    'knn_clusters',
    'same_cluster',
    'matchs_count(VP)-REC',
    'matchs_count(VP)-KNN',
    'erros_count(FP)-REC',
    'erros_count(FP)-KNN',
    'erros_count(FN)-REC',
    'erros_count(FN)-KNN',
    'recm_sub_ids',
    'nn_sub_ids',
    'knn_sub_ids',
    'matchs(VP)-REC',
    'matchs(VP)-KNN',
    'erros(FP)-REC',
    'erros(FP)-KNN',
    'erros(FN)-REC',
    'erros(FN)-KNN',
    'precision-REC',
    'precision-KNN',
    # 'recall-REC',
    # 'f1score-REC'
]
result_array = np.array(df[needed_columns])
np.savetxt(os.path.join(BASE_DIR, "dados_salvos",file_name +"_.csv"), result_array, delimiter=";",
               fmt='% s', comments='',
               header=";".join(needed_columns))

datf = pd.read_csv( os.path.join(BASE_DIR, "dados_salvos", file_name+"_.csv"),
                    converters={'matchs': literal_eval, 'erros': literal_eval},
                    sep=';')

pd.set_option("min_rows", 50)
# print(datf.iloc[:,[0,1,3,5,7,9,13]] )
print(datf.iloc[:,datf.columns.get_indexer(['submission_id', #'recm_clusters', 'same_cluster',
                                            'matchs_count(VP)-REC', 'erros_count(FP)-REC', 'precision-REC'])] )
print("\n")
print(datf.iloc[:,datf.columns.get_indexer(['submission_id', #'recm_clusters', 'same_cluster',
                                            'matchs_count(VP)-KNN', 'erros_count(FP)-KNN',  'precision-KNN'])] )
print("\n Quantidade de Classificações iguais entre o algoritmo desenvolvido e o KNN: {}".format(datf['same_cluster'].value_counts()[True]))
print("\n Quantidade de Classificações diferentes entre o algoritmo desenvolvido e o KNN: {}".format(datf['same_cluster'].value_counts()[False]))
print("\n Precisão de Classificações : {}".format(ev.precision(datf['same_cluster'].value_counts()[True], datf['same_cluster'].value_counts()[False])*100))
dicio = {"count": "Quantidade", "mean":"Media", "std": "Desvio Padrão", "min": "Mínimo",
         "max": "Máximo", "25%": "25\%(percentil)", "50%":"50\%(percentil)", "75%":"75\%(percentil)"}
# print("Valores dos Acertos : {}".format(datf['matchs_count(VP)-REC'].describe().apply("{0:.2f}".format)))
print("\nValores dos Acertos REC :")
acertos = datf['matchs_count(VP)-REC'].describe().to_dict()
acertos = dict((dicio[key], value) for (key, value) in acertos.items())
for k, v in acertos.items():
    print(k+" & " +"{:.2f}".format(v)+"\\"+"\\")
# print("Valores da Precisao: {}".format(datf['precision-REC'].describe().apply("{0:.2f}".format)))

print("\nValores da Precisao REC: ")
acertos = datf['precision-REC'].describe().to_dict()
acertos = dict((dicio[key], value) for (key, value) in acertos.items())
for k, v in acertos.items():
    print(k+" & " +"{:.2f}".format(v)+"\\"+"\\")

print("\nValores dos Acertos KNN :")
acertos = datf['matchs_count(VP)-KNN'].describe().to_dict()
acertos = dict((dicio[key], value) for (key, value) in acertos.items())
for k, v in acertos.items():
    print(k+" & " +"{:.2f}".format(v)+"\\"+"\\")
# print("Valores da Precisao: {}".format(datf['precision-REC'].describe().apply("{0:.2f}".format)))

print("\nValores da Precisao KNN: ")
acertos = datf['precision-KNN'].describe().to_dict()
acertos = dict((dicio[key], value) for (key, value) in acertos.items())
for k, v in acertos.items():
    print(k+" & " +"{:.2f}".format(v)+"\\"+"\\")
# print("\nrecall-REC mean: {}".format(datf['recall-REC'].mean()))
# print("\nf1score-REC mean: {}".format(datf['f1score-REC'].mean()))
# pdb.set_trace()

# mean & 23.88 \\
# std & 9.91 \\
# min & 4.00 \\
# 25% & 16.00 \\
# 50% & 22.00 \\
# 75% & 31.00 \\
# max & 47.00 \\
# Name: matchs_count(VP)-REC, dtype: object
# Valores da Precisao: count & 540.00
# mean & 47.75 \\
# std & 19.82 \\
# min & 8.00 \\
# 25% & 32.00 \\
# 50% & 44.00 \\
# 75% & 62.00 \\
# max & 94.00 \\
['submission_id', 'recm_clusters', 'same_cluster','matchs_count(VP)-REC', 'erros_count(FP)-REC', 'erros_count(FN)-REC', 'precision-REC']