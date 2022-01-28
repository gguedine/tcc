# README

 
### Preparação das bibliotecas:

### Passo a serem executados para gerar o arquivo de comparação entre algorítmos:

	 
 1. Na linha 103 do arquivo kmeans.py, escolher o nome do arquivos a ser gerado ex.: "classic_kmeans";
 2. Executar o comando abaixo:
 2.1. `python3 kmeans.py`
 3. Nos arquivos knn.py e recommendation.py alterar o nome dos arquivos para o nome escolhido no passo 1 e alterar a variável "file_name" para os nomes desejados;
 4. Executar os comandos abaixo:
 4.1 `python3 knn.py`
 4.2 `python3 recommendation.py`
 5. No arquivo comparing_data.py alterar as variáveis "recm_file_name" e "knn_file_name" para os nomes escolhidos no passo 3;
 6.  Executar os comandos abaixo:
 6.1 `python3 comparing_data.py`
 

### Passo a serem executados para rodar o word cloud:

	 
 1. Na linha 103 do arquivo kmeans.py, escolher o nome do arquivos a ser gerado ex.: "classic_kmeans";
 2. Executar o comando abaixo:
 2.1. `python3 kmeans.py`
 3. No arquivo recommendation.py alterar o nome do arquivo para o nome escolhido no passo 1 e alterar a variável "file_name" para o nome desejado;
 4. Descomentar a linha `# word_cloud_plot()`
 5. Executar o comando abaixo:
 5.1 `python3 recommendation.py`


\begin{verbatim}
  função build_recommendation(novos_artigos, centroides, artigos_clusterizados, qnt_recomendacoes):
    aplique a lemmatizacão nos novos_artigos;
    calcule a matriz de distância dos novos_artigos em relação aos centroides dos clusters;
    para cada linha da matriz, selecione o cluster que possui a menor distância;
    selecione os artigos dos clusters encontrados no passo anterior e aplique a lemmatização neles;
    para cada novo_artigo:
      salve as distancias do novo_artigo em relação aos artigos_clusterizados pertencentes ao cluster selecionado nos passos anteriores;
      ordene os artigos do cluster da menor distância para a maior;
      se qnt_recomendacoes > quantidade de artigos nesse cluster então:
        qnt_recomendacoes -> quantidade de artigos do cluster;
      fim se;
      salve os qnt_recomendacoes primeiros artigos ordenados como sendo os artigos que serão recomendados;
    fim para;
    retorne a estrutura de resposta contendo cada novo_artigo, seu cluster com menor distancia e os qnt_recomendacoes artigos recomendados;
  fim função;
    
\end{verbatim} 