import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import gdown

def carregar_dados(file_names):
    dados_list = []
    for file_name in file_names:
        try:
            dados = pd.read_json(file_name)
            dados_list.append(dados)
        except Exception as e:
            print(f"Erro ao carregar os dados do arquivo {file_name}: {e}")
    if dados_list:
        return pd.concat(dados_list, ignore_index=True)
    else:
        return None

def preencher_valores_nulos(dados):
    for coluna in dados.columns:
        if dados[coluna].dtype in ['int64', 'float64']:
            mediana = dados[coluna].median()
            dados[coluna] = dados[coluna].fillna(mediana)
        else:
            moda = dados[coluna].mode()[0]
            dados[coluna] = dados[coluna].fillna(moda)
    return dados

def criar_features(dados):
    if 'data' in dados.columns:
        try:
            dados['data'] = pd.to_datetime(dados['data'], unit='ms')
        except ValueError:
            dados['data'] = pd.to_datetime(dados['data'], format='%Y-%m-%d')
    if 'hora' in dados.columns:
        dados['hora'] = dados['hora'].replace('', np.nan)
        dados = dados.dropna(subset=['hora'])
        try:
            dados['hora'] = pd.to_datetime(dados['hora'], format='%H:%M:%S').dt.time
            dados['hora_do_dia'] = pd.to_datetime(dados['hora'], format='%H:%M:%S').dt.hour
        except ValueError:
            print("Erro ao converter a coluna 'hora'. Verifique se os valores estão no formato '%H:%M:%S'.")
    if 'data' in dados.columns:
        dados['dia_da_semana'] = dados['data'].dt.dayofweek
    return dados

def converter_colunas_numericas(dados):
    colunas_numericas = ['auto', 'moto', 'ciclom', 'ciclista', 'pedestre', 'onibus', 'caminhao', 'viatura', 'vitimas', 'vitimasfatais']
    for coluna in colunas_numericas:
        if dados[coluna].dtype == 'object':
            dados[coluna] = dados[coluna].str.replace(',', '.').astype(float)
    return dados

def preparar_dados(file_names):
    dados = carregar_dados(file_names)
    if dados is not None:
        dados = preencher_valores_nulos(dados)
        dados = criar_features(dados)
        dados = converter_colunas_numericas(dados)

        colunas_desejadas = ['data', 'natureza', 'tipo', 'bairro', 'vitimas', 'vitimasfatais',
                             'auto', 'moto', 'ciclom', 'ciclista', 'pedestre', 'onibus', 'caminhao', 'viatura']
        dados_filtrados = dados[colunas_desejadas]

        dados_filtrados['data'] = dados_filtrados['data'].dt.strftime('%Y-%m-%d')

        dados_filtrados.to_json("dados_preparados.json", orient='records', lines=True, force_ascii=False)
        print("Dados preparados salvos em 'dados_preparados.json'.")

        scaler = StandardScaler()
        colunas_numericas = dados_filtrados.select_dtypes(include=['int64', 'float64']).columns
        dados_filtrados.loc[:, colunas_numericas] = scaler.fit_transform(dados_filtrados[colunas_numericas])

        dados_filtrados.to_json("dados_preparados_final.json", orient='records', lines=True, force_ascii=False)
        # print("Dados finais salvos em 'dados_preparados_final.json'.")

        print(dados_filtrados.head(5))  # Exibir as primeiras 5 linhas dos dados preparados
    else:
        print("Nenhum dado foi carregado.")

def baixar_arquivos(urls):
    file_names = []
    for i, url in enumerate(urls):
        output = f'arquivo_{i}.json'
        gdown.download(url, output, quiet=False)
        file_names.append(output)
    return file_names

if __name__ == "__main__":
    urls = [
        'https://drive.google.com/uc?id=1Wd1LNivQt9s4oixRfaZi3o3Ub7PcjdRJ',  # URL do arquivo 2022.json
        'https://drive.google.com/uc?id=1Wgeqa53QjlgP7j1njdwQDweKWFK_Oual',  # URL do arquivo 2023.json
        'https://drive.google.com/uc?id=1Wn9hEiC4qVlWJrVHgagq9BrOdwzRq6AD'   # URL do arquivo 2024.json
    ]
    file_names = baixar_arquivos(urls)
    preparar_dados(file_names)
    dados = carregar_dados(file_names)
    if dados is not None:
        print(dados.head(5))

    # Exibir as primeiras 5 linhas do arquivo dados_preparados.json
    with open("dados_preparados.json", 'r') as f:
        for _ in range(5):
            print(f.readline().strip())