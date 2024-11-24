import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix
import gdown

def carregar_dados(file_path):
    """
    Carrega os dados de um arquivo JSON e retorna um DataFrame do pandas.
    """
    try:
        dados = pd.read_json(file_path, lines=True)
        return dados
    except ValueError as e:
        print(f"Erro ao carregar os dados do arquivo {file_path}: {e}")
        return None
    except FileNotFoundError as e:
        print(f"Arquivo não encontrado: {file_path}")
        return None

def analise_exploratoria(dados):
    """
    Realiza a análise exploratória dos dados, exibindo as primeiras linhas,
    informações do dataset, estatísticas descritivas e valores nulos.
    """
    print("Primeiras linhas do dataset:")
    print(dados.head())
    print("\nInformações do dataset:")
    print(dados.info())
    print("\nEstatísticas descritivas:")
    print(dados.describe())
    print("\nValores nulos:")
    print(dados.isnull().sum())

def limpeza_dados(dados):
    """
    Limpa os dados, removendo duplicatas e preenchendo valores nulos.
    """
    # Remove duplicatas
    dados = dados.drop_duplicates()
    # Preenche valores nulos com a mediana das colunas numéricas
    dados = dados.fillna(dados.median(numeric_only=True))
    # Converte colunas numéricas para inteiros
    colunas_numericas = dados.select_dtypes(include=[np.number]).columns
    dados[colunas_numericas] = dados[colunas_numericas].astype(int)

    return dados

def analise_univariada(dados):
    """
    Realiza a análise univariada das variáveis.
    """
    # Filtra colunas que não possuem valores nulos
    colunas_com_dados = dados.dropna(axis=1, how='all').select_dtypes(include=[np.number]).columns
    # Para cada coluna numérica, cria um histograma interativo
    for coluna in colunas_com_dados:
        fig = px.histogram(dados, x=coluna, nbins=50, title=f'Distribuição da variável {coluna}')
        fig.update_layout(
            xaxis_title=coluna,
            yaxis_title='Frequência',
            bargap=0.2,
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)  # Garante que os valores no eixo x sejam inteiros
        )
        fig.show()

def analise_bivariada(dados):
    """
    Realiza a análise bivariada das variáveis.
    """
    # Filtra colunas que não possuem valores nulos
    colunas_com_dados = dados.dropna(axis=1, how='all').select_dtypes(include=[np.number]).columns
    # Cria gráficos de dispersão interativos para explorar a relação entre pares de variáveis
    fig = px.scatter_matrix(dados[colunas_com_dados], title='Gráficos de Dispersão entre Variáveis')
    fig.update_layout(width=1000, height=1000)
    fig.show()

def analise_multivariada(dados):
    """
    Realiza a análise multivariada das variáveis.
    """
    # Filtra colunas que não possuem valores nulos
    dados_numericos = dados.dropna(axis=1, how='all').select_dtypes(include=[np.number])
    # Calcula a matriz de correlação
    corr = dados_numericos.corr()
    # Cria um mapa de calor interativo para visualizar a matriz de correlação
    fig = px.imshow(corr, text_auto=True, title='Matriz de Correlação entre Variáveis Numéricas')
    fig.update_layout(width=800, height=800)
    fig.show()

def preparar_dados(dados):
    """
    Prepara os dados para treinamento, incluindo normalização e seleção de características.
    """
    # Criar uma coluna 'target' fictícia para fins de demonstração
    if 'target' not in dados.columns:
        dados['target'] = np.random.randint(0, 2, size=len(dados))  # Exemplo de coluna alvo binária

    # Selecionar colunas numéricas
    colunas_numericas = dados.select_dtypes(include=[np.number]).columns.tolist()

    X = dados[colunas_numericas]
    y = dados['target']

    # Verificar se há colunas numéricas para normalizar
    if X.empty:
        raise ValueError("Nenhuma coluna numérica encontrada para normalização.")

    # Normalização dos dados
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Seleção de características
    selector = SelectKBest(chi2, k=min(10, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)

    # Divide os dados em conjuntos de treinamento e teste
    return train_test_split(X_selected, y, test_size=0.3, random_state=42)


def main():
    # URL do arquivo compartilhado no Google Drive
    url = 'https://drive.google.com/uc?id=1-95IZN_WqUEI2ersF_AZQuANGLZNMWj8'

    # Baixar o arquivo usando gdown
    output = 'dados_preparados.json'
    gdown.download(url, output, quiet=False)

    # Carregar os dados preparados
    dados = carregar_dados(output)
    if dados is not None:
        # Realiza a análise exploratória dos dados
        print("Dados carregados com sucesso:")
        print(dados.head())
    else:
        print("Erro ao carregar os dados.")
        return

    # Realiza a análise exploratória dos dados
    analise_exploratoria(dados)

    # Limpeza dos dados
    dados = limpeza_dados(dados)

    # Análise Univariada
    analise_univariada(dados)

    # Análise Bivariada
    analise_bivariada(dados)

    # Análise Multivariada
    analise_multivariada(dados)

    # Preparar os dados
    X_train, X_test, y_train, y_test = preparar_dados(dados)

if __name__ == "__main__":
    main()