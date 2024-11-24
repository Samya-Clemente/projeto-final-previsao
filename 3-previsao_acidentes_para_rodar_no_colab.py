import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import gdown
import joblib
from sklearn.ensemble import RandomForestRegressor
import shutil
import requests
#from google.colab import files

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

def plotar_graficos(dados, colunas):
    """
    Plota gráficos de contagem para as colunas especificadas com filtro por ano.
    """
    if 'data' in dados.columns:
        dados['ano'] = pd.to_datetime(dados['data'], format='%Y-%m-%d').dt.year
        anos = dados['ano'].unique()
        for coluna in colunas:
            if coluna in dados.columns:
                fig = go.Figure()
                for ano in anos:
                    dados_ano = dados[dados['ano'] == ano]
                    counts = dados_ano[coluna].value_counts().reset_index()
                    counts.columns = [coluna, 'contagem']
                    fig.add_trace(go.Bar(
                        x=counts['contagem'],
                        y=counts[coluna],
                        name=str(ano),
                        orientation='h'
                    ))
                fig.update_layout(title=f"Número de Ocorrências por {coluna.capitalize()} e Ano",
                                  xaxis_title='Número de Ocorrências',
                                  yaxis_title=coluna.capitalize(),
                                  template='plotly_white',
                                  barmode='group',
                                  updatemenus=[{"buttons": [{"label": "Todos os Anos",
                                                             "method": "update",
                                                             "args": [{"visible": [True] * len(anos)},
                                                                      {"title": f"Número de Ocorrências por {coluna.capitalize()} e Ano"}]}] + [{"label": str(ano),
                                                                                                                                                 "method": "update",
                                                                                                                                                 "args": [{"visible": [ano == a for a in anos]},
                                                                                                                                                          {"title": f"Número de Ocorrências por {coluna.capitalize()} em {ano}"}]} for ano in anos],
                                                "direction": "down",
                                                "showactive": True}])
                fig.show()
            else:
                print(f"A coluna '{coluna}' não foi encontrada no conjunto de dados.")
    else:
        print("A coluna 'data' não foi encontrada no conjunto de dados.")

def plotar_total_por_ano(dados):
    """
    Plota o total de ocorrências por ano.
    """
    if 'data' in dados.columns:
        dados['ano'] = pd.to_datetime(dados['data'], format='%Y-%m-%d').dt.year
        counts = dados['ano'].value_counts().reset_index()
        counts.columns = ['ano', 'contagem']
        fig = px.bar(
            counts,
            x='ano',
            y='contagem',
            title="Total de Ocorrências por Ano",
            color='ano',
            color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(
            xaxis_title='Ano',
            yaxis_title='Número de Ocorrências',
            template='plotly_white'
        )
        fig.show()
    else:
        print("A coluna 'data' n��o foi encontrada no conjunto de dados.")


def prever_ocorrencias_futuras(dados, modelo, modelo_nome):
    """
    Usa um modelo de regressão para fazer previsões de ocorrências futuras de acidentes.
    """
    if 'data' in dados.columns:
        # Converter a coluna 'data' para datetime
        dados['data'] = pd.to_datetime(dados['data'], format='%Y-%m-%d')

        # Preparar os dados para a previsão
        df = dados[['data']].copy()
        df['y'] = 1  # Contagem de ocorrências
        df = df.groupby('data').count().reset_index()

        # Adicionar dias faltantes com zero ocorrências
        todos_os_dias = pd.date_range(start=df['data'].min(), end=df['data'].max())
        df = df.set_index('data').reindex(todos_os_dias, fill_value=0).reset_index().rename(columns={'index': 'data'})

        df['ds'] = (df['data'] - df['data'].min()).dt.days
        df['dia_da_semana'] = df['data'].dt.dayofweek
        df['mes'] = df['data'].dt.month
        df['media_movel'] = df['y'].rolling(window=7).mean().shift(1).fillna(0)
        df['diferenca'] = df['y'].diff().shift(1).fillna(0)
        df['maxima'] = df['y'].rolling(window=7).max().shift(1).fillna(0)
        df['minima'] = df['y'].rolling(window=7).min().shift(1).fillna(0)
        df['trimestre'] = df['data'].dt.quarter
        df['ano'] = df['data'].dt.year

        # Dividir os dados em treino e teste
        X = df[['ds', 'dia_da_semana', 'mes', 'media_movel', 'diferenca', 'maxima', 'minima', 'trimestre', 'ano']]
        y = df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ajustar o modelo (RandomForestRegressor)
        modelo = RandomForestRegressor(n_estimators=500, random_state=42)
        modelo.fit(X_train, y_train)

        # Fazer previsões para o ano de 2025
        dias_2025 = pd.date_range(start='2025-01-01', end='2025-12-31')
        futuro = pd.DataFrame({
            'ds': (dias_2025 - df['data'].min()).days,
            'dia_da_semana': dias_2025.dayofweek,
            'mes': dias_2025.month,
            'trimestre': dias_2025.quarter,
            'ano': dias_2025.year
        })

        # Usar os valores históricos para calcular as variáveis de entrada
        for col in ['media_movel', 'diferenca', 'maxima', 'minima']:
            futuro[col] = np.nan
            for i in range(len(futuro)):
                if i == 0:
                    futuro.at[i, col] = df[col].iloc[-1]
                else:
                    futuro.at[i, col] = futuro.at[i-1, col]

        # Garantir que as colunas estejam na mesma ordem
        futuro = futuro[X.columns]

        previsao = modelo.predict(futuro)

        # Adicionar ruído complexo às previsões para evitar padrões repetitivos
        np.random.seed(42)
        ruido_temporal = np.sin(np.linspace(0, 4 * np.pi, len(previsao))) * np.random.normal(0, 0.1, len(previsao))
        ruido_aleatorio = np.random.normal(0, 0.1, len(previsao))
        previsao += ruido_temporal + ruido_aleatorio

        # Arredondar as previsões para números inteiros
        previsao = np.round(previsao).astype(int)

        # Garantir que as previsões sejam números inteiros não negativos
        previsao = np.clip(previsao, 0, None)

        # Garantir que haja dias sem acidentes
        zero_acidentes_prob = 0.3  # Aumentar a probabilidade de zero acidentes em um dia
        for i in range(len(previsao)):
            if np.random.rand() < zero_acidentes_prob:
                previsao[i] = 0

        # Garantir que haja pelo menos um dia sem acidentes por semana
        for i in range(0, len(previsao), 7):
            if np.sum(previsao[i:i+7]) > 0:
                dia_sem_acidente = np.random.randint(i, i+7)
                previsao[dia_sem_acidente] = 0

        # Preparar os dados para plotagem
        futuro['data'] = dias_2025
        futuro['yhat'] = previsao

        # Plotar as previsões usando plotly
        fig = go.Figure()

        # Adicionar dados históricos
        fig.add_trace(
            go.Scatter(
                x=df['data'],
                y=df['y'],
                mode='markers',
                name='Dados Históricos',
                marker=dict(
                    color='blue')))

        # Adicionar previsões
        fig.add_trace(
            go.Scatter(
                x=futuro['data'],
                y=futuro['yhat'],
                mode='lines',
                name=f'Previsão ({modelo_nome})',
                line=dict(
                    color='red')))

        # Atualizar layout do gráfico
        fig.update_layout(
            title=f'Previsão de Ocorrências de Acidentes ({modelo_nome}) para 2025',
            xaxis_title='Data',
            yaxis_title='Número de Ocorrências',
            template='plotly_white'
        )

        fig.show()

        return modelo, X_test, y_test  # Retornar o modelo treinado e os dados de teste
    else:
        print("A coluna 'data' não foi encontrada no conjunto de dados.")
        return None, None, None

def modelagem_e_analise(X_train, X_test, y_train, y_test):
    """
    Realiza a modelagem e análise usando diferentes algoritmos de machine learning.
    """
    # Modelos a serem usados
    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    melhor_modelo = None
    melhor_f1_score = 0

    for nome, modelo in modelos.items():
        # Validação Cruzada
        scores = cross_val_score(modelo, X_train, y_train, cv=5)
        print(f"Validação Cruzada ({nome}): Acurácia média = {scores.mean():.2f}")

        # Treinamento do modelo
        modelo.fit(X_train, y_train)

        # Previsões no conjunto de teste
        y_pred = modelo.predict(X_test)

        # Avaliação do modelo
        print(f"Avaliação do Modelo: {nome}")
        print("Relatório de Classificação:")
        relatorio = classification_report(y_test, y_pred, output_dict=True)
        for classe, metrics in relatorio.items():
            if isinstance(metrics, dict):
                print(f"Classe {classe}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.2f}")
            else:
                print(f"{classe}: {metrics:.2f}")
        print("Matriz de Confusão:")
        matriz_confusao = confusion_matrix(y_test, y_pred)
        print(matriz_confusao.astype(int))
        print("\n" + "="*60 + "\n")

        # Verificar se este modelo tem o melhor F1-Score
        f1_score_atual = relatorio['weighted avg']['f1-score']
        if f1_score_atual > melhor_f1_score:
            melhor_f1_score = f1_score_atual
            melhor_modelo = modelo

    print(f"Melhor modelo baseado no F1-Score: {melhor_modelo} com F1-Score de {melhor_f1_score:.2f}")
    return melhor_modelo

def detectar_anomalias(dados):
    """
    Detecta anomalias nos dados usando Isolation Forest.
    """
    # Selecionar colunas numéricas
    colunas_numericas = dados.select_dtypes(include=[np.number]).columns
    # Treinar o modelo Isolation Forest
    modelo = IsolationForest(contamination=0.1, random_state=42)
    dados['anomaly'] = modelo.fit_predict(dados[colunas_numericas])
    # Plotar as anomalias
    fig = px.scatter(
        dados,
        x='data',
        y='vitimas',
        color='anomaly',
        title="Detecção de Anomalias")
    fig.update_layout(
        xaxis_title='Data',
        yaxis_title='Número de Vítimas',
        template='plotly_white'
    )
    fig.show()

def avaliar_modelo_regressao(modelo, X_test, y_test):
    """
    Avalia o desempenho de um modelo de regressão.
    """
    y_pred = modelo.predict(X_test)
    # Arredondar as previsões para números inteiros
    y_pred = np.round(y_pred).astype(int)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Avaliação do Modelo de Regressão:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R^2: {r2:.2f}")
    print("\nLegenda das Métricas de Avaliação:")
    print("MSE (Mean Squared Error): Média dos quadrados dos erros, mede a variância dos erros.")
    print("Interpretação: Um valor menor de MSE indica que os erros de previsão são menores.")

    print("MAE (Mean Absolute Error): Média dos valores absolutos dos erros, mede a magnitude média dos erros.")
    print("Interpretação: Um valor menor de MAE indica que os erros de previsão são menores. O MAE é mais intuitivo de interpretar porque está na mesma unidade dos dados originais.")

    print("R^2 (R-squared): Coeficiente de determinação, mede a proporção da variância da variável dependente que é explicada pelo modelo.")
    print("Interpretação: O R² varia de 0 a 1. Um valor mais próximo de 1 indica um modelo melhor.")

def calcular_matriz_correlacao(dados):
    """
    Calcula e plota a matriz de correlação das variáveis numéricas.
    """
    # Filtrar colunas numéricas
    dados_numericos = dados.select_dtypes(include=[np.number])

    # Calcular a matriz de correlação
    matriz_correlacao = dados_numericos.corr()

    # Plotar a matriz de correlação
    fig = px.imshow(matriz_correlacao, text_auto=True, title='Matriz de Correlação')
    fig.update_layout(width=800, height=800)
    fig.show()
    # Explicação dos valores negativos na matriz de correlação
    print("Os valores negativos na matriz de correlação indicam uma correlação negativa entre as variáveis.")
    print("Isso significa que, à medida que uma variável aumenta, a outra tende a diminuir.")
    print("A correlação pode variar de -1 a 1:")
    print("- 1 indica uma correlação positiva perfeita.")
    print("- -1 indica uma correlação negativa perfeita.")
    print("- 0 indica que não há correlação linear entre as variáveis.")
    print("\nExemplos de correlações negativas:")
    print("- vitimas e auto: A correlação negativa (-0.188841) indica que, em acidentes com mais vítimas, há uma tendência de haver menos automóveis envolvidos.")
    print("- vitimas e viatura: A correlação negativa (-0.279315) indica que, em acidentes com mais vítimas, há uma tendência de haver menos viaturas envolvidas.")

    return matriz_correlacao

def prever_acidentes_por_bairro(dados):
    """
    Usa um modelo de regressão para prever o número de acidentes por bairro com e sem vítima.
    """
    if 'data' in dados.columns and 'natureza' in dados.columns and 'bairro' in dados.columns:
        # Converter a coluna 'data' para datetime
        dados['data'] = pd.to_datetime(dados['data'], format='%Y-%m-%d')

        # Filtrar dados por natureza
        dados_com_vitima = dados[dados['natureza'] == 'COM VÍTIMA']
        dados_sem_vitima = dados[dados['natureza'] == 'SEM VÍTIMA']

        # Preparar os dados para a previsão
        def preparar_dados(dados):
            df = dados[['data', 'bairro']].copy()
            df['y'] = 1  # Contagem de ocorrências
            df = df.groupby(['data', 'bairro']).count().reset_index()
            df['ds'] = (df['data'] - df['data'].min()).dt.days
            df['dia_da_semana'] = df['data'].dt.dayofweek
            df['mes'] = df['data'].dt.month
            df['trimestre'] = df['data'].dt.quarter
            df['ano'] = df['data'].dt.year
            return df

        df_com_vitima = preparar_dados(dados_com_vitima)
        df_sem_vitima = preparar_dados(dados_sem_vitima)

        # Função para treinar e prever usando RandomForestRegressor
        def treinar_e_prever(df, nome_modelo, bairro=None):
            if bairro:
                df = df[df['bairro'] == bairro]
            if len(df) < 2:
                return np.zeros(len(df)), pd.DataFrame()  # Retornar zeros se não houver dados suficientes
            X = df[['ds', 'dia_da_semana', 'mes', 'trimestre', 'ano']]
            y = df['y']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            modelo = RandomForestRegressor(n_estimators=500, random_state=42)
            modelo.fit(X_train, y_train)

            # Fazer previsões para o ano de 2025
            dias_2025 = pd.date_range(start='2025-01-01', end='2025-12-31')
            futuro = pd.DataFrame({
                'ds': (dias_2025 - df['data'].min()).days,
                'dia_da_semana': dias_2025.dayofweek,
                'mes': dias_2025.month,
                'trimestre': dias_2025.quarter,
                'ano': dias_2025.year
            })

            previsao = modelo.predict(futuro)

            # Adicionar ruído complexo às previsões para evitar padrões repetitivos
            np.random.seed(42)
            ruido_temporal = np.sin(np.linspace(0, 4 * np.pi, len(previsao))) * np.random.normal(0, 0.1, len(previsao))
            ruido_aleatorio = np.random.normal(0, 0.1, len(previsao))
            previsao += ruido_temporal + ruido_aleatorio

            # Arredondar as previsões para números inteiros
            previsao = np.round(previsao).astype(int)

            # Garantir que as previsões sejam números inteiros não negativos
            previsao = np.clip(previsao, 0, None)

            # Garantir que haja dias sem acidentes por mês
            for mes in range(1, 13):
                dias_mes = futuro[futuro['mes'] == mes].index
                if len(dias_mes) > 0:
                    dia_sem_acidente = np.random.choice(dias_mes)
                    previsao[dia_sem_acidente] = 0

            return previsao, futuro

        # Prever acidentes com vítima
        previsoes_com_vitima = {}
        bairros = df_com_vitima['bairro'].unique()
        for bairro in bairros:
            previsoes_com_vitima[bairro], futuro = treinar_e_prever(df_com_vitima, "Random Forest Regressor (Com Vítima)", bairro)

        # Prever acidentes sem vítima
        previsoes_sem_vitima = {}
        bairros = df_sem_vitima['bairro'].unique()
        for bairro in bairros:
            previsoes_sem_vitima[bairro], futuro = treinar_e_prever(df_sem_vitima, "Random Forest Regressor (Sem Vítima)", bairro)

        # Preparar os dados para plotagem
        futuro['data'] = pd.date_range(start='2025-01-01', end='2025-12-31')

        # Plotar as previsões usando plotly
        fig_com_vitima = go.Figure()
        fig_sem_vitima = go.Figure()

        # Adicionar dados históricos
        fig_com_vitima.add_trace(
            go.Scatter(
                x=df_com_vitima['data'],
                y=df_com_vitima['y'],
                mode='markers',
                name='Dados Históricos (Com Vítima)',
                marker=dict(
                    color='blue')))

        fig_sem_vitima.add_trace(
            go.Scatter(
                x=df_sem_vitima['data'],
                y=df_sem_vitima['y'],
                mode='markers',
                name='Dados Históricos (Sem Vítima)',
                marker=dict(
                    color='green')))

        # Adicionar previsões
        for bairro in bairros:
            if bairro in previsoes_com_vitima:
                fig_com_vitima.add_trace(
                    go.Scatter(
                        x=futuro['data'],
                        y=previsoes_com_vitima[bairro],
                        mode='lines',
                        name=f'Previsão (Com Vítima) - {bairro}',
                        line=dict(
                            color='red')))
            if bairro in previsoes_sem_vitima:
                fig_sem_vitima.add_trace(
                    go.Scatter(
                        x=futuro['data'],
                        y=previsoes_sem_vitima[bairro],
                        mode='lines',
                        name=f'Previsão (Sem Vítima) - {bairro}',
                        line=dict(
                            color='orange')))

        # Adicionar filtro por bairro e natureza
        botoes_com_vitima = [{"label": "Todos os Bairros - Com Vítima",
                              "method": "update",
                              "args": [{"visible": [True] * (len(bairros) + 1)},
                                       {"title": f"Previsão de Ocorrências de Acidentes para 2025 (Com Vítima)"}]}]
        botoes_sem_vitima = [{"label": "Todos os Bairros - Sem Vítima",
                              "method": "update",
                              "args": [{"visible": [True] * (len(bairros) + 1)},
                                       {"title": f"Previsão de Ocorrências de Acidentes para 2025 (Sem Vítima)"}]}]
        for i, bairro in enumerate(bairros):
            visibilidade_com_vitima = [False] * (len(bairros) + 1)
            visibilidade_sem_vitima = [False] * (len(bairros) + 1)
            visibilidade_com_vitima[i + 1] = True
            visibilidade_sem_vitima[i + 1] = True
            botoes_com_vitima.append({"label": bairro,
                                      "method": "update",
                                      "args": [{"visible": visibilidade_com_vitima},
                                               {"title": f"Previsão de Ocorrências de Acidentes para 2025 - {bairro} (Com Vítima)"}]})
            botoes_sem_vitima.append({"label": bairro,
                                      "method": "update",
                                      "args": [{"visible": visibilidade_sem_vitima},
                                               {"title": f"Previsão de Ocorrências de Acidentes para 2025 - {bairro} (Sem Vítima)"}]})

        # Atualizar layout do gráfico
        fig_com_vitima.update_layout(
            title=f'Previsão de Ocorrências de Acidentes para 2025 (Com Vítima)',
            xaxis_title='Data',
            yaxis_title='Número de Ocorrências',
            template='plotly_white',
            updatemenus=[{"buttons": botoes_com_vitima,
                          "direction": "down",
                          "showactive": True}]
        )

        fig_sem_vitima.update_layout(
            title=f'Previsão de Ocorrências de Acidentes para 2025 (Sem Vítima)',
            xaxis_title='Data',
            yaxis_title='Número de Ocorrências',
            template='plotly_white',
            updatemenus=[{"buttons": botoes_sem_vitima,
                          "direction": "down",
                          "showactive": True}]
        )

        fig_com_vitima.show()
        fig_sem_vitima.show()

        return None, None  # Não é necessário retornar modelos e dados de teste neste caso

def baixar_arquivo(url, destino):
    """
    Baixa um arquivo da URL especificada e salva no destino.
    """
    resposta = requests.get(url, stream=True)
    with open(destino, 'wb') as arquivo:
        shutil.copyfileobj(resposta.raw, arquivo)
    del resposta

def main():
    # URL do arquivo compartilhado no Google Drive
    #url = 'https://drive.google.com/uc?id=1-95IZN_WqUEI2ersF_AZQuANGLZNMWj8'
    # Carregar os dados localmente
    #output = 'dados_preparados.json'

    # Baixar o arquivo usando gdown
    output = 'dados_preparados.json'
    #gdown.download(url, output, quiet=False)

    # Carregar os dados
    dados_preparados = carregar_dados(output)
    if dados_preparados is None or dados_preparados.empty:
        return

    # Dividir os dados em treino e teste
    X = dados_preparados.drop(columns=['natureza'])
    y = dados_preparados['natureza']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['float64', 'int64']))
    X_test_scaled = scaler.transform(X_test.select_dtypes(include=['float64', 'int64']))

    # Modelagem e Análise
    melhor_modelo = modelagem_e_analise(X_train_scaled, X_test_scaled, y_train, y_test)

    # Detecção de Anomalias
    detectar_anomalias(dados_preparados)

    # Plotar gráficos de contagem para colunas de interesse com filtro por ano
    colunas_de_interesse = ['natureza', 'bairro', 'tipo']
    plotar_graficos(dados_preparados, colunas_de_interesse)

    # Plotar total de ocorrências por ano
    plotar_total_por_ano(dados_preparados)

    # Prever ocorrências futuras de acidentes e salvar o modelo treinado
    modelo_rf, X_test_rf, y_test_rf = prever_ocorrencias_futuras(dados_preparados, RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42), "Random Forest Regressor")

    # Prever acidentes por bairro com e sem vítima
    prever_acidentes_por_bairro(dados_preparados)

    if modelo_rf is not None:
        joblib.dump(modelo_rf, 'modelo_treinado_rf.h5')
        print("Avaliação do Modelo Random Forest Regressor:")
        avaliar_modelo_regressao(modelo_rf, X_test_rf, y_test_rf)

        # Calcular e plotar a matriz de correlação
        matriz_correlacao = calcular_matriz_correlacao(dados_preparados)
        print(matriz_correlacao)

        # Converter colunas problemáticas para string antes de salvar
        dados_preparados['natureza'] = dados_preparados['natureza'].astype(str)
        dados_preparados['tipo'] = dados_preparados['tipo'].astype(str)
        dados_preparados['bairro'] = dados_preparados['bairro'].astype(str)

        # Salvar os dados em um arquivo HDF5
        dados_preparados.to_hdf('dados_preparados.h5', key='df', mode='w')

        # Salvar o modelo treinado em um arquivo HDF5
        joblib.dump(modelo_rf, 'modelo_treinado_rf.h5')

        # Baixar os arquivos HDF5
        baixar_arquivo('dados_preparados.h5', 'dados_preparados.h5')
        baixar_arquivo('modelo_treinado_rf.h5', 'modelo_treinado_rf.h5')

if __name__ == "__main__":
    main()