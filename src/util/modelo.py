import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ALVO = "Renda Total [R$ milhões]"
COLUNA_ANO = "Ano-calendário"
COLUNAS_FORA = [COLUNA_ANO, "Ente Federativo", ALVO]

# Baseline: regressão linear simples usando só a renda do ano anterior
VARIAVEL_BASELINE = "Renda Total (ano anterior)"

# Divisão temporal: treina com os anos antigos e testa com os mais recentes.
# Divisão aleatória colocaria anos "do futuro" no treino (vazamento de dados).
ANO_CORTE = 2017


def dividir_treino_teste(data: pd.DataFrame, ano_corte: int = ANO_CORTE):
    '''
    Separa o conjunto em treino (anos <= ano_corte) e teste (anos > ano_corte).
    '''
    treino = data[data[COLUNA_ANO] <= ano_corte]
    teste = data[data[COLUNA_ANO] > ano_corte]
    return treino, teste


def calcular_metricas(y_real, y_previsto) -> dict:
    '''
    Retorna R², RMSE e MAE (em R$ milhões).
    '''
    return {
        "R²": r2_score(y_real, y_previsto),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "MAE": mean_absolute_error(y_real, y_previsto),
    }


def treinar_regressao(data: pd.DataFrame, variaveis: list[str], ano_corte: int = ANO_CORTE) -> dict:
    '''
    Treina uma regressão linear com as variáveis informadas e avalia no conjunto de teste.

    data -> DataFrame agregado (dfAgregado)
    variaveis -> lista de colunas usadas como entrada do modelo
    ano_corte -> último ano usado no treino
    '''
    treino, teste = dividir_treino_teste(data, ano_corte)

    modelo = LinearRegression()
    modelo.fit(treino[variaveis], treino[ALVO])

    prev_treino = modelo.predict(treino[variaveis])
    prev_teste = modelo.predict(teste[variaveis])

    previsoes = teste[[COLUNA_ANO, "Ente Federativo", ALVO]].copy()
    previsoes["Previsto [R$ milhões]"] = prev_teste
    previsoes["Erro [R$ milhões]"] = previsoes["Previsto [R$ milhões]"] - previsoes[ALVO]
    previsoes["Erro (%)"] = previsoes["Erro [R$ milhões]"] / previsoes[ALVO] * 100

    coeficientes = (
        pd.DataFrame({"Variável": variaveis, "Coeficiente": modelo.coef_})
        .sort_values("Coeficiente", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    return {
        "modelo": modelo,
        "variaveis": variaveis,
        "intercepto": float(modelo.intercept_),
        "coeficientes": coeficientes,
        "metricas_treino": calcular_metricas(treino[ALVO], prev_treino),
        "metricas_teste": calcular_metricas(teste[ALVO], prev_teste),
        "previsoes": previsoes,
        "n_treino": len(treino),
        "n_teste": len(teste),
    }


def treinar_modelos(data: pd.DataFrame, ano_corte: int = ANO_CORTE) -> dict:
    '''
    Treina a regressão linear múltipla (todas as variáveis do ano anterior)
    e a regressão linear simples (baseline, só a renda do ano anterior).
    '''
    variaveis_multipla = [c for c in data.columns if c not in COLUNAS_FORA]

    return {
        "multipla": treinar_regressao(data, variaveis_multipla, ano_corte),
        "simples": treinar_regressao(data, [VARIAVEL_BASELINE], ano_corte),
    }
