from src.util.regressao.testes_estatistica import dividir_treino_teste, obter_metrics
from src.util.regressao.cenarios import treinar_cenario

import pandas as pd

from sklearn.linear_model import LinearRegression

def treinar_regressao(data: pd.DataFrame, VARIAVEIS: list[str], ANO_CORTE: int = 2017, COLUNA_ANO: str = "Ano-calendário", ALVO: str = "Renda Total [R$ milhões]") -> dict:
    '''
    Treina uma regressão linear com as variáveis informadas e avalia no conjunto de teste.

    data -> DataFrame agregado (dfAgregado)
    variaveis -> lista de colunas usadas como entrada do modelo
    ANO_CORTE -> último ano usado no treino (2017)
    COLUNA_ANO -> coluna relacionado ao ano, "Ano-calendário"
    ALVO -> o que será predito, está definido como "Renda Total [R$ milhões]" por padrão
    '''
    treino, teste = dividir_treino_teste(data, COLUNA_ANO, ANO_CORTE)

    modelo = LinearRegression()
    modelo.fit(treino[VARIAVEIS], treino[ALVO])

    prev_treino = modelo.predict(treino[VARIAVEIS])
    prev_teste = modelo.predict(teste[VARIAVEIS])

    previsoes = teste[[COLUNA_ANO, "Ente Federativo", ALVO]].copy()
    previsoes["Previsto [R$ milhões]"] = prev_teste
    previsoes["Erro [R$ milhões]"] = previsoes["Previsto [R$ milhões]"] - previsoes[ALVO]
    previsoes["Erro (%)"] = previsoes["Erro [R$ milhões]"] / previsoes[ALVO] * 100

    coeficientes = (
        pd.DataFrame({"Variável": VARIAVEIS, "Coeficiente": modelo.coef_})
        .sort_values("Coeficiente", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    return {
        "modelo": modelo,
        "variaveis": VARIAVEIS,
        "intercepto": float(modelo.intercept_),
        "coeficientes": coeficientes,
        "metricas_treino": obter_metrics(treino[ALVO], prev_treino),
        "metricas_teste": obter_metrics(teste[ALVO], prev_teste),
        "previsoes": previsoes,
        "n_treino": len(treino),
        "n_teste": len(teste)
    }


def treinar_modelo_multipla(data: pd.DataFrame, COLUNAS: list[str], ANO_CORTE: int):
    '''
    Treina a regressão linear múltipla (todas as variáveis do ano anterior)
    '''
    variaveis_multipla = [c for c in data.columns if c not in COLUNAS]

    return treinar_regressao(data, variaveis_multipla, ANO_CORTE)

def treinar_linear(data: pd.DataFrame, usar_ano: bool = False, usar_ente: bool = False,
                   so_lag_renda: bool = False, ANO_CORTE: int = 2017) -> dict:
    '''
    LinearRegression em um cenário (mesmas features, mesmo pipeline e mesmo corte do Ridge).
    '''
    return treinar_cenario(data, LinearRegression(), usar_ano, usar_ente, so_lag_renda, ANO_CORTE)