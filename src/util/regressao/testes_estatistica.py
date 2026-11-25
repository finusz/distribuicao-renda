from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

import pandas as pd
import numpy as np

ALVO = "Renda Total [R$ milhões]"
LAG_RENDA = "Renda Total (ano anterior)"

def dividir_treino_teste(data: pd.DataFrame, COLUNA_ANO, ANO_CORTE: int):
    '''
    Separa o conjunto em treino (anos <= ano_corte) e teste (anos > ano_corte).
    '''
    treino = data[data[COLUNA_ANO] <= ANO_CORTE]
    teste = data[data[COLUNA_ANO] > ANO_CORTE]
    return treino, teste

def obter_metrics(y_real, y_previsto):
    '''
    Retorna R², RMSE e MAE (em R$ milhões).
    '''
    return {
        "R²": r2_score(y_real, y_previsto),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "MAE": mean_absolute_error(y_real, y_previsto),
    }

def baseline_ano_anterior(data: pd.DataFrame, ANO_CORTE: int = 2017,
                          COLUNA_ANO: str = "Ano-calendário") -> dict:
    '''
    Previsão ingênua: renda deste ano = renda do ano anterior.
    É a "régua mínima": um modelo só vale a pena se ganhar dela.
    '''
    _, teste = dividir_treino_teste(data, COLUNA_ANO, ANO_CORTE)
    metricas = obter_metrics(teste[ALVO], teste[LAG_RENDA])
    metricas["MAPE (%)"] = mean_absolute_percentage_error(teste[ALVO], teste[LAG_RENDA]) * 100
    return metricas

def resumo_modelos(data: pd.DataFrame, resultados: dict, ANO_CORTE: int = 2017) -> pd.DataFrame:
    '''
    Monta uma tabela comparando vários modelos no MESMO conjunto de teste.

    data -> dfAgregado (usado para calcular o baseline)
    resultados -> dicionário {"nome do modelo": dicionário devolvido por treinar_regressao/treinar_ridge}
    '''
    linhas = {"Baseline (ano anterior)": baseline_ano_anterior(data, ANO_CORTE)}

    for nome, res in resultados.items():
        linha = dict(res["metricas_teste"])
        linha["MAPE (%)"] = res["previsoes"]["Erro (%)"].abs().mean()
        linhas[nome] = linha

    return pd.DataFrame(linhas).T.sort_values("MAPE (%)")