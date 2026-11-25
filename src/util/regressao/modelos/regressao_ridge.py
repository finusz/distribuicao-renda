from src.util.regressao.cenarios import (
    ALVO, COL_ANO, criar_pipeline, montar_features, treinar_cenario,
)
from src.util.regressao.testes_estatistica import dividir_treino_teste

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]


def janelas_expansivas(treino: pd.DataFrame, n_janelas: int = 4, anos_validacao: int = 1):
    """Gera pares (treino, validação) por ano, sempre treinando no passado e validando no ano seguinte."""
    anos = sorted(treino[COL_ANO].unique())
    for i in range(n_janelas, 0, -1):
        corte = len(anos) - i * anos_validacao
        anos_tr = anos[:corte]
        anos_val = anos[corte:corte + anos_validacao]
        yield treino[treino[COL_ANO].isin(anos_tr)], treino[treino[COL_ANO].isin(anos_val)]


def escolher_alpha(treino: pd.DataFrame, numericas: list[str], categoricas: list[str],
                   alphas: list[float] = ALPHAS) -> float:
    """Testa cada alpha nas janelas expansivas (só com dados de treino) e devolve o de menor MAPE médio."""
    colunas = numericas + categoricas
    media_erro = {}

    for alpha in alphas:
        erros = []
        for tr, val in janelas_expansivas(treino):
            modelo = criar_pipeline(Ridge(alpha=alpha), numericas, categoricas)
            modelo.fit(tr[colunas], tr[ALVO])
            erros.append(mean_absolute_percentage_error(val[ALVO], modelo.predict(val[colunas])))
        media_erro[alpha] = sum(erros) / len(erros)

    return min(media_erro, key=media_erro.get)


def treinar_ridge(data: pd.DataFrame, usar_ano: bool = False, usar_ente: bool = False,
                  so_lag_renda: bool = False, ANO_CORTE: int = 2017, alpha: float | None = None) -> dict:
    '''
    Ridge em um cenário. Se alpha for None, ele é escolhido (só com o treino) pelas janelas expansivas.
    '''
    if alpha is None:
        treino, _ = dividir_treino_teste(data, COL_ANO, ANO_CORTE)
        _, numericas, categoricas = montar_features(treino, usar_ano, usar_ente, so_lag_renda)
        alpha = escolher_alpha(treino, numericas, categoricas)

    resultado = treinar_cenario(data, Ridge(alpha=alpha), usar_ano, usar_ente, so_lag_renda, ANO_CORTE)
    resultado["alpha"] = alpha
    return resultado