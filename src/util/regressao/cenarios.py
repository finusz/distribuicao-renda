import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.util.regressao.testes_estatistica import dividir_treino_teste, obter_metrics

ALVO = "Renda Total [R$ milhões]"
COL_ANO = "Ano-calendário"
COL_ENTE = "Ente Federativo"
LAG_RENDA = "Renda Total (ano anterior)"

CENARIOS = {
    "A. Só o ano anterior da renda": dict(so_lag_renda=True),
    "B. Todas as colunas lag":       dict(),
    "C. B + ano":                    dict(usar_ano=True),
    "D. B + estado":                 dict(usar_ente=True),
    "E. B + ano + estado":           dict(usar_ano=True, usar_ente=True),
}


def colunas_lag(data: pd.DataFrame) -> list[str]:
    """Retorna todas as colunas terminadas em '(ano anterior)'."""
    return [c for c in data.columns if c.endswith("(ano anterior)")]


def montar_features(data: pd.DataFrame, usar_ano: bool, usar_ente: bool, so_lag_renda: bool = False):
    """Escolhe as colunas de entrada (X) conforme o cenário. NÃO codifica nada."""
    numericas = [LAG_RENDA] if so_lag_renda else colunas_lag(data)
    if usar_ano:
        numericas = numericas + [COL_ANO]
    categoricas = [COL_ENTE] if usar_ente else []
    return data[numericas + categoricas], numericas, categoricas


def criar_pipeline(regressor, numericas: list[str], categoricas: list[str]) -> Pipeline:
    """Pré-processamento (escala + one-hot) + QUALQUER regressor (LinearRegression, Ridge...)."""
    etapas = [("num", StandardScaler(), numericas)]
    if categoricas:
        etapas.append(("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categoricas))

    return Pipeline([
        ("prep", ColumnTransformer(etapas)),
        ("reg", regressor),
    ])


def treinar_cenario(data: pd.DataFrame, regressor, usar_ano: bool = False, usar_ente: bool = False,
                    so_lag_renda: bool = False, ANO_CORTE: int = 2017) -> dict:
    '''
    Treina UM regressor em UM cenário e avalia no teste.
    Devolve o mesmo formato de treinar_regressao, para tudo poder ser comparado.

    regressor -> instância do modelo (LinearRegression(), Ridge(alpha=1), ...)
    '''
    treino, teste = dividir_treino_teste(data, COL_ANO, ANO_CORTE)

    X_tr, numericas, categoricas = montar_features(treino, usar_ano, usar_ente, so_lag_renda)
    X_te, _, _ = montar_features(teste, usar_ano, usar_ente, so_lag_renda)

    modelo = criar_pipeline(regressor, numericas, categoricas)
    modelo.fit(X_tr, treino[ALVO])

    prev_treino = modelo.predict(X_tr)
    prev_teste = modelo.predict(X_te)

    previsoes = teste[[COL_ANO, COL_ENTE, ALVO]].copy()
    previsoes["Previsto [R$ milhões]"] = prev_teste
    previsoes["Erro [R$ milhões]"] = previsoes["Previsto [R$ milhões]"] - previsoes[ALVO]
    previsoes["Erro (%)"] = previsoes["Erro [R$ milhões]"] / previsoes[ALVO] * 100

    nomes = [n.split("__", 1)[1] for n in modelo.named_steps["prep"].get_feature_names_out()]
    coeficientes = (
        pd.DataFrame({"Variável": nomes, "Coeficiente (padronizado)": modelo.named_steps["reg"].coef_})
        .sort_values("Coeficiente (padronizado)", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    return {
        "modelo": modelo,
        "variaveis": numericas + categoricas,
        "intercepto": float(modelo.named_steps["reg"].intercept_),
        "coeficientes": coeficientes,
        "metricas_treino": obter_metrics(treino[ALVO], prev_treino),
        "metricas_teste": obter_metrics(teste[ALVO], prev_teste),
        "previsoes": previsoes,
        "n_treino": len(treino),
        "n_teste": len(teste),
    }


def comparar_cenarios(data: pd.DataFrame, treinar_fn, ANO_CORTE: int = 2017):
    '''
    Roda uma função de treino em TODOS os cenários.

    treinar_fn -> função com assinatura (data, ANO_CORTE=..., usar_ano=..., usar_ente=..., so_lag_renda=...)
                  (treinar_linear ou treinar_ridge)
    Retorna (tabela ordenada por MAPE, dicionário {nome do cenário: resultado completo}).
    '''
    resultados, linhas = {}, {}

    for nome, params in CENARIOS.items():
        res = treinar_fn(data, ANO_CORTE=ANO_CORTE, **params)
        resultados[nome] = res

        linha = {
            "MAE [R$ milhões]": res["metricas_teste"]["MAE"],
            "MAPE (%)": res["previsoes"]["Erro (%)"].abs().mean(),
        }
        if "alpha" in res:
            linha["alpha"] = res["alpha"]
        linhas[nome] = linha

    tabela = pd.DataFrame(linhas).T.sort_values("MAPE (%)")
    return tabela, resultados

def prever_proximo_ano(data: pd.DataFrame, futuro: pd.DataFrame, regressor,
                       usar_ano: bool = False, usar_ente: bool = False,
                       so_lag_renda: bool = False) -> pd.DataFrame:
    '''
    Treina o modelo com TODOS os anos disponíveis e prevê o ano seguinte.

    data -> dfAgregado (histórico completo)
    futuro -> dfFuturo (valores do último ano, já com nomes de "ano anterior")
    regressor -> LinearRegression() ou Ridge(alpha=...)
    '''
    X_tr, numericas, categoricas = montar_features(data, usar_ano, usar_ente, so_lag_renda)

    modelo = criar_pipeline(regressor, numericas, categoricas)
    modelo.fit(X_tr, data[ALVO])

    saida = futuro[[COL_ENTE, COL_ANO]].copy()
    saida["Previsto [R$ milhões]"] = modelo.predict(futuro[numericas + categoricas])
    return saida