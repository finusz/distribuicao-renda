import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALVO = "Renda Total [R$ milhões]"
COL_ANO = "Ano-calendário"
COL_ENTE = "Ente Federativo"
LAG_RENDA = "Renda Total (ano anterior)"


def colunas_lag(data: pd.DataFrame) -> list[str]:
    """Retorna todas as colunas terminadas em '(ano anterior)'."""
    return [c for c in data.columns if c.endswith("(ano anterior)")]


def montar_features(data, usar_ano: bool, usar_ente: bool, so_lag_renda: bool = False):
    """Escolhe as colunas de entrada (X). Não codifica nada."""
    numericas = [LAG_RENDA] if so_lag_renda else colunas_lag(data)
    if usar_ano:
        numericas = numericas + [COL_ANO]
    categoricas = [COL_ENTE] if usar_ente else []
    return data[numericas + categoricas], numericas, categoricas


def criar_pipeline(estimador, numericas: list[str], categoricas: list[str]) -> Pipeline:
    """Pré-processamento (escala + one-hot) + o estimador recebido."""
    etapas = [("num", StandardScaler(), numericas)]
    if categoricas:
        etapas.append(("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categoricas))

    return Pipeline([
        ("prep", ColumnTransformer(etapas)),
        ("reg", clone(estimador)),
    ])


def dividir_por_ano(data: pd.DataFrame, anos_teste: int = 3):
    """Os últimos `anos_teste` anos do dataset viram teste; o resto é treino."""
    anos = sorted(data[COL_ANO].unique())
    if anos_teste >= len(anos):
        raise ValueError(f"anos_teste={anos_teste}, mas só há {len(anos)} anos no conjunto.")
    corte = anos[-anos_teste]
    return data[data[COL_ANO] < corte], data[data[COL_ANO] >= corte]


def obter_metrics(y_real, y_previsto) -> dict:
    """R², RMSE e MAE (R$ milhões) e MAPE (fração; 0.05 = 5%)."""
    return {
        "R²": r2_score(y_real, y_previsto),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_previsto))),
        "MAE": mean_absolute_error(y_real, y_previsto),
        "MAPE": mean_absolute_percentage_error(y_real, y_previsto),
    }


def avaliar(data, estimador, anos_teste: int = 3,
            usar_ano: bool = True, usar_ente: bool = True, so_lag_renda: bool = False) -> dict:
    """Treina `estimador` nos anos antigos e avalia nos últimos `anos_teste` anos."""
    treino, teste = dividir_por_ano(data, anos_teste)
    X_tr, num, cat = montar_features(treino, usar_ano, usar_ente, so_lag_renda)
    X_te, _, _ = montar_features(teste, usar_ano, usar_ente, so_lag_renda)

    modelo = criar_pipeline(estimador, num, cat)
    modelo.fit(X_tr, treino[ALVO])
    prev_tr = modelo.predict(X_tr)
    prev_te = modelo.predict(X_te)

    previsoes = teste[[COL_ANO, COL_ENTE, ALVO]].copy()
    previsoes["Previsto [R$ milhões]"] = prev_te
    previsoes["Erro [R$ milhões]"] = previsoes["Previsto [R$ milhões]"] - previsoes[ALVO]
    previsoes["Erro (%)"] = previsoes["Erro [R$ milhões]"] / previsoes[ALVO] * 100

    reg = modelo.named_steps["reg"]
    nomes = modelo.named_steps["prep"].get_feature_names_out()
    coeficientes = (
        pd.DataFrame({"Variável": nomes, "Coeficiente": reg.coef_})
        .sort_values("Coeficiente", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    return {
        "modelo": modelo,
        "intercepto": float(reg.intercept_),
        "coeficientes": coeficientes,
        "metricas_treino": obter_metrics(treino[ALVO], prev_tr),
        "metricas_teste": obter_metrics(teste[ALVO], prev_te),
        "previsoes": previsoes,
        "n_treino": len(treino),
        "n_teste": len(teste),
    }


def baseline_ano_anterior(data: pd.DataFrame, anos_teste: int = 3) -> dict:
    """Previsão ingênua: renda deste ano = renda do ano anterior."""
    _, teste = dividir_por_ano(data, anos_teste)
    return obter_metrics(teste[ALVO], teste[LAG_RENDA])