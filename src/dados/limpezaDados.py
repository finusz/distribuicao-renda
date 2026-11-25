from pathlib import Path

import pandas as pd

from src.dados.extracaoDados import main
from src.util.demais_funcoes import configurar_log

#------------ LOGS ------------

PASTA_LOGS = Path(__file__).parent / "logs"
PASTA_LOGS.mkdir(parents=True, exist_ok=True)

logger = configurar_log('limpeza',
                        PASTA_LOGS,
                        'limpeza.log')

#------------ CARREGAMENTO ------------

dfO = main()

#------------ TRADUÇÃO ------------

nome_colunas = [
    "Ano-calendário",
    "Ente Federativo",
    "Centil",
    "Quantidade de Contribuintes",
    "Rendimentos Tributáveis - Limite Superior da RB2 do Centil [R$ milhões]",
    "Rendimentos Tributáveis - Soma da RB2 do Centil [R$ milhões]",
    "Rendimentos Tributáveis - RB2 Acumulada do Centil [R$ milhões]",
    "Rendimentos Tributáveis - Média da RB2 do Centil [R$]",
    "Rendimentos Sujeitos à Tribut. Exclusiva [R$ milhões]",
    "Rendimentos Isentos - Lucros e dividendos [R$ milhões]",
    "Rendimentos Isentos - Rendim. Sócio/Titular ME/EPP Opt SIMPLES [R$ milhões]",
    "Rendimentos Isentos - Outros Rendimentos Isentos [R$ milhões]",
    "Despesas Dedutíveis - Previdência [R$ milhões]",
    "Despesas Dedutíveis - Dependentes [R$ milhões]",
    "Despesas Dedutíveis - Instrução [R$ milhões]",
    "Despesas Dedutíveis - Médicas [R$ milhões]",
    "Despesas Dedutíveis - Pensão Alimentícia [R$ milhões]",
    "Despesas Dedutíveis - Livro-Caixa [R$ milhões]",
    "Imposto Devido [R$ milhões]",
    "Bens e Direitos - Imóveis [R$ milhões]",
    "Bens e Direitos - Móveis [R$ milhões]",
    "Bens e Direitos - Financeiros [R$ milhões]",
    "Bens e Direitos - Outros Bens e Direitos [R$ milhões]",
    "Dívidas e Ônus [R$ milhões]",
]

colunas_unicas = [
    "Ano-calendário",
    "Ente Federativo",
    "Centil",
    "Quantidade de Contribuintes",
    "Nivel",
    "Subcentil",
    "Rendimentos Sujeitos à Tribut. Exclusiva [R$ milhões]",
    "Imposto Devido [R$ milhões]",
    "Dívidas e Ônus [R$ milhões]",
    "Rendimentos Isentos - Outros Rendimentos Isentos [R$ milhões]"
]

if len(dfO.columns) != len(nome_colunas):
    logger.error(f"{len(dfO.columns)} colunas no DataFrame, {len(nome_colunas)} nomes na lista")
    raise ValueError("Quantidade de colunas diferente do esperado")

for antigo, novo in zip(dfO.columns, nome_colunas):
    logger.info(f"coluna '{antigo}' alterada para '{novo}'")

dfO.columns = nome_colunas

#------------ SEPARAÇÃO DO CENTIL ------------
# O centil 100 é aberto em 10 subgrupos (100.1 ... 100.10).
# esse trecho apenas separa em duas colunas, uma mantendo o subgrupo e outra juntando eles

def separar_centil(data):
    data = data.copy()
    texto = data["Centil"].astype(str).str.strip()
    partes = texto.str.split(".", n=1, expand=True)

    data["Centil"] = pd.to_numeric(partes[0]).astype(int)

    if partes.shape[1] > 1:
        data["Subcentil"] = partes[1].fillna("")
    else:
        data["Subcentil"] = ""

    # 0 = centil normal, 1 = 100.x, 2 = 100.10.x
    data["Nivel"] = texto.str.count(r"\.").astype(int)

    colunas = list(data.columns)
    for c in ("Subcentil", "Nivel"):
        colunas.remove(c)
    i = colunas.index("Centil") + 1
    colunas[i:i] = ["Subcentil", "Nivel"]

    logger.info("coluna 'Centil' separada em 'Centil', 'Subcentil' e 'Nivel'")
    return data[colunas]


dfO = separar_centil(dfO)


#------------ MUDANÇA DE TIPO ------------

COLUNAS_IGNORADAS = [
    "Ano-calendário",
    "Ente Federativo",
    "Centil",
    "Quantidade de Contribuintes",
    "Nivel",
    "Subcentil"
]


def _para_float(serie):
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)

    serie = (
        serie.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(serie, errors="coerce")


def converter_tipos(data, colunas_ignoradas=COLUNAS_IGNORADAS):
    """Converte as colunas numéricas para float e 'Quantidade de Contribuintes' para inteiro."""
    data = data.copy()

    for coluna in data.columns:
        if coluna in colunas_ignoradas:
            continue
        data[coluna] = _para_float(data[coluna])
        logger.info(f"coluna '{coluna}' convertida para float")

    c = "Centil"
    data[c] = _para_float(data[c]).round().astype("Int64")
    logger.info(f"coluna '{c} convertida para inteiro")

    return data


dfO = converter_tipos(dfO)

logger.info('Dataset original com os tipos de dados alterados')
logger.info(dfO.head())

#------------ DATASET COMPACTADO ------------

grupos_compactados = {
    "Rendimentos Tributáveis Totais [R$ milhões]": [
        "Rendimentos Tributáveis - Soma da RB2 do Centil [R$ milhões]"
    ],
    "Rendimentos Isentos Totais [R$ milhões]": [
        "Rendimentos Isentos - Lucros e dividendos [R$ milhões]",
        "Rendimentos Isentos - Rendim. Sócio/Titular ME/EPP Opt SIMPLES [R$ milhões]"
    ],
    "Despesas Dedutíveis Totais [R$ milhões]": [
        "Despesas Dedutíveis - Previdência [R$ milhões]",
        "Despesas Dedutíveis - Dependentes [R$ milhões]",
        "Despesas Dedutíveis - Instrução [R$ milhões]",
        "Despesas Dedutíveis - Médicas [R$ milhões]",
        "Despesas Dedutíveis - Pensão Alimentícia [R$ milhões]",
        "Despesas Dedutíveis - Livro-Caixa [R$ milhões]",
    ],
    "Bens e Direitos Totais [R$ milhões]": [
        "Bens e Direitos - Imóveis [R$ milhões]",
        "Bens e Direitos - Móveis [R$ milhões]",
        "Bens e Direitos - Financeiros [R$ milhões]",
        "Bens e Direitos - Outros Bens e Direitos [R$ milhões]",
    ],
}

dfT = dfO[colunas_unicas].copy()

for nome_coluna, colunas in grupos_compactados.items():
    dfT[nome_coluna] = dfO[colunas].sum(axis=1)

logger.info('Dataset compactado/transformado')
logger.info(dfT.head())

dfT = (
    dfT[(dfT['Ente Federativo'] != 'BRASIL') & (dfT["Nivel"] == 0)]
        .drop(columns=["Subcentil", "Nivel"])
)
logger.info('Exclusão das linhas que tem o ente federativo do brasil da base de dados')

#------------ DATASET AGRAGADO ------------
#Por causa do grande volume de dados distribuidos entre os centis, esse dataset agrupa o ano e ente federativo dos itens

dfAgregado = (
    dfT
    .drop(columns=["Centil", "Rendimentos Isentos Totais [R$ milhões]", "Rendimentos Sujeitos à Tribut. Exclusiva [R$ milhões]"])
    .groupby(['Ano-calendário', 'Ente Federativo'], as_index=False)
    .sum(numeric_only=True)
)
logger.info('data set agregado criado')
logger.info(dfAgregado.head())

dfAgregado = dfAgregado.rename(columns={
    "Rendimentos Tributáveis Totais [R$ milhões]": "Renda Total [R$ milhões]"
})
logger.info('Mudança do nome da coluna Rendimentos tributáveis totais por Renda Total')

#------------ DEFASAGEM (LAG) ------------
# Para prever o ano seguinte, só podemos usar informações já conhecidas: os valores do ano anterior.

ALVO = "Renda Total [R$ milhões]"

COLUNAS_LAG = [
    ALVO,
    "Quantidade de Contribuintes",
    "Rendimentos Isentos - Outros Rendimentos Isentos [R$ milhões]",
    "Imposto Devido [R$ milhões]",
    "Despesas Dedutíveis Totais [R$ milhões]",
    "Bens e Direitos Totais [R$ milhões]",
    "Dívidas e Ônus [R$ milhões]",
]

dfAgregado = dfAgregado.sort_values(["Ente Federativo", "Ano-calendário"]).reset_index(drop=True)

grupo = dfAgregado.groupby("Ente Federativo")
ano_anterior_ok = grupo["Ano-calendário"].diff() == 1 #Verifica se é realmente o ano anterior

colunas_lag_criadas = []
for coluna in COLUNAS_LAG:
    nome_lag = coluna.replace(" [R$ milhões]", "") + " (ano anterior)"
    dfAgregado[nome_lag] = grupo[coluna].shift(1).astype(float).where(ano_anterior_ok)
    colunas_lag_criadas.append(nome_lag)
    logger.info(f"Criada a coluna '{nome_lag}'")

dfAgregado = dfAgregado.drop(columns=[c for c in COLUNAS_LAG if c != ALVO])
dfAgregado = dfAgregado.dropna(subset=colunas_lag_criadas)
logger.info("Removidas as variáveis do mesmo ano e as linhas sem ano anterior")

dfAgregado['Ente Federativo'] = dfAgregado['Ente Federativo'].astype('category')