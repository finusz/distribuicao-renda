#python -m src.dados.arrecadacao2021
"""
Arrecadação das receitas federais por UF em 2021 (Receita Federal).

Baixa as 12 planilhas mensais (.ods) da página da Receita Federal, converte cada uma
para CSV (src/dados/arrecadacao_2021/) e oferece funções para a página do app ler os CSVs.
"""
import io
from pathlib import Path

import pandas as pd
import requests

from src.util.demais_funcoes import configurar_log

#------------ LOGS ------------

PASTA_LOGS = Path(__file__).parent / "logs"
PASTA_LOGS.mkdir(parents=True, exist_ok=True)

logger = configurar_log('arrecadacao', PASTA_LOGS, 'arrecadacao.log')

#------------ CONFIGURAÇÕES ------------

URL_PAGINA = (
    "https://www.gov.br/receitafederal/pt-br/acesso-a-informacao/dados-abertos/"
    "receitadata/arrecadacao/arrecadacao-por-estado/arrecadacao-uf-2021"
)

# mês -> nome do arquivo na página (os nomes não seguem um padrão único)
ARQUIVOS = {
    1: "arrecadacao-por-uf-internet-jan21.ods",
    2: "arrecadacao-por-uf-internet-fev21.ods",
    3: "arrecadacao-por-uf-internet-mar21.ods",
    4: "arrecadacao-por-uf-internet-abr21.ods",
    5: "arrecadacao-por-uf-internet-maio21.ods",
    6: "arrecadacao-por-uf-internet-junho-2021.ods",
    7: "arrecadacao-por-uf-internet-julho-2021.ods",
    8: "arrecadacao-por-uf-internet-agosto-2021.ods",
    9: "arrecadacao-por-uf-internet-setembro-2021.ods",
    10: "arrecadacao-por-uf-internet-outubro-2021.ods",
    11: "arrecadacao-por-uf-internet-novembro-2021.ods",
    12: "arrecadacao-por-uf-internet-dezembro-2021.ods",
}

MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}

PASTA_CSV = Path(__file__).parent / "arrecadacao_2021"

COLUNA_RECEITA = "RECEITAS"

# Linhas "FINANCEIRAS"/"DEMAIS" se repetem embaixo de vários impostos;
# recebem o nome do imposto "pai" para não ficarem ambíguas.
PAIS_SUBLINHAS = {
    "IRPJ": "IRPJ",
    "COFINS": "COFINS",
    "CONTRIBUIÇÃO PARA O PIS/PASEP": "PIS/PASEP",
    "CSLL": "CSLL",
}
SUBLINHAS = {"ENTIDADES FINANCEIRAS", "DEMAIS EMPRESAS", "FINANCEIRAS", "DEMAIS"}


def caminho_csv(mes: int) -> Path:
    return PASTA_CSV / f"arrecadacao_uf_2021_{mes:02d}.csv"


#------------ EXTRAÇÃO E CONVERSÃO ------------

def baixar_ods(mes: int) -> bytes:
    url = f"{URL_PAGINA}/{ARQUIVOS[mes]}/@@download/file"
    resposta = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resposta.raise_for_status()
    if not resposta.content.startswith(b"PK"):
        raise ValueError(f"O arquivo de {MESES[mes]} não veio como planilha .ods ({url})")
    logger.info(f"{MESES[mes]}: {len(resposta.content)} bytes baixados")
    return resposta.content


def converter_ods(conteudo: bytes) -> pd.DataFrame:
    '''
    Lê a planilha .ods e devolve só a tabela: uma linha por receita, uma coluna por UF + TOTAL.
    A linha do cabeçalho muda de posição entre os meses, então ela é procurada pelo texto "RECEITAS".
    '''
    raw = pd.read_excel(io.BytesIO(conteudo), engine="odf", header=None)

    primeira_coluna = raw.iloc[:, 0].astype(str).str.strip().str.upper()
    linha_cabecalho = primeira_coluna[primeira_coluna.str.startswith(COLUNA_RECEITA)].index[0]

    df = raw.iloc[linha_cabecalho + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[linha_cabecalho]]
    df = df.dropna(how="all").reset_index(drop=True)

    df[COLUNA_RECEITA] = (
        df[COLUNA_RECEITA].astype(str)
        .str.split().str.join(" ")                       # espaços extras
        .str.replace("IMPORTACAO", "IMPORTAÇÃO")         # grafia muda entre os meses
    )

    pai = None
    for i, nome in df[COLUNA_RECEITA].items():
        if nome in PAIS_SUBLINHAS:
            pai = PAIS_SUBLINHAS[nome]
        elif nome in SUBLINHAS and pai:
            df.at[i, COLUNA_RECEITA] = f"{pai} - {nome}"

    colunas_valor = [c for c in df.columns if c != COLUNA_RECEITA]
    df[colunas_valor] = df[colunas_valor].apply(pd.to_numeric, errors="coerce")

    return df


def extrair_e_salvar():
    """Baixa os 12 meses de 2021 e grava um CSV por mês em src/dados/arrecadacao_2021/."""
    PASTA_CSV.mkdir(parents=True, exist_ok=True)
    for mes in ARQUIVOS:
        df = converter_ods(baixar_ods(mes))
        df.to_csv(caminho_csv(mes), sep=";", decimal=",", index=False,
                  encoding="utf-8-sig", float_format="%.2f")
        logger.info(f"{MESES[mes]}: {df.shape[0]} receitas salvas em {caminho_csv(mes)}")


#------------ LEITURA (usada pela página) ------------

def carregar_mes(mes: int) -> pd.DataFrame:
    return pd.read_csv(caminho_csv(mes), sep=";", decimal=",", encoding="utf-8-sig")


def carregar_ano() -> pd.DataFrame:
    '''
    Junta os 12 CSVs em formato longo: Mês, Receita, UF, Valor [R$].
    '''
    faltando = [MESES[m] for m in ARQUIVOS if not caminho_csv(m).exists()]
    if faltando:
        raise FileNotFoundError(
            f"CSVs de arrecadação faltando ({', '.join(faltando)}). "
            "Rode 'python -m src.dados.arrecadacao2021' para gerar."
        )

    partes = []
    for mes in ARQUIVOS:
        longo = carregar_mes(mes).melt(id_vars=COLUNA_RECEITA, var_name="UF", value_name="Valor [R$]")
        longo.insert(0, "Mês", mes)
        partes.append(longo)

    return pd.concat(partes, ignore_index=True).rename(columns={COLUNA_RECEITA: "Receita"})


if __name__ == "__main__":
    extrair_e_salvar()
