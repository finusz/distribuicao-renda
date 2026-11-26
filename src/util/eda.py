import streamlit as st
import html

import pandas as pd

from src.util.tema import cartao

def n_valores(data:pd.DataFrame, n:int=5, primeiros:bool=True):
    '''
    Função para exibir n valores de um dataset

    data -> Recebe um dataframe;
    n -> Recebe um número inteiro (por padrão, 5) que indica a quantidade de valores a serem exibidos;
    primeiros -> Recebe True/False  (por padrão, True), se verdadeiro, retorna os primeiros itens, se falso, os últimos.
    '''
    if primeiros:
        return data.head(n)

    return data.tail(n)

ESTILO_TABELA_HTML = """<style>
.info-table {
    width: 100%;
    border-collapse: collapse;
    font-family: sans-serif;
}
.info-header {
    display: flex;
    justify-content: space-between;
    padding: 8px 4px;
    border-bottom: 1.5px solid rgba(128, 128, 128, 0.35);
    color: rgba(128, 128, 128, 0.9);
    font-size: 0.9rem;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 4px;
    border-bottom: 1px solid rgba(128, 128, 128, 0.18);
    gap: 16px;
}
.info-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 4px;
}
.info-desc {
    font-size: 0.88rem;
    color: rgba(128, 128, 128, 0.9);
}
.info-right {
    font-size: 1.15rem;
    color: rgba(128, 128, 128, 0.95);
    white-space: nowrap;
    text-align: right;
}
</style>"""


def renderizar_eda(data:pd.DataFrame, prefixo: str):
    '''
    Retornar uma pequena analise do dataframe.
    data -> o conjunto de dados (no formato pd.DataFrame)
    prefixo -> Recebe um string que deve ter o nome da variavel que armazena o conjunto de dados
    '''
    col1, col2, col3 = st.columns(3)
    with col1:
        with cartao(f"{prefixo}_linhas"):
            st.metric("Linhas (características)", f"{len(data):,}".replace(",", "."))
    with col2:
        with cartao(f"{prefixo}_colunas"):
            st.metric("Colunas (atributos)", len(data.columns))
    with col3:
        with cartao(f"{prefixo}_nulos"):
            st.metric("Valores nulos", int(data.isna().sum().sum()))

    linhas_html = []
    for col in data.columns:
        nome_col = html.escape(str(col))
        qtd_valores = f"{int(data[col].count()):,}".replace(",", ".")
        tipo_dado = html.escape(str(data[col].dtype))

        linhas_html.append(f"""<div class="info-row">
<div class="info-left">
<div class="info-title">{nome_col}</div>
<div class="info-desc">Tipo de dado (dtype): <code>{tipo_dado}</code></div>
</div>
<div class="info-right">{qtd_valores} valores</div>
</div>""")

    tabela_html = f"""{ESTILO_TABELA_HTML}
<div class="info-table">
<div class="info-header">
<span>Coluna</span>
<span>Quantidade de valores</span>
</div>
{"".join(linhas_html)}
</div>"""

    with cartao(f"{prefixo}_colunas_info"):
        st.markdown(tabela_html, unsafe_allow_html=True)

    with cartao(f"{prefixo}_descricao"):
        st.subheader('Descrição do conjunto de dados')
        colunas_irrelevantes = ['Ano-calendário', 'Centil', 'Subcentil', 'Nivel', 'Ente Federativo'] #Colunas que não são relevantes para o describe
        descricao = data.drop(columns=colunas_irrelevantes, errors="ignore").describe(include="all")
        # colunas de texto no describe misturam números (count, unique) e texto (top),
        # o que o Streamlit (pyarrow) não consegue exibir; elas são convertidas para texto
        for col in descricao.columns[descricao.dtypes == object]:
            descricao[col] = descricao[col].map(lambda v: None if pd.isna(v) else str(v))
        st.dataframe(descricao)

    with cartao(f"{prefixo}_exibicao"):
        st.subheader("Exibição do conjunto de dados")

        c_esq, c_dir = st.columns([1, 3])

        with c_esq:
            n = st.number_input(
                "Escolha a quantidade de linhas:",
                min_value=1,
                max_value=max(1, len(data)),
                value=min(5, max(1, len(data))),
                step=1,
                key=f"num_linhas_{prefixo}",
                width=150
            )

            opcao = st.radio(
                "Selecione quais linhas vão ser exibidas:",
                ("Primeiras", "Últimas"),
                key=f"radio_linhas_{prefixo}",
            )
            opcao_bool = opcao == "Primeiras"

        with c_dir:
            st.dataframe(
                n_valores(data, n=n, primeiros=opcao_bool),
                width="stretch"
                         )