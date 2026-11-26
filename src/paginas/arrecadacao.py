from src.dados.arrecadacao2021 import carregar_ano, carregar_mes, caminho_csv, MESES, URL_PAGINA
from src.util.graficos import bar_lateral_dados
from src.util.eda import renderizar_eda
from src.util.tema import cartao

import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt

COLUNA_VALOR = "Valor [R$ bilhões]"


@st.cache_data
def dados_ano() -> pd.DataFrame:
    '''Junta as 12 tabelas mensais em um só conjunto (formato longo), com o valor em R$ bilhões.'''
    data = carregar_ano()
    data[COLUNA_VALOR] = data.pop("Valor [R$]") / 1e9
    data["Mês"] = data["Mês"].map(MESES)
    return data


dfArrecadacao = dados_ano()
dfEstados = dfArrecadacao[dfArrecadacao["UF"] != "TOTAL"]     # sem a coluna TOTAL, para não somar em dobro
receitas = list(dict.fromkeys(dfArrecadacao["Receita"]))      # na ordem da planilha

with cartao("arrecadacao_titulo"):
    st.title("Arrecadação federal por estado em 2021", text_alignment="center")
    st.markdown(
        """Arrecadação mensal das receitas federais (impostos e contribuições) em cada estado no ano de 2021,
        com destaque para o Imposto de Renda da Pessoa Física (IRPF).""",
        text_alignment="center",
    )

with cartao("arrecadacao_dados"):
    st.subheader("Sobre os dados")
    st.markdown(f"""
Os dados foram obtidos através da página da Receita Federal: <{URL_PAGINA}>.

A página fornece 12 tabelas, uma para cada mês de 2021, no formato .ods (planilha do LibreOffice). Cada tabela traz o valor
arrecadado de cada receita federal em cada estado no mês.

- As tabelas foram convertidas para CSV, uma por mês, na pasta `src/dados/arrecadacao_2021/`
(para gerar de novo: `python -m src.dados.arrecadacao2021`);
- Os valores das tabelas estão em R\\$ 1,00 (preços correntes, sem correção pela inflação); nos gráficos, em R\\$ bilhões;
- As linhas "FINANCEIRAS" e "DEMAIS" se repetiam embaixo de vários impostos, então receberam o nome do imposto
(ex.: "COFINS - FINANCEIRAS") para não ficarem ambíguas.

O **IRPF** é o imposto pago pela própria pessoa física (declaração, carnê-leão, ganho de capital). O imposto descontado
direto do salário aparece em outra linha, "IRRF - RENDIMENTOS DO TRABALHO".
""")

aba1, aba2 = st.tabs(['Tabelas mensais', 'Conjunto agregado (2021)'])

with aba1:
    with cartao("arrecadacao_mes"):
        st.subheader("Tabela do mês")
        c_esq, c_dir = st.columns([1, 3])
        with c_esq:
            mes = st.selectbox("Selecione o mês", list(MESES), format_func=lambda m: MESES[m], index=11, key="sel_mes")
        with c_dir:
            st.download_button(
                f"Baixar CSV de {MESES[mes]}",
                data=caminho_csv(mes).read_bytes(),
                file_name=caminho_csv(mes).name,
                mime="text/csv",
            )

    renderizar_eda(carregar_mes(mes), prefixo='arrec_mes')

with aba2:
    with cartao("arrecadacao_agregado"):
        st.markdown('''
Para comparar os meses e os estados, as 12 tabelas foram juntadas em um único conjunto, com uma linha para cada
combinação de mês, receita e estado:
- Mês;
- Receita;
- UF (os 27 estados e o TOTAL);
- Valor [R$ bilhões].
        ''')

    renderizar_eda(dfArrecadacao, prefixo='arrec_ano')

    with cartao("arrecadacao_receita"):
        opcao_receita = st.selectbox('Selecione uma receita', receitas, index=receitas.index("IRPF"), key='sel_receita')

    dfReceita = dfEstados[dfEstados["Receita"] == opcao_receita]

    col1, col2 = st.columns(2)
    with col1:
        with cartao("arrecadacao_estados"):
            st.subheader('Comparação entre os entes federativos')
            fig_uf, top5_uf = bar_lateral_dados(dfReceita, eixo_x=COLUNA_VALOR, eixo_y='UF')
            st.pyplot(fig_uf)
            plt.close(fig_uf)
            participacao = dfReceita.groupby("UF")[COLUNA_VALOR].sum()
            participacao = participacao / participacao.sum() * 100
            st.markdown(f"""Os principais Entes Federativos na arrecadação de **{opcao_receita}** em 2021 (R\\$ bilhões):

{top5_uf}

Só **{participacao.idxmax()}** concentra **{f"{participacao.max():.1f}".replace(".", ",")}%** do total do país.""")

    with col2:
        with cartao("arrecadacao_meses"):
            st.subheader('Comparação entre os meses')
            fig_mes, top5_mes = bar_lateral_dados(dfReceita, eixo_x=COLUNA_VALOR, eixo_y='Mês')
            st.pyplot(fig_mes)
            plt.close(fig_mes)
            st.markdown(f"""Os meses com maior arrecadação de **{opcao_receita}** em 2021 (R\\$ bilhões):

{top5_mes}""")
            if opcao_receita == "IRPF":
                st.markdown("""Maio se destaca porque foi o mês do prazo final da declaração em 2021, quando vencem a
cota única e a primeira cota do imposto a pagar.""")
