from src.dados.limpezaDados import dfAgregado, dfFuturo
from src.util.regressao.cenarios import comparar_cenarios, prever_proximo_ano, CENARIOS
from src.util.graficos import real_previsto, grafico_previsao
from src.util.regressao.testes_estatistica import *
from src.util.regressao.cenarios import comparar_cenarios
from src.util.regressao.modelos.regressao_linear_mult import treinar_linear
from src.util.regressao.modelos.regressao_ridge import treinar_ridge
from src.util.graficos import real_previsto

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from src.util.eda import renderizar_eda
from src.util.tema import cartao

import streamlit as st

import pandas as pd

with cartao("predito_titulo"):
    st.title('Predição da distruibuição de renda', text_alignment="center")
    st.markdown(
            """Como dito na página inicial, a predição ocorrerá apenas para o próximo ano, e isso ocorre devido a como o modelo foi treinado,
            considerando os dados do ano anterior para prever a renda total do ano seguinte .""",
            text_alignment="center",
        )

with cartao("predito_preparo"):
    st.markdown(
        '''
Para atingir o objetivo, o conjunto de dados foi mais uma vez modificado, modificando o conjunto transformado. Na modificação, 
a base inclui uma comparação do RB2 do ano atual com o anterior, e também pega os valores do ano anterior:
- O conjunto transformado foi agrupado com base no ano e ente federativo, ordenando pelo ano;
- Os valores de 2006 foram excluídos da tabela por serem os primeiros anos do conjunto de dados, logo, não possuem um ano anterior para realizar
a comparação. (se por acaso retirarem o ano de 2006 futuramente, ou adicionarem <=2005, o primeiro ano do conjunto de dados vai ser excluído da tabela);
- A coluna "Rendimentos Tributáveis Totais [R\$ milhões]" foi renomeada para "Renda Total [R\$ milhões]";
- As colunas dos anos anteriores escolhidas não fazem parte do cálculo padrão do RB2, o que acaba não comentendo vazamento de dados.

A coluna a ser predita, o nosso target, será "Renda Total [R\$ milhões]". 

E as características utilizadas para prever serão:
- Ano-calendário;
- Ente Federativo;
- Renda Total (ano anterior);
- Quantidade de Contribuintes (ano anterior);
- Rendimentos Isentos - Outros Rendimentos Isentos (ano anterior);
- Imposto Devido (ano anterior);
- Despesas Dedutíveis Totais (ano anterior);
- Bens e Direitos Totais (ano anterior);
- Dívidas e Ônus (ano anterior).

Todos os modelos testados foram treinados com os anos até 2017 e testados com os anos de 2018 a 2020
'''
    )

renderizar_eda(dfAgregado, 'dfAgregado')

#------------ MODELO ------------

ANO_CORTE = 2017


@st.cache_data(show_spinner="Treinando os modelos nos 5 cenários...")
def rodar_cenarios(ano_corte: int):
    """Treina Linear e Ridge nos 5 cenários. O cache evita refazer tudo a cada clique no Streamlit."""
    return (
        comparar_cenarios(dfAgregado, treinar_linear, ano_corte),
        comparar_cenarios(dfAgregado, treinar_ridge, ano_corte),
    )


(tab_lin, res_lin), (tab_rid, res_rid) = rodar_cenarios(ANO_CORTE)

melhor_lin = tab_lin.index[0]   # as tabelas já vêm ordenadas pelo MAPE
melhor_rid = tab_rid.index[0]


def exibir_modelo(res: dict, cenario: str, eh_ridge: bool = False):
    """Desenha o conteúdo de um modelo (métricas, tabela treino/teste, explicações e coeficientes)."""
    mt = res["metricas_teste"]
    erro_medio_pct = res["previsoes"]["Erro (%)"].abs().mean()

    st.caption(f"Melhor cenário: **{cenario}**")
    st.markdown("##### Desempenho no conjunto de teste")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("R²", f"{mt['R²']:.4f}")
    with c2:
        st.metric("RMSE [R$ milhões]", f"{mt['RMSE']:,.2f}")
    with c3:
        st.metric("MAE [R$ milhões]", f"{mt['MAE']:,.2f}")

    st.dataframe(
        pd.DataFrame({"Treino": res["metricas_treino"], "Teste": res["metricas_teste"]}),
        width="stretch",
    )

    texto_alpha = f"\n        - **alpha** escolhido (força da penalização): **{res['alpha']}**;" if eh_ridge else ""

    st.markdown(f'''
        - **R²**: proporção da variação da renda explicada pelo modelo (1 = perfeito);
        - **RMSE**: erro médio em R\\$ milhões, penalizando mais os erros grandes;
        - **MAE**: erro absoluto médio em R\\$ milhões;
        - Erro percentual absoluto médio no teste: **{erro_medio_pct:.2f}%**.{texto_alpha}

        Obs.: como os estados têm tamanhos muito diferentes (SP é muito maior que RR, por exemplo), o R² tende a ficar alto
        de qualquer forma; o MAE e o erro percentual mostram melhor o quanto o modelo erra em cada estado.

        ##### Coeficientes do modelo

        Os coeficientes estão **padronizados** (as variáveis foram colocadas na mesma escala): cada um indica quanto a renda
        prevista muda (em R\\$ milhões) quando a variável aumenta 1 desvio padrão, mantendo as outras fixas.
        Intercepto: **{res["intercepto"]:,.2f}**.

        Obs.: as variáveis do ano anterior são muito correlacionadas entre si (ver mapa de correlação na página "Sobre"),
        então os coeficientes individuais devem ser interpretados com cuidado.
        ''')
    st.dataframe(res["coeficientes"], width="stretch", hide_index=True)

    st.markdown("##### Real x Previsto (teste)")
    st.markdown(
        "Cada ponto é um estado em um ano do teste. A linha tracejada é a previsão perfeita: pontos **acima** dela "
        "indicam que o modelo previu mais do que aconteceu (superestimou), e pontos **abaixo**, que previu menos "
        "(subestimou). Os eixos estão em escala logarítmica porque a renda de SP é dezenas de vezes maior que a dos "
        "estados menores."
    )
    st.pyplot(real_previsto(res["previsoes"], ALVO))

    st.markdown("##### Previsões por estado (teste)")
    st.dataframe(
        res["previsoes"].sort_values(
            ["Ano-calendário", "Erro (%)"],
            key=lambda s: s.abs() if s.name == "Erro (%)" else s,
        ),
        width="stretch",
        hide_index=True,
    )

    
with cartao("avaliação_modelos"):
    st.title("Avaliando a escolha do modelo", text_alignment='center')
    st.markdown("""
    Para avaliar corretamente qual modelo deve ser utilizado, separamos dois modelos diferentes para serem considerados, e a escolha vai ser definida pelo qual apresentar
    melhor rendimento. O teste será feito por ano, não contendo aleatoriedade. Ambos passam pelos mesmos cenários de variáveis,
    e o que aparece abaixo é o **melhor cenário de cada modelo**.
    """)

with cartao("cenarios"):
    st.subheader("Verificando Cenários", text_alignment='center')
    st.markdown('''
    Primeiro, será testado através do calculo do MAE e MAPE se o ano e o estado fornecem alguma mudança significativa no erro médio absoluto.
    Para isso foram testados 5 cenários, em cada um dos dois modelos:
    - A. Só o ano anterior da renda;
    - B. Todas as colunas lags;
    - C. B + ano;
    - D. B + estado;
    - E. B + ano + estado.
    ''')

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("##### Regressão Linear Múltipla")
        st.dataframe(tab_lin, width="stretch")
    with t2:
        st.markdown("##### Regressão Ridge")
        st.dataframe(tab_rid, width="stretch")

    st.markdown(f'''
O cenário com menor erro médio percentual absoluto foi **{melhor_lin}** na Regressão Linear Múltipla
({tab_lin.iloc[0]["MAPE (%)"]:.2f}%) e **{melhor_rid}** na Regressão Ridge ({tab_rid.iloc[0]["MAPE (%)"]:.2f}%).
Olhando apenas para o MAE, o menor erro absoluto foi **{tab_lin["MAE [R$ milhões]"].idxmin()}** na Linear
({tab_lin["MAE [R$ milhões]"].min():,.2f} [em milhões de R\\$]) e **{tab_rid["MAE [R$ milhões]"].idxmin()}** no Ridge
({tab_rid["MAE [R$ milhões]"].min():,.2f} [em milhões de R\\$]).
    ''')

col_lin, col_rid = st.columns(2)
with col_lin:
    with cartao("av_reg_mult"):
        st.subheader("Regressão Linear Múltipla", text_alignment='center')
        exibir_modelo(res_lin[melhor_lin], melhor_lin)

with col_rid:
    with cartao("av_reg_ridge"):
        st.subheader("Regressão Ridge", text_alignment='center')
        exibir_modelo(res_rid[melhor_rid], melhor_rid, eh_ridge=True)

with cartao("comparacao_final"):
    st.title("Comparando os melhores de cada modelo")
    tabela_final = resumo_modelos(dfAgregado, {
        f"Linear ({melhor_lin})": res_lin[melhor_lin],
        f"Ridge ({melhor_rid})": res_rid[melhor_rid],
    }, ANO_CORTE)
    st.dataframe(tabela_final, width="stretch")

    vencedor = tabela_final.index[0]
    st.markdown(f'''
O menor erro percentual médio foi o de **{vencedor}** ({tabela_final.iloc[0]["MAPE (%)"]:.2f}%).
A linha "Baseline" repete a renda do ano anterior: se um modelo não ficar abaixo dela, ele não se justifica.
Como o modelo de Regressão Linear apresentou o menor MAE e MAPE, ela acaba sendo escolhida para predizer o anos seguintes,
e para isso, será treinado utilizando todos os anos existentes dentro do dataset.
    ''')

with cartao("valor_futuro"):
    st.title("Previsão de anos seguintes", text_alignment='center')

@st.cache_data(show_spinner="Treinando o modelo final...")
def previsoes_proximo_ano(cenario: str):
    """Treina a Linear no cenário vencedor com todos os anos e prevê o ano seguinte para os 27 estados."""
    return prever_proximo_ano(dfAgregado, dfFuturo, LinearRegression(), **CENARIOS[cenario])


prev_futuro = previsoes_proximo_ano(melhor_lin)
ano_prev = int(dfFuturo["Ano-calendário"].iloc[0])
erro_teste = tab_lin.iloc[0]["MAPE (%)"]
estados = sorted(dfAgregado["Ente Federativo"].unique())

with cartao("valor_previsto"):
    st.title(f"Previsão de {ano_prev}", text_alignment="center")
    st.markdown(
        f"O modelo escolhido ({melhor_lin}, Regressão Linear) foi treinado com todos os anos do conjunto. "
        f"Como ele usa os valores do ano anterior, só consegue prever {ano_prev}, a partir dos dados de {ano_prev - 1}."
    )

param, graf = st.columns([1, 2])

with param:
    with cartao("escolhas"):
        st.markdown("#### Selecione um estado")
        opcao_est = st.selectbox("Estado", estados, key="sel_estado_previsao",
                                 label_visibility="collapsed")

        historico = dfAgregado[dfAgregado["Ente Federativo"] == opcao_est]
        previsto = prev_futuro.loc[
            prev_futuro["Ente Federativo"] == opcao_est, "Previsto [R$ milhões]"
        ].iloc[0]
        ultimo_real = historico.sort_values("Ano-calendário")["Renda Total [R$ milhões]"].iloc[-1]

        st.metric(f"Renda prevista para {ano_prev} [R$ milhões]", f"{previsto:,.2f}")

with graf:
    with cartao("grafico_previsao"):
        fig_prev = grafico_previsao(historico, ano_prev, previsto, opcao_est, faixa_pct=erro_teste)
        st.pyplot(fig_prev)
        plt.close(fig_prev)
