from src.dados.limpezaDados import dfAgregado
from src.util.eda import renderizar_eda
from src.util.tema import cartao
<<<<<<< HEAD
from src.util.regressao.testes_estatistica import teste_cenarios
=======
from src.util.modelo import treinar_modelos, ALVO, ANO_CORTE
>>>>>>> 09ee3c8 (tabelas)

import streamlit as st
import html

import pandas as pd

with cartao("predito_titulo"):
    st.title('Predição da distruibuição de renda', text_alignment="center")

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

Todos os modelos testados usaram os últimos 4 anos disponíveis no conjunto de dados para teste
'''
    )

renderizar_eda(dfAgregado, 'dfAgregado')


<<<<<<< HEAD
with cartao("cenarios"):
    st.title("Verificando Cenários")
    st.markdown('''
    Primeiro, será testado através do calculo do MAE e MAPE se o ano e o estado fornecem alguma mudança significativa no erro médio absoluto.
    Para isso foram testados 5 cenários:
    - A. Só o ano anterior da renda;
    - B. Todas as colunas lags;
    - C. B + ano;
    - D. B + estado;
    - E. B + ano + estado.
    ''')
    resultado_cenarios = teste_cenarios(dfAgregado)
    st.dataframe(resultado_cenarios)
    st.markdown(f'''
O cenário que mostrou o menor erro  médio porcentual absoluto foi o cenário E, que considera todas as colunas + ano + estado, 
com {(resultado_cenarios.iloc[0, 1])*100:.2f}% de erro médio. O cenário mostra que para Regressão Linear Múltipla ele é ideal, porém, olhando apenas para o MAE, o cenário A apresenta
o menor erro médio absoluto com {resultado_cenarios.iloc[4, 0]:.2f} [em milhões de R$] podendo ser usado Regressão Linear Simples apenas para predizer o rendimento, sem considerar
o estado ou ano.
    ''')

=======
#------------ MODELO ------------

resultados = treinar_modelos(dfAgregado)
multipla = resultados["multipla"]
simples = resultados["simples"]

with cartao("predito_modelo"):
    st.subheader("Regressão Linear Múltipla")
    st.markdown(
        f'''
O modelo prevê a **{ALVO.replace("$", r"\$")}** de cada estado usando apenas as variáveis do ano anterior.

- **Treino:** anos até {ANO_CORTE} ({multipla["n_treino"]} linhas);
- **Teste:** anos depois de {ANO_CORTE} ({multipla["n_teste"]} linhas);
- A divisão é feita por ano (e não aleatória) para que o modelo nunca veja dados "do futuro" durante o treino;
- Como comparação (baseline), foi treinada uma **regressão linear simples** que usa somente a renda do ano anterior.
'''
    )

with cartao("predito_metricas"):
    st.subheader("Desempenho no conjunto de teste")

    col1, col2, col3 = st.columns(3)
    mt, ms = multipla["metricas_teste"], simples["metricas_teste"]
    with col1:
        st.metric("R²", f"{mt['R²']:.4f}", f"{mt['R²'] - ms['R²']:+.4f} vs. simples")
    with col2:
        st.metric("RMSE [R$ milhões]", f"{mt['RMSE']:,.2f}", f"{mt['RMSE'] - ms['RMSE']:+,.2f} vs. simples", delta_color="inverse")
    with col3:
        st.metric("MAE [R$ milhões]", f"{mt['MAE']:,.2f}", f"{mt['MAE'] - ms['MAE']:+,.2f} vs. simples", delta_color="inverse")

    tabela_metricas = pd.DataFrame({
        "Múltipla - treino": multipla["metricas_treino"],
        "Múltipla - teste": multipla["metricas_teste"],
        "Simples - treino": simples["metricas_treino"],
        "Simples - teste": simples["metricas_teste"],
    })
    st.dataframe(tabela_metricas, width="stretch")

    erro_medio_pct = multipla["previsoes"]["Erro (%)"].abs().mean()
    st.markdown(
        rf'''
- **R²**: proporção da variação da renda explicada pelo modelo (1 = perfeito);
- **RMSE**: erro médio em R\$ milhões, penalizando mais os erros grandes;
- **MAE**: erro absoluto médio em R\$ milhões;
- Erro percentual absoluto médio no teste: **{erro_medio_pct:.2f}%**.

Obs.: como os estados têm tamanhos muito diferentes (SP é muito maior que RR, por exemplo), o R² tende a ficar alto
de qualquer forma; o MAE e o erro percentual mostram melhor o quanto o modelo erra em cada estado.
'''
    )

with cartao("predito_coeficientes"):
    st.subheader("Coeficientes do modelo")
    st.markdown(
        rf'''
Cada coeficiente indica quanto a renda prevista muda (em R\$ milhões) quando a variável aumenta 1 unidade,
mantendo as outras fixas. Intercepto: **{multipla["intercepto"]:,.2f}**.

Obs.: as variáveis do ano anterior são muito correlacionadas entre si (ver mapa de correlação na página "Sobre"),
então os coeficientes individuais devem ser interpretados com cuidado.
'''
    )
    st.dataframe(multipla["coeficientes"], width="stretch", hide_index=True)

with cartao("predito_tabela"):
    st.subheader("Previsões por estado (teste)")
    st.dataframe(
        multipla["previsoes"].sort_values(["Ano-calendário", "Erro (%)"], key=lambda s: s.abs() if s.name == "Erro (%)" else s),
        width="stretch",
        hide_index=True,
    )
>>>>>>> 09ee3c8 (tabelas)
