from pathlib import Path

import streamlit as st

from src.util.tema import aplicar_tema

ICONE_PAGINA = Path(__file__).parent / "assets" / "brasao.png"

st.set_page_config(layout="wide", page_icon=str(ICONE_PAGINA))

sobre_page = st.Page("src/paginas/sobre.py", title="Sobre")
predito_page = st.Page("src/paginas/predito.py", title="Predito")
arrecadacao_page = st.Page("src/paginas/arrecadacao.py", title="Arrecadação 2021")
projeto_page = st.Page("src/paginas/projeto.py", title="Sobre o projeto")

pg = st.navigation([sobre_page, predito_page, arrecadacao_page, projeto_page])

aplicar_tema(pg.title)

pg.run()
#python -m streamlit run app.py
#python -m src.dados.extracaoDados
#