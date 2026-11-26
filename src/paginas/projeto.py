import html

import requests
import streamlit as st

from src.util.tema import cartao, AZUL_GOV, AZUL_TEXTO, CARTAO_BORDA

CRIADORES = {
    "Felipe Nunes": ("finusz", "finusz"),
    "Heloísa Azevedo": ("heloisa-azevedo", "heloisa-azevedo"),
    "Catharina Gato": ("gatocatharinap", "catharina-gato-b4246537b"),
}

CSS_CRIADORES = f"""
<style>
.avatar {{
    width: 140px;
    height: 140px;
    border-radius: 50%;
    object-fit: cover;
    border: 4px solid {AZUL_GOV};
    display: block;
    margin: 0 auto 0.75rem;
}}
[class*="st-key-cartao_"] .nome {{
    text-align: center;
    font-size: 1.3rem;
    font-weight: 700;
    color: {AZUL_GOV};
    margin: 0;
}}
[class*="st-key-cartao_"] .bio {{
    text-align: center;
    font-size: 0.85rem;
    font-style: italic;
    margin: 0.25rem 0 0;
}}
.chips {{
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 0.5rem 0;
}}
[class*="st-key-cartao_"] .chip {{
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    background: #ffffff;
    border: 1px solid {CARTAO_BORDA};
    color: {AZUL_TEXTO};
    font-size: 0.75rem;
    font-weight: 500;
}}
</style>
"""


#------------ BUSCA DE DADOS (com cache) ------------
@st.cache_data(ttl=3600)
def buscar_dados_perfil(username: str) -> dict:
    '''
    Busca o perfil na API do GitHub.
    Em caso de falha gera uma exceção, e assim o erro NÃO fica guardado no cache.
    '''
    resposta = requests.get(f"https://api.github.com/users/{username}", timeout=5)
    resposta.raise_for_status()
    return resposta.json()


@st.cache_data(ttl=3600)
def buscar_readme(username: str) -> str:
    '''
    Busca o README de perfil (repositório com o mesmo nome do usuário).
    Gera uma exceção se não encontrar em nenhuma das branches.
    '''
    for branch in ("main", "master"):
        url = f"https://raw.githubusercontent.com/{username}/{username}/{branch}/README.md"
        resposta = requests.get(url, timeout=5)
        if resposta.status_code == 200:
            return resposta.text
    raise requests.RequestException(f"README de {username} não encontrado")


#------------ CARTÃO DO CRIADOR ------------
def renderizar_criador(nome_exibicao: str, username: str, username_linkedin: str):
    '''
    Desenha o cartão de um criador.

    nome_exibicao -> nome usado caso o perfil do GitHub não tenha um nome
    username -> usuário no GitHub
    username_linkedin -> usuário no LinkedIn
    '''
    with cartao(f"criador_{username}"):
        try:
            dados = buscar_dados_perfil(username)
        except requests.RequestException:
            dados = {}

        nome = html.escape(dados.get("name") or nome_exibicao)
        bio = html.escape(dados.get("bio") or "")
        avatar = html.escape(dados.get("avatar_url") or f"https://github.com/{username}.png")

        st.markdown(f'<img class="avatar" src="{avatar}" alt="Foto de {nome}">', unsafe_allow_html=True)
        st.markdown(f'<p class="nome">{nome}</p>', unsafe_allow_html=True)
        if bio:
            st.markdown(f'<p class="bio">{bio}</p>', unsafe_allow_html=True)

        chips = [
            f'<span class="chip">{icone} {html.escape(dados[campo])}</span>'
            for icone, campo in (("📍", "location"), ("🏢", "company"))
            if dados.get(campo)
        ]
        if chips:
            st.markdown(f'<div class="chips">{"".join(chips)}</div>', unsafe_allow_html=True)

        st.divider()

        if "public_repos" in dados:
            m1, m2, m3 = st.columns(3)
            m1.metric("Repositórios", dados.get("public_repos", 0))
            m2.metric("Seguidores", dados.get("followers", 0))
            m3.metric("Seguindo", dados.get("following", 0))

        b1, b2 = st.columns(2)
        b1.link_button(
            "GitHub",
            dados.get("html_url") or f"https://github.com/{username}",
            type="primary",
            width="stretch",
        )
        b2.link_button(
            "LinkedIn",
            f"https://www.linkedin.com/in/{username_linkedin}/",
            type="primary",
            width="stretch",
        )

        with st.expander("📄 Ver README do perfil"):
            try:
                st.markdown(buscar_readme(username), unsafe_allow_html=True)
            except requests.RequestException:
                st.info("README não disponível ou não configurado.")


#------------ INTERFACE ------------
st.markdown(CSS_CRIADORES, unsafe_allow_html=True)

with cartao("projeto_equipe"):
    st.subheader("Conheça os responsáveis por este projeto", text_alignment="center")

colunas = st.columns(len(CRIADORES), gap="medium")
for coluna, (nome_exibicao, usernames) in zip(colunas, CRIADORES.items()):
    with coluna:
        renderizar_criador(nome_exibicao, *usernames)