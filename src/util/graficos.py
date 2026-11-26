import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
#------------ COR PRO GRÁFICO ------------ 
COR_VERMELHO = "#E52207"
COR_AMARELO = "#FFCD07"
COR_VERDE = "#168821"
COR_VERDE_ESCURO = "#0C4A2F"
COR_AZUL = "#1351B4"


def _pintar_ranking(ax):
    """Pinta as barras: 1º vermelho, 2º amarelo, 3º verde, 4º e 5º verde escuro, demais azul."""
    cores_top = [COR_VERMELHO, COR_AMARELO, COR_VERDE, COR_VERDE_ESCURO, COR_VERDE_ESCURO]
    for i, barra in enumerate(ax.containers[0].patches):
        barra.set_facecolor(cores_top[i] if i < len(cores_top) else COR_AZUL)

#------------ GRÁFICOS ------------ 
def corr_dados(data:pd.DataFrame):
    '''
    Recebe um dataframe e cria um gráfico de correlação
    '''

    corr_matrix = data.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap="coolwarm",
                linewidths=0.5,
                cbar_kws={"shrink": 1},
                annot_kws={"size": 10},
                vmin=-1,
                vmax=1)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig

def _quebrar_texto(texto: str, largura: int = 18) -> str:
    palavras = texto.split()
    linhas = []
    atual = ""
    for palavra in palavras:
        if len(atual) + len(palavra) + (1 if atual else 0) <= largura:
            atual = f"{atual} {palavra}".strip()
        else:
            if atual:
                linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return "\n".join(linhas)


def disp_dados(data: pd.DataFrame, max_colunas: int = 6):
    """Recebe um DataFrame e retorna um PairGrid bem espaçado das colunas numéricas."""
    data_filtrado = (
        data.select_dtypes(include="number")
        .drop(columns=["Ano-calendário", "Centil"], errors="ignore")
        .astype(float)
    )

    colunas = list(data_filtrado.columns[:max_colunas])
    data_plot = data_filtrado[colunas]

    sns.set_theme(style="whitegrid", context="notebook")

    grid = sns.pairplot(
        data_plot,
        corner=True,
        plot_kws=dict(
            alpha=0.45,
            s=18,
            edgecolor="none",
        ),
        diag_kws=dict(
            bins=20,
            alpha=0.75,
        ),
        height=2.2,
        aspect=1.15,
    )

    for ax in grid.axes.flatten():
        if ax is None:
            continue

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        if xlabel:
            ax.set_xlabel(_quebrar_texto(xlabel, largura=16), fontsize=8, labelpad=8)
        if ylabel:
            ax.set_ylabel(_quebrar_texto(ylabel, largura=16), fontsize=8, labelpad=10)

        ax.tick_params(axis="x", labelsize=7.5, rotation=30)
        ax.tick_params(axis="y", labelsize=7.5, rotation=0)

    grid.fig.subplots_adjust(
        left=0.14,
        bottom=0.14,
        right=0.98,
        top=0.90,
        wspace=0.28,
        hspace=0.28,
    )

    return grid.fig

def bar_lateral_dados(
    data: pd.DataFrame,
    eixo_x = None,
    eixo_y = None):
    
    """Retorna um gráfico de barras horizontal usando Seaborn e Matplotlib.
        data: DataFrame contendo os dados para plotagem.
        eixo_x: Nome da coluna para o eixo X (normalmente valores numéricos).
        eixo_y: Nome da coluna para o eixo Y (normalmente categorias).
    """

    data_plot = (
        data.groupby(eixo_y, observed=False)[eixo_x]
        .sum()
        .dropna()
        .sort_values(ascending=False)
        .reset_index()
    )

    top5 = (
        data.groupby(eixo_y, observed=False)[eixo_x]
        .sum()
        .dropna()
        .sort_values(ascending=False)
        .head(5)
    )

    lista_top5 = "\n".join(
    f"{i}. **{ente}**: R$ {valor:,.2f}"
    for i, (ente, valor) in enumerate(top5.items(), start=1)
    )

    sns.set_style("white")    
    fig, ax = plt.subplots(figsize=(10, 8))

    ax = sns.barplot(data=data_plot,
                x=eixo_x,
                y=eixo_y,
                orient="h",
                errorbar=None)

    sns.despine(ax=ax)
    ax.bar_label(ax.containers[0], fontsize=10)
    _pintar_ranking(ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig, lista_top5

def bplot_dados(data: pd.DataFrame, eixo_x=None, eixo_y=None, agrupar_centis=True, log=True):
    '''Recebe um DataFrame e retorna um boxplot.

    Se eixo_x == 'Centil' e agrupar_centis=True, os 100 centis são agrupados em 10 faixas (decis).
    log=True usa escala logarítmica no eixo Y (valores <= 0 são descartados).
    '''
    data = data.copy()

    if eixo_x == 'Centil' and agrupar_centis:
        faixas = list(range(0, 101, 10))
        rotulos = [f"{a + 1}–{b}" for a, b in zip(faixas[:-1], faixas[1:])]
        data['Faixa de centil'] = pd.cut(data['Centil'], bins=faixas, labels=rotulos)
        eixo_x = 'Faixa de centil'

    if log:
        data = data[data[eixo_y] > 0]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.boxplot(
        data=data,
        x=eixo_x,
        y=eixo_y,
        ax=ax,
        linewidth=1,
        width=0.6,
        fliersize=2,
        flierprops={"alpha": 0.35},
        color=COR_AZUL
    )

    if log:
        ax.set_yscale("log")
        ax.set_ylabel(f"{eixo_y} (escala log)")

    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()

    return fig

def bar_dados_lista(data: pd.DataFrame, colunas: list[str], coluna_nome: str):

    data_longo = data.melt(
        value_vars=colunas,
        var_name=coluna_nome,
        value_name="Valor [R$ trilhões]",
    )

    data_longo["Valor [R$ trilhões]"] = data_longo["Valor [R$ trilhões]"] / 1000000

    data_longo[coluna_nome] = (
        data_longo[coluna_nome]
        .str.replace(coluna_nome + " - ", "", regex=False)
        .str.replace(" [R$ milhões]", "", regex=False)
        .str.replace(" [R$ trilhões]", "", regex=False)
    )

    soma_agrupada = (
        data_longo.groupby(coluna_nome, observed=False)["Valor [R$ trilhões]"]
        .sum()
        .dropna()
        .sort_values(ascending=False)
    )

    ordem = soma_agrupada.index

    top5 = soma_agrupada.head(5)

    lista_top5 = "\n".join(
        f"{i}. **{ente}**: R$ {valor:,.2f} tri"
        for i, (ente, valor) in enumerate(top5.items(), start=1)
    )

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(
        data=data_longo,
        x="Valor [R$ trilhões]",
        y=coluna_nome,
        order=ordem,
        estimator="sum",
        errorbar=None,
        ax=ax,
    )

    _pintar_ranking(ax)
    sns.despine(ax=ax)
    ax.bar_label(ax.containers[0], fmt='%.2f',fontsize=12, padding=8)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()

    return fig, lista_top5
#grafico realxprevisto
def real_previsto(previsoes: pd.DataFrame, coluna_real: str, coluna_previsto: str = "Previsto [R$ milhões]",
                  coluna_ano: str = "Ano-calendário"):
    '''
    Gráfico de dispersão Real x Previsto (conjunto de teste).
    Cada ponto é um estado em um ano; a linha tracejada é a previsão perfeita (y = x).
    Escala logarítmica nos dois eixos para que estados pequenos e SP fiquem visíveis.
    '''
    cores = [COR_AZUL, COR_VERDE, COR_VERMELHO, COR_AMARELO, COR_VERDE_ESCURO]

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (ano, grupo) in enumerate(previsoes.groupby(coluna_ano)):
        ax.scatter(grupo[coluna_real], grupo[coluna_previsto], s=45, alpha=0.8,
                   color=cores[i % len(cores)], edgecolor="white", linewidth=0.6, label=str(ano))

    minimo = min(previsoes[coluna_real].min(), previsoes[coluna_previsto].min()) * 0.8
    maximo = max(previsoes[coluna_real].max(), previsoes[coluna_previsto].max()) * 1.2
    ax.plot([minimo, maximo], [minimo, maximo], linestyle="--", color="gray", linewidth=1.2,
            label="Previsão perfeita")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(minimo, maximo)
    ax.set_ylim(minimo, maximo)
    ax.set_aspect("equal")

    ax.set_xlabel(f"Real - {coluna_real} (escala log)")
    ax.set_ylabel(f"Previsto - {coluna_previsto} (escala log)")
    ax.set_title("Real x Previsto (teste)")
    ax.grid(True, which="major", linestyle=":", alpha=0.5)
    ax.legend(title="Ano")

    fig.tight_layout()
    return fig

def grafico_previsao(historico: pd.DataFrame, ano_prev: int, valor_prev: float,
                     estado: str, faixa_pct: float | None = None,
                     alvo: str = "Renda Total [R$ milhões]"):
    """Linha com a renda real do estado + ponto da previsão do ano seguinte.

    faixa_pct -> erro percentual médio do teste; se informado, desenha uma barra de erro.
    """
    historico = historico.sort_values("Ano-calendário")
    anos = historico["Ano-calendário"].astype(int)
    real = historico[alvo]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(anos, real, marker="o", color=COR_AZUL, label="Real")

    ax.plot([anos.iloc[-1], ano_prev], [real.iloc[-1], valor_prev],
            linestyle="--", color=COR_VERMELHO)
    ax.scatter([ano_prev], [valor_prev], color=COR_VERMELHO, s=80, zorder=3,
               label=f"Previsto {ano_prev}")

    if faixa_pct:
        ax.errorbar([ano_prev], [valor_prev], yerr=valor_prev * faixa_pct / 100,
                    fmt="none", ecolor=COR_VERMELHO, capsize=6,
                    label=f"± erro médio do teste ({faixa_pct:.1f}%)")

    ax.annotate(f"R$ {valor_prev:,.0f} mi", (ano_prev, valor_prev),
                textcoords="offset points", xytext=(-10, 10), ha="right")

    ax.set_title(f"{estado}: renda total real e prevista")
    ax.set_xlabel("Ano-calendário")
    ax.set_ylabel("Renda Total [R$ milhões]")
    ax.set_xticks(list(anos) + [ano_prev])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig