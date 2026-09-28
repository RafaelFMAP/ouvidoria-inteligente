"""
Ouvidoria Inteligente — triagem semântica de manifestações cidadãs.

Executar com:
    streamlit run app_ouvidoria.py

Abas:
    🔎 Busca Semântica  — manifestações mais parecidas com uma descrição livre
    📋 Base Completa    — tabela filtrável, matriz de similaridade e possíveis duplicatas
    🗺️ Espaço Vetorial  — projeção 2D (PCA / t-SNE) colorida pela categoria oficial
    ✂️ Chunking         — divisão de textos longos em chunks e seus embeddings
"""

import html
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from transformers.utils import logging as hf_logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# O app conta tokens de textos maiores que o limite do modelo de propósito (para avisar o usuário)
hf_logging.set_verbosity_error()

# ----------------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------------
MODELOS = {
    "MiniLM multilíngue (384 dim., rápido)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "MPNet multilíngue (768 dim., mais preciso)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
ARQUIVO_PADRAO = "manifestacoes.json"
LIMITE_TEXTO_LONGO = 500  # caracteres: acima disso a manifestação é indexada por chunks
CHUNK_BUSCA = dict(chunk_size=300, chunk_overlap=100)  # melhor configuração da Entrega 3
SEP_FRASE = [". ", "! ", "? ", "; ", ", ", " ", ""]

CORES_CATEGORIA = {
    "infraestrutura": "#1f77b4",
    "saúde": "#d62728",
    "segurança": "#9467bd",
    "educação": "#2ca02c",
    "meio ambiente": "#ff7f0e",
}

EXEMPLOS_BUSCA = [
    "a rua da minha casa está cheia de buracos",
    "não tem médico para atender no posto",
    "barulho de festa de madrugada que não me deixa dormir",
    "meu filho não tem vaga na creche",
    "assalto no ponto de ônibus",
]

st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

st.markdown(
    """
    <style>
    .cartao {border: 1px solid rgba(128,128,128,.35); border-left-width: 6px; border-radius: 8px;
             padding: .75rem 1rem; margin-bottom: .75rem;}
    .cartao.alta {border-left-color: #2e7d32;}
    .cartao.media {border-left-color: #f9a825;}
    .cartao.baixa {border-left-color: #c62828;}
    .cartao .topo {display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; margin-bottom: .35rem;}
    .etiqueta {border-radius: 999px; padding: .05rem .6rem; font-size: .8rem; color: white;}
    .score {font-weight: 700; font-size: 1.05rem;}
    .meta {opacity: .7; font-size: .85rem;}
    mark.trecho {background: rgba(255, 213, 79, .55); padding: 0 .15rem; border-radius: 3px; color: inherit;}
    mark.overlap {background: rgba(100, 181, 246, .45); padding: 0 .1rem; border-radius: 3px; color: inherit;}
    .chunk {border: 1px dashed rgba(128,128,128,.5); border-radius: 6px; padding: .5rem .75rem; margin-bottom: .5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Modelo, dados e embeddings (com cache)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo de embeddings…")
def carregar_modelo(caminho):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(caminho)


@st.cache_data(show_spinner="Gerando embeddings…")
def gerar_embeddings(caminho_modelo, textos):
    """Embeddings normalizados (produto escalar = cosseno). `textos` é uma tupla para ser 'hasheável'."""
    return carregar_modelo(caminho_modelo).encode(list(textos), normalize_embeddings=True)


@st.cache_data
def ler_manifestacoes(conteudo):
    df = pd.DataFrame(json.loads(conteudo))
    obrigatorias = {"id", "data", "categoria_oficial", "texto"}
    faltando = obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(f"campos ausentes no JSON: {', '.join(sorted(faltando))}")
    df["n_caracteres"] = df["texto"].str.len()
    return df


def dividir(texto, estrategia, chunk_size, chunk_overlap):
    """Divide um texto conforme a estratégia escolhida."""
    if estrategia == "Janela fixa de caracteres":
        passo = max(1, chunk_size - chunk_overlap)
        return [texto[i:i + chunk_size] for i in range(0, max(1, len(texto) - chunk_overlap), passo)]
    kwargs = dict(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if estrategia == "Recursivo por frases":
        kwargs.update(separators=SEP_FRASE, keep_separator="end")
    return RecursiveCharacterTextSplitter(**kwargs).split_text(texto)


@st.cache_data
def indice_de_busca(textos):
    """Unidades de busca: textos curtos inteiros e textos longos divididos em chunks."""
    unidades, dono = [], []
    for i, texto in enumerate(textos):
        partes = dividir(texto, "Recursivo por frases", **CHUNK_BUSCA) if len(texto) > LIMITE_TEXTO_LONGO else [texto]
        unidades += partes
        dono += [i] * len(partes)
    return unidades, np.array(dono)


def classificar(score):
    if score > 0.7:
        return "🟢", "alta", "Alta similaridade"
    if score > 0.5:
        return "🟡", "media", "Similaridade moderada"
    return "🔴", "baixa", "Baixa similaridade"


def etiqueta(categoria):
    cor = CORES_CATEGORIA.get(categoria, "#607d8b")
    return f'<span class="etiqueta" style="background:{cor}">{html.escape(categoria)}</span>'


def destacar(texto, trecho):
    """Marca `trecho` dentro de `texto` (quando o resultado veio de um chunk)."""
    t, c = html.escape(texto), html.escape(trecho)
    if trecho and trecho != texto and c in t:
        return t.replace(c, f'<mark class="trecho">{c}</mark>', 1)
    return t


def ordem_por_categoria(df):
    return df.sort_values(["categoria_oficial", "id"]).index.to_numpy()


# ----------------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🏛️ Ouvidoria Inteligente")
    st.caption("Triagem semântica de manifestações cidadãs")

    nome_modelo = st.selectbox("Modelo de embedding", list(MODELOS), index=1,
                               help="Os dois modelos são multilíngues (sentence-transformers). "
                                    "O MPNet separou melhor as duplicatas na Entrega 2.")
    caminho_modelo = MODELOS[nome_modelo]
    top_k = st.slider("Top-k resultados", 1, 15, 5, help="Quantas manifestações mostrar na busca.")
    limiar_dup = st.slider("Limiar de duplicata", 0.50, 0.95, 0.75, 0.01,
                           help="Pares com similaridade acima deste valor são sinalizados como possíveis duplicatas "
                                "(0,75 foi o limiar calibrado para o MPNet na Entrega 2).")
    usar_chunks = st.toggle("Buscar por chunks nos textos longos", value=True,
                            help=f"Manifestações com mais de {LIMITE_TEXTO_LONGO} caracteres são divididas em chunks "
                                 "(300 caracteres, overlap 100, por frases); a nota é a do melhor chunk.")

    st.divider()
    enviado = st.file_uploader("Usar outro arquivo de manifestações (JSON)", type="json",
                               help="Mesmo formato de manifestacoes.json: id, data, categoria_oficial, texto.")

try:
    if enviado is not None:
        df = ler_manifestacoes(enviado.getvalue().decode("utf-8"))
        origem = enviado.name
    else:
        with open(ARQUIVO_PADRAO, encoding="utf-8") as f:
            df = ler_manifestacoes(f.read())
        origem = ARQUIVO_PADRAO
except (OSError, ValueError, json.JSONDecodeError) as erro:
    st.error(f"Não foi possível carregar as manifestações: {erro}")
    st.stop()

textos = tuple(df["texto"])
E = gerar_embeddings(caminho_modelo, textos)
modelo = carregar_modelo(caminho_modelo)

with st.sidebar:
    st.caption(f"📄 {origem}: **{len(df)}** manifestações · {df['categoria_oficial'].nunique()} categorias · "
               f"{(df['n_caracteres'] > LIMITE_TEXTO_LONGO).sum()} textos longos")

aba_busca, aba_base, aba_espaco, aba_chunk = st.tabs(
    ["🔎 Busca Semântica", "📋 Base Completa", "🗺️ Espaço Vetorial", "✂️ Chunking"]
)

# ----------------------------------------------------------------------------
# Aba 1 — Busca semântica
# ----------------------------------------------------------------------------
with aba_busca:
    st.subheader("Descreva o problema com suas palavras")
    st.caption("O sistema procura manifestações com o mesmo **significado**, mesmo que usem palavras diferentes.")

    if "consulta" not in st.session_state:
        st.session_state.consulta = ""
    exemplo = st.pills("Exemplos", EXEMPLOS_BUSCA, selection_mode="single", label_visibility="collapsed")
    if exemplo and exemplo != st.session_state.get("_ultimo_exemplo"):
        st.session_state.consulta = exemplo
    st.session_state._ultimo_exemplo = exemplo

    consulta = st.text_area("Sua manifestação", key="consulta", height=100,
                            placeholder="Ex.: o poste da minha rua está apagado há semanas e está tudo escuro…")

    if consulta.strip():
        eq = modelo.encode([consulta], normalize_embeddings=True)[0]
        if usar_chunks:
            unidades, dono = indice_de_busca(textos)
            Eu = gerar_embeddings(caminho_modelo, tuple(unidades))
            s_unid = Eu @ eq
            notas = np.full(len(df), -1.0)
            melhor_trecho = [""] * len(df)
            for u, (s, i) in enumerate(zip(s_unid, dono)):
                if s > notas[i]:
                    notas[i], melhor_trecho[i] = s, unidades[u]
        else:
            notas = E @ eq
            melhor_trecho = list(textos)

        ordem = np.argsort(-notas)[:top_k]
        c1, c2, c3 = st.columns(3)
        c1.metric("Melhor score", f"{notas[ordem[0]]:.3f}")
        c2.metric("🟢 Alta (> 0,7)", int((notas[ordem] > 0.7).sum()))
        c3.metric("🟡 Moderada (> 0,5)", int(((notas[ordem] > 0.5) & (notas[ordem] <= 0.7)).sum()))

        if notas[ordem[0]] >= limiar_dup:
            st.warning(f"⚠️ Esta descrição é muito parecida com **{df.loc[ordem[0], 'id']}** "
                       f"(similaridade {notas[ordem[0]]:.2f} ≥ limiar {limiar_dup:.2f}). "
                       "Talvez o problema já tenha sido registrado.")

        for rank, i in enumerate(ordem, start=1):
            emoji, classe, rotulo = classificar(notas[i])
            veio_de_chunk = usar_chunks and melhor_trecho[i] != textos[i]
            st.markdown(
                f"""
                <div class="cartao {classe}">
                  <div class="topo">
                    <span class="score">{emoji} {notas[i]:.3f}</span>
                    <b>#{rank} · {html.escape(df.loc[i, 'id'])}</b>
                    {etiqueta(df.loc[i, 'categoria_oficial'])}
                    <span class="meta">{html.escape(str(df.loc[i, 'data']))} · {rotulo}
                    {' · trecho destacado = chunk mais parecido' if veio_de_chunk else ''}</span>
                  </div>
                  <div>{destacar(textos[i], melhor_trecho[i] if veio_de_chunk else '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption("Legenda: 🟢 > 0,7 · 🟡 > 0,5 · 🔴 demais. Scores são similaridade de cosseno entre embeddings.")
    else:
        st.info("Digite uma descrição ou escolha um exemplo acima para buscar.")

# ----------------------------------------------------------------------------
# Aba 2 — Base completa
# ----------------------------------------------------------------------------
with aba_base:
    st.subheader("Todas as manifestações")
    f1, f2 = st.columns([2, 3])
    categorias = sorted(df["categoria_oficial"].unique())
    filtro_cat = f1.multiselect("Categorias", categorias, default=categorias)
    filtro_txt = f2.text_input("Contém o texto", placeholder="filtro por palavra (busca literal)")

    visivel = df[df["categoria_oficial"].isin(filtro_cat)]
    if filtro_txt:
        visivel = visivel[visivel["texto"].str.contains(re.escape(filtro_txt), case=False)]
    st.dataframe(
        visivel[["id", "data", "categoria_oficial", "n_caracteres", "texto"]],
        hide_index=True,
        width="stretch",
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "data": st.column_config.TextColumn("Data", width="small"),
            "categoria_oficial": st.column_config.TextColumn("Categoria", width="small"),
            "n_caracteres": st.column_config.ProgressColumn("Tamanho", format="%d car.", min_value=0,
                                                            max_value=int(df["n_caracteres"].max())),
            "texto": st.column_config.TextColumn("Texto", width="large"),
        },
    )
    st.caption(f"{len(visivel)} de {len(df)} manifestações")

    if st.button("🧮 Gerar matriz de similaridade", type="primary"):
        st.session_state.matriz_modelo = caminho_modelo
    if st.session_state.get("matriz_modelo") == caminho_modelo:
        S = E @ E.T
        ordem = ordem_por_categoria(df)
        rot = [f"{df.loc[i, 'id']} ({df.loc[i, 'categoria_oficial'][:4]}.)" for i in ordem]
        fig = go.Figure(go.Heatmap(
            z=S[np.ix_(ordem, ordem)], x=rot, y=rot, colorscale="Viridis", zmin=0, zmax=1,
            customdata=np.dstack(np.meshgrid([textos[i][:90] for i in ordem], [textos[i][:90] for i in ordem])),
            hovertemplate="%{y} × %{x}<br>cosseno = %{z:.3f}<br><br>%{customdata[1]}…<br>%{customdata[0]}…<extra></extra>",
            colorbar=dict(title="cosseno"),
        ))
        fig.update_layout(height=720, margin=dict(l=10, r=10, t=40, b=10),
                          title=f"Similaridade de cosseno — {nome_modelo.split(' (')[0]} (ordenado por categoria)",
                          yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-60))
        st.plotly_chart(fig, width="stretch")

        i, j = np.triu_indices(len(df), k=1)
        mask = S[i, j] >= limiar_dup
        pares = sorted(zip(S[i, j][mask], i[mask], j[mask]), reverse=True)
        st.markdown(f"#### Possíveis duplicatas (similaridade ≥ {limiar_dup:.2f}): {len(pares)}")
        if pares:
            st.dataframe(pd.DataFrame([{
                "par": f"{df.loc[a, 'id']} × {df.loc[b, 'id']}", "similaridade": round(float(s), 3),
                "texto 1": textos[a], "texto 2": textos[b]} for s, a, b in pares]),
                hide_index=True, width="stretch")
        else:
            st.caption("Nenhum par acima do limiar. Ajuste o limiar na barra lateral.")

# ----------------------------------------------------------------------------
# Aba 3 — Espaço vetorial
# ----------------------------------------------------------------------------
with aba_espaco:
    st.subheader("Mapa semântico das manifestações")
    c1, c2, c3 = st.columns([1, 1, 2])
    metodo = c1.radio("Projeção", ["PCA", "t-SNE"], horizontal=True)
    perplexidade = c2.slider("Perplexidade (t-SNE)", 2, max(3, min(30, len(df) - 1)), min(10, len(df) - 1),
                             disabled=metodo != "t-SNE")
    mostrar_rotulos = c3.toggle("Mostrar IDs no gráfico", value=False)

    @st.cache_data
    def projetar(caminho_modelo, textos, metodo, perplexidade):
        X = gerar_embeddings(caminho_modelo, textos)
        if metodo == "PCA":
            pca = PCA(n_components=2, random_state=42)
            return pca.fit_transform(X), float(pca.explained_variance_ratio_.sum())
        return TSNE(n_components=2, perplexity=perplexidade, init="pca", metric="cosine",
                    random_state=42).fit_transform(X), None

    P, variancia = projetar(caminho_modelo, textos, metodo, perplexidade)
    mapa = df.assign(x=P[:, 0], y=P[:, 1],
                     texto_quebrado=df["texto"].str.wrap(70).str.replace("\n", "<br>"))
    fig = px.scatter(mapa, x="x", y="y", color="categoria_oficial", color_discrete_map=CORES_CATEGORIA,
                     text="id" if mostrar_rotulos else None,
                     hover_data={"id": True, "texto_quebrado": True, "x": False, "y": False},
                     labels={"categoria_oficial": "Categoria oficial", "texto_quebrado": "Texto"})
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")), textposition="top center")
    fig.update_layout(height=560, xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, width="stretch")
    if variancia is not None:
        st.caption(f"PCA: os 2 componentes explicam {variancia:.0%} da variância dos embeddings — "
                   "boa parte da estrutura fica escondida em 2D; o t-SNE preserva melhor as vizinhanças locais.")

    # Métricas que dependem só dos embeddings originais (não da projeção)
    Sn = E @ E.T
    np.fill_diagonal(Sn, -1)
    cat = df["categoria_oficial"].to_numpy()
    vizinho_mesma = cat[Sn.argmax(axis=1)] == cat
    silhueta = silhouette_samples(E, cat, metric="cosine")
    resumo = (pd.DataFrame({"categoria": cat, "vizinho da mesma categoria": vizinho_mesma, "silhueta": silhueta})
              .groupby("categoria").agg(manifestações=("silhueta", "size"),
                                        **{"% vizinho mais próximo da mesma categoria": ("vizinho da mesma categoria", "mean"),
                                           "silhueta média": ("silhueta", "mean")}))
    resumo["% vizinho mais próximo da mesma categoria"] = (100 * resumo["% vizinho mais próximo da mesma categoria"]).round(0)
    st.markdown("#### Os clusters semânticos coincidem com as categorias?")
    m1, m2 = st.columns([1, 2])
    m1.metric("Vizinho mais próximo na mesma categoria", f"{vizinho_mesma.mean():.0%}")
    m1.metric("Silhueta por categoria (−1 a 1)", f"{silhueta.mean():.3f}")
    m2.dataframe(resumo.round(3), width="stretch")

    with st.expander("💬 Análise", expanded=True):
        st.markdown(
            """
**Em parte.** Com o dataset de exemplo, *saúde* e *educação* formam grupos bem definidos: quase todas as
manifestações dessas categorias têm como vizinho mais próximo outra da mesma categoria, e elas aparecem
agrupadas nas duas projeções. Já **infraestrutura, meio ambiente e segurança se sobrepõem** em uma grande
região "urbana" do mapa:

* **Segurança é a categoria menos coesa** (57% de vizinhos da mesma categoria no MPNet e 29% no MiniLM). Várias
  reclamações de segurança falam de coisas físicas da rua: *"Praça da Matriz sem policiamento à noite"* fica perto de
  *"rua escura, poste apagado"* (infraestrutura), e *"carros em alta velocidade"* fica perto de *"buraco na Av. Brasil"*.
  Para o cidadão, iluminação e segurança são o mesmo problema; para a prefeitura, são secretarias diferentes.
* **Infraestrutura × meio ambiente** se misturam em temas como lixo e água: *"coleta de lixo não passou"* (rotulada como
  infraestrutura) tem como vizinha *"entulho no terreno baldio"* (meio ambiente), e o alagamento (M006) fica ao lado do
  lixão clandestino (M036).
* A silhueta média baixa (≈ 0,1) confirma que as categorias oficiais **não** são clusters bem separados no espaço dos
  embeddings: os embeddings agrupam pelo *assunto concreto* (rua, lixo, escola, posto), e as categorias
  refletem a *secretaria responsável*.

**Consequência para a triagem:** os embeddings são ótimos para achar manifestações parecidas e duplicatas, mas
para rotear a manifestação para a secretaria certa, um classificador supervisionado treinado com as categorias
(por exemplo, regressão logística sobre os embeddings) tende a funcionar melhor do que só agrupar por proximidade.
Casos de fronteira, como iluminação × segurança, podem ir para as duas secretarias.

*Os percentuais citados foram medidos com o dataset de exemplo; a tabela acima é recalculada para o modelo e os dados carregados.*
            """
        )

# ----------------------------------------------------------------------------
# Aba 4 — Chunking
# ----------------------------------------------------------------------------
with aba_chunk:
    st.subheader("Divisão de manifestações longas em chunks")
    longas = df[df["n_caracteres"] > LIMITE_TEXTO_LONGO].sort_values("n_caracteres", ascending=False)
    opcoes = ["✍️ Colar meu próprio texto"] + [f"{r.id} · {r.categoria_oficial} ({r.n_caracteres} car.)"
                                                 for r in longas.itertuples()]
    escolha = st.selectbox("Texto de partida", opcoes, index=1 if len(opcoes) > 1 else 0)
    inicial = "" if escolha == opcoes[0] else longas.loc[longas["id"] == escolha.split(" · ")[0], "texto"].iloc[0]
    texto_longo = st.text_area("Manifestação longa", value=inicial, height=160, key=f"texto_chunk_{escolha}")

    p1, p2, p3 = st.columns(3)
    estrategia = p1.selectbox(
        "Estratégia", ["Recursivo por frases", "RecursiveCharacterTextSplitter (padrão)", "Janela fixa de caracteres"],
        help="• Recursivo por frases: RecursiveCharacterTextSplitter com separadores de pontuação.\n"
             "• Padrão: separadores \\n\\n, \\n, espaço — em texto sem quebras de linha, corta no meio das frases.\n"
             "• Janela fixa: corta a cada N caracteres, sem respeitar nem palavras.")
    chunk_size = p2.slider("chunk_size (caracteres)", 50, 800, 300, 10)
    chunk_overlap = p3.slider("chunk_overlap (caracteres)", 0, chunk_size // 2, min(100, chunk_size // 2), 10)

    if texto_longo.strip():
        partes = dividir(texto_longo.strip(), estrategia, chunk_size, chunk_overlap)
        Ec = gerar_embeddings(caminho_modelo, tuple(partes))
        e_doc = gerar_embeddings(caminho_modelo, (texto_longo.strip(),))[0]
        n_tokens = [len(modelo.tokenizer(p)["input_ids"]) for p in partes]
        tokens_doc = len(modelo.tokenizer(texto_longo.strip())["input_ids"])
        consecutivos = [float(Ec[k] @ Ec[k + 1]) for k in range(len(partes) - 1)]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Chunks", len(partes))
        k2.metric("Tamanho médio", f"{np.mean([len(p) for p in partes]):.0f} car.")
        k3.metric("Coesão entre consecutivos", f"{np.mean(consecutivos):.3f}" if consecutivos else "—",
                  help="Similaridade de cosseno média entre o chunk k e o chunk k+1.")
        k4.metric("Fidelidade ao texto inteiro", f"{float(np.mean(Ec @ e_doc)):.3f}",
                  help="Similaridade média entre cada chunk e o embedding do texto completo.")
        if tokens_doc > modelo.max_seq_length:
            st.info(f"ℹ️ O texto inteiro tem **{tokens_doc} tokens**, mas o modelo lê no máximo "
                    f"**{modelo.max_seq_length}**: sem chunking, o final do texto seria ignorado.")

        col_chunks, col_graficos = st.columns([1, 1])
        with col_chunks:
            st.markdown("##### Chunks gerados")
            st.caption("🟦 trecho repetido do chunk anterior (overlap)")
            anterior = ""
            for k, parte in enumerate(partes):
                # maior sufixo do chunk anterior que é prefixo deste chunk = overlap
                sobreposto = next((n for n in range(min(len(anterior), len(parte)), 0, -1)
                                   if anterior.endswith(parte[:n])), 0)
                corpo = html.escape(parte[sobreposto:])
                if sobreposto:
                    corpo = f'<mark class="overlap">{html.escape(parte[:sobreposto])}</mark>{corpo}'
                alerta = " ⚠️ excede o limite do modelo" if n_tokens[k] > modelo.max_seq_length else ""
                st.markdown(f'<div class="chunk"><span class="meta">chunk {k + 1} · {len(parte)} car. · '
                            f"{n_tokens[k]} tokens{alerta}</span><br>{corpo}</div>", unsafe_allow_html=True)
                anterior = parte

        with col_graficos:
            st.markdown("##### Similaridade entre chunks")
            nomes = [f"c{k + 1}" for k in range(len(partes))]
            fig = px.imshow(Ec @ Ec.T, x=nomes, y=nomes, zmin=0, zmax=1, color_continuous_scale="Viridis",
                            text_auto=".2f")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

            st.markdown("##### Chunks no espaço das manifestações (PCA)")
            pca = PCA(n_components=2, random_state=42).fit(E)
            Pb, Pc = pca.transform(E), pca.transform(Ec)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=Pb[:, 0], y=Pb[:, 1], mode="markers", name="manifestações da base",
                                     marker=dict(color="lightgray", size=8), text=df["id"],
                                     hovertemplate="%{text}<extra></extra>"))
            fig.add_trace(go.Scatter(x=Pc[:, 0], y=Pc[:, 1], mode="markers+lines+text", name="chunks",
                                     text=nomes, textposition="top center",
                                     marker=dict(size=12, color=list(range(len(partes))), colorscale="Plasma"),
                                     line=dict(width=1, dash="dot"), hovertext=[p[:80] for p in partes],
                                     hovertemplate="%{text}: %{hovertext}…<extra></extra>"))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig, width="stretch")

        with st.expander(f"🔢 Ver os embeddings ({Ec.shape[1]} dimensões por chunk)"):
            n_dim = st.slider("Dimensões exibidas", 5, min(64, Ec.shape[1]), 16)
            st.dataframe(pd.DataFrame(Ec[:, :n_dim], index=nomes,
                                      columns=[f"d{d}" for d in range(n_dim)]).style.background_gradient(
                cmap="RdBu_r", axis=None, vmin=-0.15, vmax=0.15).format("{:.3f}"), width="stretch")
            st.caption("Cada linha é o vetor de um chunk (normalizado). Cores: azul = negativo, vermelho = positivo.")
    else:
        st.info("Cole um texto ou escolha uma manifestação longa para ver os chunks.")
