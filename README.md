# Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs

Protótipo de triagem semântica para a Ouvidoria de um município: compara representações esparsas (BoW, TF-IDF) e densas (embeddings), detecta manifestações duplicadas, trata textos longos com *chunking* e oferece um buscador semântico em Streamlit.

## Entregas

| Arquivo | Conteúdo |
|---|---|
| [`análise_comparativa.ipynb`](análise_comparativa.ipynb) | **Entrega 1** — BoW × TF-IDF × embeddings (2 modelos) nos pares M003×M017, M008×M022 e M008×M031 |
| [`deteccao_duplicatas.ipynb`](deteccao_duplicatas.ipynb) | **Entrega 2** — `detectar_duplicatas(textos, limiar=0.85)`, heatmap, calibração do limiar e análise de falsos positivos/negativos |
| [`chunking_manifestacoes.ipynb`](chunking_manifestacoes.ipynb) | **Entrega 3** — `RecursiveCharacterTextSplitter` com 5 configurações, coesão, teste de busca e visualização PCA/t-SNE |
| [`app_ouvidoria.py`](app_ouvidoria.py) | **Entrega 4** — app Streamlit (Busca Semântica, Base Completa, Espaço Vetorial, Chunking) |
| [`RELATORIO.pdf`](RELATORIO.pdf) | Síntese de decisões, dificuldades e aprendizados |
| [`manifestacoes.json`](manifestacoes.json) | Base com 40 manifestações |
| [`gabarito_duplicatas.json`](gabarito_duplicatas.json) | Pares de duplicatas reais, usados para avaliar a Entrega 2 |
| [`figuras/`](figuras) | Gráficos gerados pelos notebooks |

> **Sobre os dados:** `manifestacoes.json` é um dataset de exemplo construído segundo a especificação do desafio (40 manifestações, 5 categorias, 6 duplicatas semânticas = 15%, 5 textos longos). Notebooks e app leem o arquivo JSON, então basta substituí-lo pelo arquivo oficial (e atualizar `gabarito_duplicatas.json`) para reexecutar tudo. O app também permite enviar outro JSON pela barra lateral.

## Como executar

Requer Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Notebooks:** `jupyter notebook` e executar todas as células (as saídas já estão salvas nos arquivos).

**App:**

```bash
streamlit run app_ouvidoria.py
```

Na primeira execução, os modelos `paraphrase-multilingual-MiniLM-L12-v2` (~0,5 GB) e `paraphrase-multilingual-mpnet-base-v2` (~1,1 GB) são baixados do Hugging Face.

## Principais resultados

- **Esparso × denso:** nas frases curtas do enunciado (sem nomes de lugar em comum), BoW e TF-IDF dão similaridade **0,00** para "buraco enorme na Av. Brasil" × "asfalto todo esburacado da avenida principal"; o mpnet dá **0,59** (contra 0,17 do par sem relação).
- **Duplicatas:** o limiar fixo de 0,85 encontra só 2 das 6 duplicatas; o percentil 90 gera 72 falsos positivos. O limiar calibrado de **0,75 com o mpnet** acerta 5/6 sem nenhum falso positivo.
- **Chunking:** os modelos leem no máximo 128 tokens e as manifestações longas têm 159–185, então o final (onde ficam os pedidos) é descartado sem aviso. Com chunks de 300 caracteres, overlap 100 e separadores por frase, nenhuma frase é cortada e 5/5 consultas de teste encontram a manifestação correta em 1º lugar (3/5 sem chunking).
