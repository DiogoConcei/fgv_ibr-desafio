# Semantic News Search

Motor de busca semântica sobre um corpus de notícias econômicas brasileiras, construído em Python com pipeline de sanitização de texto e busca por similaridade vetorial via Sentence Transformers.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Etapa 1 — Sanitização](#etapa-1--sanitização)
- [Etapa 2 — Embeddings e Busca Semântica](#etapa-2--embeddings-e-busca-semântica)
- [Resultados](#resultados)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Executar](#como-executar)
- [Dependências](#dependências)

---

## Visão Geral

O projeto recebe um corpus de notícias em formato JSON com texto bruto contendo marcação HTML, entidades HTML, metadados de publicação e outros ruídos. A solução é dividida em duas etapas principais:

1. **Sanitização** — limpeza e normalização do texto bruto, com persistência local do resultado para evitar reprocessamento.
2. **Busca semântica** — geração de embeddings vetoriais sobre o corpus limpo e recuperação dos artigos mais relevantes para uma consulta em texto livre.

---

## Arquitetura

```
noticias_brutas.json
        │
        ▼
┌─────────────────────┐
│   NewsDataManager   │  Sanitização (BeautifulSoup + Regex)
└─────────────────────┘
        │
        ▼
noticias_limpas.json  ──── cache local, evita reprocessamento
        │
        ▼
┌─────────────────────┐
│  SentenceTransformer│  Geração de embeddings do corpus
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Busca semântica  │  encode_query → similaridade cosseno → top 3
└─────────────────────┘
        │
        ▼
      Usuário
```

---

## Etapa 1 — Sanitização

Toda a lógica de limpeza está encapsulada na classe `NewsDataManager` (`news_data_manager.py`).

### Por que Regex + BeautifulSoup?

A escolha de Regex para a sanitização de texto veio da familiaridade prévia com a ferramenta — já utilizada em outro projeto para sanitizar nomes e caminhos de arquivo e localizar capas de documentos. Além disso, uma pesquisa rápida confirmou que o motor de Regex do Python opera em **O(n)**, tornando-o significativamente mais eficiente do que a abordagem inicial considerada, que consistia em varrer o texto caractere por caractere em múltiplas passagens — uma solução O(n²) que se tornaria custosa para corpora maiores.

O **BeautifulSoup** foi incorporado após pesquisa no GeeksForGeeks como solução para a remoção de tags HTML. Ele se mostrou especialmente valioso por resolver duas camadas do problema de uma vez: remove as tags HTML e decodifica automaticamente as entidades HTML (`&amp;`, `&nbsp;`, `&eacute;`, `&ccedil;`, entre outras) durante o parsing, sem adicionar complexidade algorítmica relevante ao pipeline. O custo total da sanitização permanece em torno de **O(k · n)**, onde `k` é o número fixo de etapas de limpeza e `n` é o tamanho do texto — na prática tratado como **O(n)**.

### O que é limpo

O pipeline de sanitização aplica as seguintes transformações, nesta ordem:

| Etapa                  | O que remove/normaliza                         |
| ---------------------- | ---------------------------------------------- |
| BeautifulSoup          | Tags HTML, entidades HTML                      |
| `replace("\xa0", " ")` | Non-breaking spaces (`&nbsp;`)                 |
| `METADATA_PATTERNS`    | Timestamps e metadados de publicação           |
| Regex de espaços       | Espaços múltiplos consecutivos                 |
| Regex de quebras       | Quebras de linha múltiplas consecutivas        |
| Strip por linha        | Linhas vazias remanescentes após remoções      |
| Filtro mínimo          | Artigos com menos de 20 tokens são descartados |

### Decisão de design: o que não é removido

Referências, fontes e links presentes no corpo do texto são **preservados intencionalmente**. Trechos como `"Leia o comunicado completo em www.bcb.gov.br"` ou `"Fonte: MDIC — www.gov.br/mdic"` são parte legítima do conteúdo jornalístico e contribuem para a integridade semântica do artigo. Removê-los seria tratar como ruído informação que não é ruído.

### Cache local

Ao instanciar `NewsDataManager`, a classe verifica se o arquivo `noticias_limpas.json` já existe. Se existir, carrega diretamente sem reprocessar. Caso contrário, executa a limpeza completa e persiste o resultado. Isso garante que o custo de sanitização seja pago apenas uma vez.

```python
manager = NewsDataManager("dados/noticias_brutas.json")
# → "Dados já limpos encontrados, carregando..."  (nas execuções seguintes)
# → "Executando limpeza... Aprovados: 19 | Descartados: [18]"  (primeira execução)
```

---

## Etapa 2 — Embeddings e Busca Semântica

### Escolha do modelo

Três modelos foram avaliados:

| Modelo                                  | Tipo                       | Característica              |
| --------------------------------------- | -------------------------- | --------------------------- |
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual (50+ línguas) | Leve e rápido               |
| `paraphrase-multilingual-mpnet-base-v2` | Multilingual (50+ línguas) | Maior precisão, mais pesado |
| `all-mpnet-base-v2`                     | Inglês                     | Referência em inglês        |

A busca por um modelo exclusivamente em português não foi frutífera — a biblioteca Sentence Transformers não oferece modelos oficiais focados em PT-BR para similaridade semântica. Os modelos multilingual, treinados com dados paralelos em 50+ idiomas incluindo português, são a opção recomendada para este caso.

Embora modelos em inglês dominem os benchmarks de qualidade — reflexo natural do ecossistema de IA concentrado nos EUA e China —, o modelo selecionado foi o **`paraphrase-multilingual-MiniLM-L12-v2`**. Inicialmente escolhido apenas por ser o mais leve, ele surpreendentemente superou o `mpnet` nos testes com o corpus em português, produzindo resultados mais coerentes para as queries avaliadas.

### Como a busca funciona

O texto de cada artigo é indexado como a concatenação do título com o corpo: `"{titulo}. {texto}"`. Essa decisão é relevante porque o título frequentemente concentra as palavras-chave mais discriminativas do artigo, enquanto o corpo fornece contexto semântico mais rico. A combinação dos dois tende a produzir embeddings mais representativos do que usar apenas um deles.

Os embeddings do corpus são gerados **uma única vez** na inicialização, antes do loop de busca. A cada consulta, apenas o embedding da query é computado, e a similaridade cosseno é calculada entre ela e todos os documentos do corpus. Os três artigos com maior score são retornados como resultado.

```
query → encode_query() → vetor de 384 dimensões
corpus → encode_document() → matriz (19, 384)
similarity(query, corpus) → tensor de scores → top 3
```

### Por que similaridade cosseno?

A similaridade cosseno mede o ângulo entre dois vetores no espaço de embeddings, ignorando a magnitude. Isso é adequado para comparação de textos de tamanhos diferentes — um artigo longo não é artificialmente favorecido por ter mais tokens.

---

## Resultados

Os testes foram realizados com três queries sobre o corpus de 19 artigos aprovados. O score indica a similaridade cosseno entre a query e o artigo (0 a 1).

**Query: "mudanças na taxa de juros"**

```
[1] (0.6207) Copom mantém Selic em 13,75% ao ano pela quarta reunião consecutiva
[2] (0.5976) Selic deve recuar a 9% até o fim de 2024, projetam economistas
[3] (0.5941) Crédito total no Brasil atinge R$ 5,6 trilhões com desaceleração no crescimento
```

**Query: "mercado de trabalho e desemprego"**

```
[1] (0.6628) Desemprego juvenil no Brasil ainda preocupa apesar de melhora geral
[2] (0.6180) Taxa de desemprego cai para 7,9% no segundo trimestre, menor nível desde 2014
[3] (0.4595) Setor de serviços cresce 0,6% em junho e supera expectativas
```

**Query: "inflação e preços ao consumidor"**

```
[1] (0.6191) Inflação ao produtor (IPA) desacelera e pressão sobre preços finais diminui
[2] (0.5883) IGP-M registra terceira deflação consecutiva em agosto
[3] (0.5812) Selic deve recuar a 9% até o fim de 2024, projetam economistas
```

Os resultados são consistentes. O primeiro colocado em cada query é diretamente relevante ao tema buscado, o que é o comportamento esperado — na prática, é o resultado que chega primeiro ao usuário. O terceiro colocado apresenta maior variação, o que é natural: artigos que tratam de temas adjacentes (como juros e inflação) tendem a ter scores próximos e podem aparecer em queries relacionadas.

Vale notar que a análise semântica captura relações que uma busca por palavras-chave perderia. Para "mercado de trabalho e desemprego", o terceiro resultado é o artigo sobre o setor de serviços — que não menciona desemprego diretamente no título, mas trata de emprego e consumo das famílias no corpo do texto, o que o modelo identifica como semanticamente relacionado.

---

## Estrutura do Projeto

```
.
├── dados/
│   ├── noticias_brutas.json      # corpus original (entrada)
│   └── noticias_limpas.json      # corpus sanitizado (gerado automaticamente)
├── modelos/                       # cache dos modelos baixados
├── news_data_manager.py           # sanitização e gerenciamento dos dados
├── main.py                        # motor de busca e interface CLI
├── requirements.txt
└── README.md
```

---

## Como Executar

### Pré-requisitos

- Python 3.10+
- pip

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/semantic-news-search.git
cd semantic-news-search

# Instale as dependências
pip install -r requirements.txt
```

### Execução

```bash
python main.py
```

Na **primeira execução**, o sistema irá:

1. Baixar e armazenar o modelo em `modelos/` (ocorre uma única vez)
2. Sanitizar o corpus e salvar `dados/noticias_limpas.json` (ocorre uma única vez)
3. Gerar os embeddings do corpus
4. Iniciar o loop de busca

Nas execuções seguintes, as etapas 1 e 2 são puladas — o modelo é carregado do cache local e os dados limpos são lidos diretamente do arquivo.

### Uso

```
Busca (ou 'sair' para sair): taxa de juros selic

Top 3 resultados:
  [1] (0.6207) Copom mantém Selic em 13,75% ao ano pela quarta reunião consecutiva
  [2] (0.5976) Selic deve recuar a 9% até o fim de 2024, projetam economistas
  [3] (0.5941) Crédito total no Brasil atinge R$ 5,6 trilhões com desaceleração no crescimento

Escolha um artigo (1-3), nova busca (b) ou sair (s): 1

==================================================
Título: Copom mantém Selic em 13,75% ao ano pela quarta reunião consecutiva
Data: 2023-08-02 | Fonte: Banco Central do Brasil
==================================================
O Comitê de Política Monetária (Copom) do Banco Central do Brasil decidiu...
==================================================

Voltar para (b) busca, (r) resultados ou (s) sair:
```

---

## Dependências

```
beautifulsoup4
sentence-transformers
```
