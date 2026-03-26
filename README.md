# Laboratório 6 – Tokenizador BPE e WordPiece

Implementação do algoritmo **Byte Pair Encoding (BPE)** do zero e exploração do tokenizador **WordPiece** do BERT multilíngue via Hugging Face Transformers.

## Estrutura do projeto

```text
bpe-wordpiece-tokenizer/
├── bpe/
│   ├── __init__.py
│   ├── corpus.py    # corpus de treinamento
│   ├── stats.py     # get_stats(): motor de frequências
│   ├── merger.py    # merge_vocab(): passo de fusão
│   └── trainer.py   # loop de treinamento BPE
├── wordpiece/
│   ├── __init__.py
│   └── tokenizer.py # integração com bert-base-multilingual-cased
├── main.py          # ponto de entrada
├── requirements.txt
└── README.md
```

## Como rodar

### Pré-requisitos

- Python 3.10 ou superior
- pip

### 1. Clonar o repositório

```bash
git clone https://github.com/lucasrbsouza/bpe-wordpiece-tokenizer.git
cd bpe-wordpiece-tokenizer
```

### 2. Criar e ativar um ambiente virtual

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

> A instalação do PyTorch pode levar alguns minutos dependendo da sua conexão.
> Na primeira execução, o tokenizador BERT (~700 MB) será baixado automaticamente do Hugging Face Hub.

### 4. Executar

```bash
python main.py
```

## O que o código faz

### Tarefa 1 – Motor de Frequências

`get_stats(vocab)` percorre o corpus e conta a frequência de cada par de símbolos adjacentes.
Dado o corpus inicial, o par `('e', 's')` atinge a contagem máxima de **9** (6 de *newest* + 3 de *widest*).

### Tarefa 2 – Loop de Fusão

`merge_vocab(pair, v_in)` substitui todas as ocorrências isoladas do par pelo token fundido.
O loop de treinamento executa **5 iterações**, produzindo tokens morfológicos como `est</w>`.

### Tarefa 3 – WordPiece com BERT

O tokenizador `bert-base-multilingual-cased` segmenta a frase de teste em sub-palavras WordPiece.

**Saída obtida:**

```python
['Os', 'hip', '##er', '-', 'par', '##âm', '##etros', 'do', 'transform', '##er',
 'são', 'in', '##cons', '##tit', '##uc', '##ional', '##mente', 'di', '##f',
 '##í', '##cei', '##s', 'de', 'aj', '##usta', '##r', '.']
```

#### Análise: o que significam os tokens com `##` e por que sub-palavras evitam o vocabulário desconhecido

O prefixo `##` indica que aquele fragmento **não é o início de uma palavra** — ele é uma continuação do token anterior. Por exemplo, `inconstitucionalmente` é decomposto em `['in', '##cons', '##tit', '##uc', '##ional', '##mente']`: o token `in` abre a palavra e cada `##xxx` seguinte é colado ao anterior para reconstruí-la. Essa estratégia resolve diretamente o problema de vocabulário desconhecido: um modelo baseado em palavras inteiras atribui o token especial `[UNK]` a qualquer palavra não vista durante o treinamento, perdendo completamente seu significado. Com sub-palavras, o modelo compõe qualquer palavra nova a partir de fragmentos já conhecidos — `inconstitucionalmente` não precisa estar no vocabulário porque seus morfemas (`in-`, `const-`, `-mente`) já carregam significado individualmente. Isso elimina praticamente todos os tokens `[UNK]` e garante que o modelo generalize para textos fora da distribuição de treinamento.

---

## Crédito de uso de IA Generativa

Este laboratório foi desenvolvido com auxílio de IA(Claude Sonnet).
A IA foi utilizada para:

- Revisão da função `merge_vocab` em `bpe/merger.py`, incluindo a expressão regular `(?<!\S)` / `(?!\S)` para garantir que apenas pares isolados fossem substituídos (evitando fusões parciais dentro de tokens já fundidos).
- Sugestão da estrutura modular do projeto (`bpe/` e `wordpiece/` como pacotes separados).
- Revisão geral do código.

Todo o código gerado foi lido, compreendido e validado pelo aluno antes de ser commitado.

---

## Referências

- SENNRICH, R.; HADDOW, B.; BIRCH, A. **Neural Machine Translation of Rare Words with Subword Units**. ACL, 2016. Artigo original que propõe o BPE aplicado a NLP.
- VASWANI, A. et al. **Attention Is All You Need**. NeurIPS, 2017. Arquitetura Transformer e o uso de vocabulários de sub-palavras de 32k–37k tokens.
- DEVLIN, J. et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL, 2019. Introduz o WordPiece e o modelo `bert-base-multilingual-cased`.
- WOLF, T. et al. **Transformers: State-of-the-Art Natural Language Processing**. EMNLP, 2020. Biblioteca Hugging Face Transformers utilizada na Tarefa 3.
- Hugging Face. **AutoTokenizer documentation**. Disponível em: [huggingface.co/docs/transformers/main_classes/tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)
