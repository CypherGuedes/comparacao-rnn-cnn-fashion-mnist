# Comparação entre RNN e CNN no Fashion-MNIST

Projeto educacional de **Deep Learning em Python e TensorFlow/Keras**, com treinamento e avaliação de uma rede neural recorrente (**SimpleRNN**) e de uma rede convolucional (**CNN**) para classificação de imagens de roupas.

> **Nota de transparência:** o repositório anterior chamava-se `treinamento-de-rnn`, mas seu notebook original implementava **somente uma CNN**. O material original foi preservado em `original/`. A RNN e o experimento comparativo foram adicionados nesta revisão; resultados não foram inventados.

## Objetivo

Estudar a diferença entre uma arquitetura sequencial e uma convolucional usando o mesmo conjunto de dados. A comparação é **didática**, não uma demonstração de que RNN seja a arquitetura recomendada para reconhecimento de imagens.

## Dados

Utilizamos o [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), disponível pelo TensorFlow: 70.000 imagens 28×28 em tons de cinza, divididas originalmente em 60.000 para treino e 10.000 para teste, com 10 categorias. A primeira execução pode baixar os dados.

| Partição | Exemplos | Uso |
| --- | ---: | --- |
| Treinamento | 54.000 | Ajuste dos pesos |
| Validação | 6.000 | Acompanhamento durante treinamento |
| Teste | 10.000 | Avaliação final, sem uso no `fit` |

A separação entre treinamento e validação é aleatória com semente 42. Os pixels são normalizados para [0, 1], sem data augmentation.

## Arquiteturas

| Aspecto | SimpleRNN | CNN |
| --- | --- | --- |
| Entrada | 28 passos × 28 atributos | 28 × 28 × 1 |
| Camadas principais | SimpleRNN(64), Dense(32) | Conv2D(32), MaxPool, Conv2D(64), MaxPool, Dense(64) |
| Saída | Dense(10, softmax) | Dense(10, softmax) |
| Perda | Sparse categorical crossentropy | Sparse categorical crossentropy |
| Otimizador | Adam | Adam |

A RNN trata **cada linha da imagem como um passo temporal artificial**, sem sugerir que pixels sejam dados temporais reais. A CNN mantém relações espaciais por meio de filtros convolucionais.

## Estrutura

```text
comparacao-rnn-cnn-fashion-mnist/
├── Comparacao_RNN_CNN_Fashion_MNIST.ipynb  # estudo interativo
├── src/
│   ├── __init__.py
│   └── treinamento.py                     # pipeline por linha de comando
├── tests/
│   └── test_estrutura.py                  # testes básicos
├── original/
│   └── Copia_de_Treinamento_de_RNN.ipynb # notebook histórico preservado
├── docs/
│   └── ALTERACOES.md
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalação e execução local

Recomendado Python 3.10 a 3.12 com versão compatível do TensorFlow para seu sistema.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
python -m src.treinamento --modelo ambos --epocas 5 --batch-size 128
```

Para executar apenas a RNN ou CNN, substitua `ambos` por `rnn` ou `cnn`. A pasta `resultados/` guarda modelos `.keras`, curvas de treinamento, métricas JSON, relatórios por classe e matrizes de confusão CSV. Essa pasta é ignorada pelo Git.

### Google Colab

Abra o notebook `Comparacao_RNN_CNN_Fashion_MNIST.ipynb` no Colab e envie também a pasta `src/` para a mesma pasta de trabalho, ou clone o repositório. Rode as células na ordem. O arquivo original mantém seu próprio link histórico do Colab; ele não representa a nova comparação.

### Testes rápidos

```bash
python -m unittest discover -s tests -v
```

Estes testes verificam dimensões e parâmetros básicos; **não substituem** testes de integração com TensorFlow nem execução completa do treinamento.

## Como avaliar

O pipeline calcula **accuracy e F1 macro** no teste, além de relatórios de precisão/recall/F1 por classe e matriz de confusão. Os gráficos permitem examinar evolução de loss e accuracy no treino e validação.

**Resultados:** devem ser obtidos após a execução. Nenhuma acurácia foi declarada sem treinamento realizado e confirmado. Para comparação mais consistente, repetir com diferentes sementes e relatar média/desvio-padrão e custo computacional.

## Limitações

- Um único conjunto de dados e poucas épocas não são suficientes para conclusões gerais sobre arquiteturas.
- A representação sequencial de linhas é artificial e a SimpleRNN pode perder dependências longas.
- As arquiteturas têm tamanhos distintos; a comparação não controla o número de parâmetros.
- GPU, versões das bibliotecas e semente podem alterar resultados e duração.

## Origem e evolução

Consulte [`docs/ALTERACOES.md`](docs/ALTERACOES.md). O notebook histórico não foi apagado, permitindo observar a evolução do trabalho.

## Tecnologias

Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn, Fashion-MNIST e notebooks Jupyter compatíveis com Google Colab.

## Referências

- [Fashion-MNIST, Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- [Keras SimpleRNN](https://keras.io/api/layers/recurrent_layers/simple_rnn/)
- [Keras Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/)

## Licença

Nenhuma licença foi adicionada automaticamente: escolha uma licença apenas se desejar estabelecer condições explícitas de reutilização.
