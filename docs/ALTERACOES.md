# Origem e alterações

O ZIP recebido continha apenas `Cópia_de_Treinamento_de_RNN.ipynb` e um README de uma linha. Apesar do nome, o notebook utilizava convoluções (Conv2D, MaxPooling2D) e não RNN. Além disso, passava o conjunto de teste como dados de validação durante o treinamento.

**Revisão:** notebook original preservado; implementação verdadeira de SimpleRNN; CNN comparativa; separação treino/validação/teste; avaliação final com accuracy/F1 e matriz de confusão; código de linha de comando; testes; README e dependências. A arquitetura da CNN comparativa foi simplificada em relação à original e não é reprodução exata daquela configuração.

**Limites da validação:** sintaxe e testes estruturais podem ser verificados sem baixar Fashion-MNIST; o treinamento completo exige TensorFlow e acesso ao dataset. Não confundir resultados históricos eventualmente guardados no notebook original com resultados da comparação nova.
