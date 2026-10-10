"""Treinamento comparativo de SimpleRNN e CNN no Fashion-MNIST."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

LABELS = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco", "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]


def preparar_dados(seed: int = 42, validacao: float = 0.1):
    """Divide somente o treino original em treino/validação; preserva teste."""
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    if not 0 < validacao < 1:
        raise ValueError("validacao deve estar entre zero e um")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train))
    corte = int(len(x_train) * (1 - validacao))
    treino, val = idx[:corte], idx[corte:]
    return (x_train[treino], y_train[treino]), (x_train[val], y_train[val]), (x_test, y_test)


def criar_modelo(tipo: str = "rnn"):
    """RNN recebe linhas de pixels como sequência; CNN recebe canal de imagem."""
    from tensorflow.keras import Sequential, layers
    if tipo == "rnn":
        return Sequential([
            layers.Input(shape=(28, 28)),
            layers.SimpleRNN(64, activation="tanh"),
            layers.Dense(32, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ], name="rnn_fashion_mnist")
    if tipo == "cnn":
        return Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ], name="cnn_fashion_mnist")
    raise ValueError("tipo deve ser 'rnn' ou 'cnn'")


def formato_entrada(x, tipo):
    if tipo == "cnn":
        return x[..., np.newaxis]
    if tipo == "rnn":
        return x
    raise ValueError("tipo inválido")


def executar(tipo='ambos', epocas=5, batch_size=128, seed=42, saida='resultados'):
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tf.keras.utils.set_random_seed(seed)
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = preparar_dados(seed)
    pasta = Path(saida); pasta.mkdir(parents=True, exist_ok=True)
    tipos = ['rnn', 'cnn'] if tipo == 'ambos' else [tipo]
    resultados = {}
    for nome in tipos:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        modelo = criar_modelo(nome)
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        historico = modelo.fit(formato_entrada(x_tr, nome), y_tr,
                              validation_data=(formato_entrada(x_val, nome), y_val),
                              epochs=epocas, batch_size=batch_size, verbose=2)
        perda, acuracia = modelo.evaluate(formato_entrada(x_te, nome), y_te, verbose=0)
        pred = np.argmax(modelo.predict(formato_entrada(x_te, nome), verbose=0), axis=1)
        from sklearn.metrics import classification_report, confusion_matrix
        relatorio = classification_report(y_te, pred, target_names=LABELS, output_dict=True, zero_division=0)
        resultados[nome] = {'perda_teste': float(perda), 'acuracia_teste': float(acuracia),
                            'f1_macro': float(relatorio['macro avg']['f1-score']),
                            'epocas': epocas, 'seed': seed}
        (pasta/f'{nome}_relatorio.json').write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding='utf-8')
        np.savetxt(pasta/f'{nome}_matriz_confusao.csv', confusion_matrix(y_te,pred), fmt='%d', delimiter=',')
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        for ax, chave, rotulo in [(axes[0],'accuracy','Acurácia'),(axes[1],'loss','Perda')]:
            ax.plot(historico.history[chave], label='Treinamento')
            ax.plot(historico.history['val_'+chave], label='Validação')
            ax.set(xlabel='Época', ylabel=rotulo, title=f'{nome.upper()}: {rotulo}')
            ax.legend()
        fig.tight_layout(); fig.savefig(pasta/f'{nome}_curvas.png', dpi=140); plt.close(fig)
        modelo.save(pasta/f'{nome}.keras')
        print(f'{nome.upper()}: accuracy={acuracia:.4f}, F1 macro={resultados[nome]["f1_macro"]:.4f}')
    (pasta/'metricas.json').write_text(json.dumps(resultados,indent=2,ensure_ascii=False), encoding='utf-8')
    return resultados


def main():
    parser = argparse.ArgumentParser(description='Compara RNN e CNN no Fashion-MNIST')
    parser.add_argument('--modelo', choices=['rnn','cnn','ambos'], default='ambos')
    parser.add_argument('--epocas', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--saida', default='resultados')
    args = parser.parse_args()
    if args.epocas < 1 or args.batch_size < 1:
        parser.error('epocas e batch-size devem ser positivos')
    executar(args.modelo, args.epocas, args.batch_size, args.seed, args.saida)

if __name__ == '__main__':
    main()
