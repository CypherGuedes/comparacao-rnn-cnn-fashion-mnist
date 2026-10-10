import unittest
import numpy as np
from src.treinamento import formato_entrada, LABELS

class TestEstrutura(unittest.TestCase):
    def test_formatos(self):
        x = np.zeros((2, 28, 28), dtype=np.float32)
        self.assertEqual(formato_entrada(x, 'rnn').shape, (2, 28, 28))
        self.assertEqual(formato_entrada(x, 'cnn').shape, (2, 28, 28, 1))
    def test_rotulos(self):
        self.assertEqual(len(LABELS), 10)
    def test_tipo_invalido(self):
        with self.assertRaises(ValueError): formato_entrada(np.zeros((1,28,28)), 'outro')

if __name__ == '__main__': unittest.main()
