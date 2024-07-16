from typing import List

import numpy
import numpy as np


class SmilesEncoder:

    def __init__(self):

        self._vocab = ["unk"]
        self._max_len = 0

    def fit(self, smiles: List[str]) -> None:

        for smi in smiles:
            for char in smi:
                if char not in self._vocab:
                    self._vocab.append(char)
            if len(smi) > self._max_len:
                self._max_len = len(smi)

    def encode(self, smiles: str) -> numpy.ndarray:

        encoding = np.zeros((self._max_len, len(self._vocab)))
        for idx, char in enumerate(smiles):
            encoding[idx, self._vocab.index(char)] = 1.0
        return encoding

    def encode_many(self, smiles: List[str]) -> numpy.ndarray:

        return np.concatenate([
            np.expand_dims(self.encode(smi), axis=0)
            for smi in smiles
        ], axis=0)

    def decode(self, X: numpy.ndarray) -> str:

        smiles = ""
        for item in X:
            _loc = np.where(item == 1)[0]
            if len(_loc) > 0:
                smiles += self._vocab[int(_loc[0])]
        return smiles

    def decode_many(self, X: numpy.ndarray) -> List[str]:

        return [self.decode(entry) for entry in X]
