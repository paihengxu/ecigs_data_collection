"""
Utilities for dealing with text data.

Adrian Benton
2/26/2015
"""


class Alphabet:
    def __init__(self):
        self._wToI = {}
        self._iToW = {}
        self.__size = 0
        self.isFixed = False

    def wtoi(self, w):
        if w in self._wToI:
            return self._wToI[w]
        elif not self.isFixed:
            self._wToI[w] = self.__size
            self._iToW[self.__size] = w
            self.__size += 1
            return self._wToI[w]
        else:
            return None

    def put(self, w):
        return self.wtoi(w)

    def itow(self, i):
        if i not in self._iToW:
            return None
        else:
            return self._iToW[i]

    def get(self, i):
        return self.itow(i)

    def __len__(self):
        return self.__size
