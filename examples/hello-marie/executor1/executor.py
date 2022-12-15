import numpy as np
import torch

from marie import DocumentArray, Executor, requests


class MyExecutorAA(Executor):
    @requests(on="/aa")
    def foo(self, docs: DocumentArray, **kwargs):
        print('AAA')
        docs[0].text = 'AA - hello, world!'
        docs[1].text = 'AA - goodbye, world!'

    @requests(on='/crunch-numbers-aa')
    def bar(self, docs: DocumentArray, **kwargs):
        docs[0].text = 'crunch, aa'
