import numpy as np
import torch

from marie import DocumentArray, Executor, requests


class MyExecutorXX(Executor):
    @requests(on="/xx")
    def foo(self, docs: DocumentArray, **kwargs):
        print('XXX')
        docs[0].text = 'XX - hello, world!'
        docs[1].text = 'XX - goodbye, world!'

    @requests(on='/crunch-numbers-xx')
    def bar(self, docs: DocumentArray, **kwargs):
        docs[0].text = 'crunch, xx'
