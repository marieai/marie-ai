from marie import DocumentArray, Executor, requests


class ReloadExecutor(Executor):
    def __init__(self, argument, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.argument = argument

    @requests()
    def exec(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc.tags['argument'] = self.argument
