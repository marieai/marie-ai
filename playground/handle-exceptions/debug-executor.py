from marie import Deployment, Executor, requests


class CustomExecutor(Executor):
    @requests
    def foo(self, **kwargs):
        a = 25
        import epdb

        epdb.set_trace()
        print(f'\n\na={a}\n\n')


def main():
    dep = Deployment(uses=CustomExecutor)
    with dep:
        dep.post(on='')


if __name__ == '__main__':
    main()
