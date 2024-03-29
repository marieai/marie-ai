def check_health_pod(addr: str):
    """check if a pods is healthy

    :param addr: the address on which the pod is serving ex : localhost:1234
    """
    from marie.serve.runtimes.servers import BaseServer

    is_ready = BaseServer.is_ready(addr)

    if not is_ready:
        raise Exception('Pod is unhealthy')

    print('The Pod is healthy')


if __name__ == '__main__':
    """
    Health check cli (for docker):

    Example:
        python marie.resources.health_check.pod localhost:1234
    """
    import sys

    if len(sys.argv) < 2:
        raise ValueError('You need to specify a address to check health')

    addr = sys.argv[1]
    check_health_pod(addr)
