def is_notebook():  # pragma: no cover
    """
    Check if we're running in a Jupyter notebook, using magic command `get_ipython` that only available in Jupyter.
    :return: True if run in a Jupyter notebook else False.
    """

    try:
        get_ipython  # noqa: F821
    except NameError:
        return False

    shell = get_ipython().__class__.__name__  # noqa: F821

    if shell == 'ZMQInteractiveShell':
        return True

    elif shell == 'Shell':
        return True

    elif shell == 'TerminalInteractiveShell':
        return False

    else:
        return False
