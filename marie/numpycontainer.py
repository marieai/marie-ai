class NumpyContainer:
    """
    Numpy ndarray wrapper
    """

    def __init__(self, value) -> None:
        self.value = value

    def __repr__(self):
        return f"<NumpyContainer:({self.value})"

    def __str__(self):
        return f"{self.value}"

    def val(self):
        return self.value
