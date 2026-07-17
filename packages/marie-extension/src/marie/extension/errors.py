class ExtensionError(Exception):
    """Base class for marie-extension errors."""


class PackageLoadError(ExtensionError):
    pass


class PackageValidationError(ExtensionError):
    pass
