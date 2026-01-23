# Context providers for prompt template injection
# Import built-in providers to trigger registration

from .table_context import (  # noqa: F401
    TableClaimContextProvider,
    TableContextProvider,
    TableRemarkCodesContextProvider,
)
