"""Note, this should be consistent with hubble errors."""
from typing import Optional


class BaseError(Exception):
    response = None
    data = {}
    code = -1
    message = "An unknown error occurred"

    def __init__(
        self,
        response: dict,
        data: dict = {},
        message: Optional[str] = None,
        code: Optional[int] = None,
    ):
        self.response = response
        if message:
            self.message = message
        if code:
            self.code = code
        if data:
            self.data = data

    def __str__(self):
        if self.code:
            return f'{self.code}: {self.message}'
        return self.message


class ParamValidationError(BaseError):
    ...


class SQLCreationError(BaseError):
    ...


class DataStreamBrokenError(BaseError):
    ...


class UnexpectedMinetypeError(BaseError):
    ...


class SSOLoginRequiredError(BaseError):
    ...


class AuthenticationFailedError(BaseError):
    ...


class AuthenticationRequiredError(BaseError):
    ...


class OperationNotAllowedError(BaseError):
    ...


class InternalResourceNotFoundError(BaseError):
    ...


class RpcMethodNotFOundError(BaseError):
    ...


class RequestedEntityNotFoundError(BaseError):
    ...


class InternalResourceMethodNotFoundError(BaseError):
    ...


class IncompatiableMethodError(BaseError):
    ...


class InternalResourceIdConflictError(BaseError):
    ...


class FileTooLargeError(BaseError):
    ...


class MaximumUploadSizeReachedError(BaseError):
    ...


class InternalDataCorruptionError(BaseError):
    ...


class IdentifierNamespaceOccupiedError(BaseError):
    ...


class SubmittedDataMalformedError(BaseError):
    ...


class ExternalServiceFailureError(BaseError):
    ...


class ServerInternalError(BaseError):
    ...


class DownstreamServiceError(BaseError):
    ...


class ServerSubprocessError(BaseError):
    ...


class SandboxBuildNotfoundError(BaseError):
    ...


class NotImplementedError(BaseError):
    ...


class ResponseStreamClosedError(BaseError):
    ...


errorcodes = {
    -1: BaseError,
    40001: ParamValidationError,
    40002: SQLCreationError,
    40003: DataStreamBrokenError,
    40004: UnexpectedMinetypeError,
    40101: SSOLoginRequiredError,
    40102: AuthenticationFailedError,
    40103: AuthenticationRequiredError,
    40301: OperationNotAllowedError,
    40401: InternalResourceNotFoundError,
    40402: RpcMethodNotFOundError,
    40403: RequestedEntityNotFoundError,
    40501: InternalResourceMethodNotFoundError,
    40502: IncompatiableMethodError,
    40901: InternalResourceIdConflictError,
    41301: FileTooLargeError,
    41302: MaximumUploadSizeReachedError,
    42201: InternalDataCorruptionError,
    42202: IdentifierNamespaceOccupiedError,
    42203: SubmittedDataMalformedError,
    42204: ExternalServiceFailureError,
    50001: ServerInternalError,
    50002: DownstreamServiceError,
    50003: ServerSubprocessError,
    50004: SandboxBuildNotfoundError,
    50005: NotImplementedError,
    50006: ResponseStreamClosedError,
}
