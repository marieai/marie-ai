# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from ... import serializer as jina__pb2


class JinaDataRequestRPCStub(object):
    """*
    jina gRPC service for DataRequests.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.process_data = channel.unary_unary(
                '/jina.JinaDataRequestRPC/process_data',
                request_serializer=jina__pb2.DataRequestListProto.SerializeToString,
                response_deserializer=jina__pb2.DataRequestProto.FromString,
                _registered_method=True)


class JinaDataRequestRPCServicer(object):
    """*
    jina gRPC service for DataRequests.
    """

    def process_data(self, request, context):
        """Used for passing DataRequests to the Executors
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaDataRequestRPCServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'process_data': grpc.unary_unary_rpc_method_handler(
                    servicer.process_data,
                    request_deserializer=jina__pb2.DataRequestListProto.FromString,
                    response_serializer=jina__pb2.DataRequestProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaDataRequestRPC', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaDataRequestRPC', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaDataRequestRPC(object):
    """*
    jina gRPC service for DataRequests.
    """

    @staticmethod
    def process_data(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/jina.JinaDataRequestRPC/process_data',
            jina__pb2.DataRequestListProto.SerializeToString,
            jina__pb2.DataRequestProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaSingleDataRequestRPCStub(object):
    """*
    jina gRPC service for DataRequests.
    This is used to send requests to Executors when a list of requests is not needed
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.process_single_data = channel.unary_unary(
                '/jina.JinaSingleDataRequestRPC/process_single_data',
                request_serializer=jina__pb2.DataRequestProto.SerializeToString,
                response_deserializer=jina__pb2.DataRequestProto.FromString,
                _registered_method=True)


class JinaSingleDataRequestRPCServicer(object):
    """*
    jina gRPC service for DataRequests.
    This is used to send requests to Executors when a list of requests is not needed
    """

    def process_single_data(self, request, context):
        """Used for passing DataRequests to the Executors
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaSingleDataRequestRPCServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'process_single_data': grpc.unary_unary_rpc_method_handler(
                    servicer.process_single_data,
                    request_deserializer=jina__pb2.DataRequestProto.FromString,
                    response_serializer=jina__pb2.DataRequestProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaSingleDataRequestRPC', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaSingleDataRequestRPC', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaSingleDataRequestRPC(object):
    """*
    jina gRPC service for DataRequests.
    This is used to send requests to Executors when a list of requests is not needed
    """

    @staticmethod
    def process_single_data(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/jina.JinaSingleDataRequestRPC/process_single_data',
            jina__pb2.DataRequestProto.SerializeToString,
            jina__pb2.DataRequestProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaSingleDocumentRequestRPCStub(object):
    """*
    jina gRPC service for DataRequests.
    This is used to send requests to Executors when a list of requests is not needed
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.stream_doc = channel.unary_stream(
                '/jina.JinaSingleDocumentRequestRPC/stream_doc',
                request_serializer=jina__pb2.SingleDocumentRequestProto.SerializeToString,
                response_deserializer=jina__pb2.SingleDocumentRequestProto.FromString,
                _registered_method=True)


class JinaSingleDocumentRequestRPCServicer(object):
    """*
    jina gRPC service for DataRequests.
    This is used to send requests to Executors when a list of requests is not needed
    """

    def stream_doc(self, request, context):
        """Used for streaming one document to the Executors
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaSingleDocumentRequestRPCServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'stream_doc': grpc.unary_stream_rpc_method_handler(
                    servicer.stream_doc,
                    request_deserializer=jina__pb2.SingleDocumentRequestProto.FromString,
                    response_serializer=jina__pb2.SingleDocumentRequestProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaSingleDocumentRequestRPC', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaSingleDocumentRequestRPC', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaSingleDocumentRequestRPC(object):
    """*
    jina gRPC service for DataRequests.
    This is used to send requests to Executors when a list of requests is not needed
    """

    @staticmethod
    def stream_doc(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(
            request,
            target,
            '/jina.JinaSingleDocumentRequestRPC/stream_doc',
            jina__pb2.SingleDocumentRequestProto.SerializeToString,
            jina__pb2.SingleDocumentRequestProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaRPCStub(object):
    """*
    jina streaming gRPC service.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Call = channel.stream_stream(
                '/jina.JinaRPC/Call',
                request_serializer=jina__pb2.DataRequestProto.SerializeToString,
                response_deserializer=jina__pb2.DataRequestProto.FromString,
                _registered_method=True)


class JinaRPCServicer(object):
    """*
    jina streaming gRPC service.
    """

    def Call(self, request_iterator, context):
        """Pass in a Request and a filled Request with matches will be returned.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaRPCServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Call': grpc.stream_stream_rpc_method_handler(
                    servicer.Call,
                    request_deserializer=jina__pb2.DataRequestProto.FromString,
                    response_serializer=jina__pb2.DataRequestProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaRPC', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaRPC', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaRPC(object):
    """*
    jina streaming gRPC service.
    """

    @staticmethod
    def Call(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            '/jina.JinaRPC/Call',
            jina__pb2.DataRequestProto.SerializeToString,
            jina__pb2.DataRequestProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaDiscoverEndpointsRPCStub(object):
    """*
    jina gRPC service to expose Endpoints from Executors.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.endpoint_discovery = channel.unary_unary(
                '/jina.JinaDiscoverEndpointsRPC/endpoint_discovery',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=jina__pb2.EndpointsProto.FromString,
                _registered_method=True)


class JinaDiscoverEndpointsRPCServicer(object):
    """*
    jina gRPC service to expose Endpoints from Executors.
    """

    def endpoint_discovery(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaDiscoverEndpointsRPCServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'endpoint_discovery': grpc.unary_unary_rpc_method_handler(
                    servicer.endpoint_discovery,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=jina__pb2.EndpointsProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaDiscoverEndpointsRPC', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaDiscoverEndpointsRPC', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaDiscoverEndpointsRPC(object):
    """*
    jina gRPC service to expose Endpoints from Executors.
    """

    @staticmethod
    def endpoint_discovery(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/jina.JinaDiscoverEndpointsRPC/endpoint_discovery',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            jina__pb2.EndpointsProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaGatewayDryRunRPCStub(object):
    """*
    jina gRPC service to expose Endpoints from Executors.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.dry_run = channel.unary_unary(
                '/jina.JinaGatewayDryRunRPC/dry_run',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=jina__pb2.StatusProto.FromString,
                _registered_method=True)


class JinaGatewayDryRunRPCServicer(object):
    """*
    jina gRPC service to expose Endpoints from Executors.
    """

    def dry_run(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaGatewayDryRunRPCServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'dry_run': grpc.unary_unary_rpc_method_handler(
                    servicer.dry_run,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=jina__pb2.StatusProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaGatewayDryRunRPC', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaGatewayDryRunRPC', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaGatewayDryRunRPC(object):
    """*
    jina gRPC service to expose Endpoints from Executors.
    """

    @staticmethod
    def dry_run(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/jina.JinaGatewayDryRunRPC/dry_run',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            jina__pb2.StatusProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaInfoRPCStub(object):
    """*
    jina gRPC service to expose information about running jina version and environment.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self._status = channel.unary_unary(
                '/jina.JinaInfoRPC/_status',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=jina__pb2.JinaInfoProto.FromString,
                _registered_method=True)


class JinaInfoRPCServicer(object):
    """*
    jina gRPC service to expose information about running jina version and environment.
    """

    def _status(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaInfoRPCServicer_to_server(servicer, server):
    rpc_method_handlers = {
            '_status': grpc.unary_unary_rpc_method_handler(
                    servicer._status,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=jina__pb2.JinaInfoProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaInfoRPC', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaInfoRPC', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaInfoRPC(object):
    """*
    jina gRPC service to expose information about running jina version and environment.
    """

    @staticmethod
    def _status(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/jina.JinaInfoRPC/_status',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            jina__pb2.JinaInfoProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaExecutorSnapshotStub(object):
    """*
    jina gRPC service to trigger a snapshot at the Executor Runtime.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.snapshot = channel.unary_unary(
                '/jina.JinaExecutorSnapshot/snapshot',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=jina__pb2.SnapshotStatusProto.FromString,
                _registered_method=True)


class JinaExecutorSnapshotServicer(object):
    """*
    jina gRPC service to trigger a snapshot at the Executor Runtime.
    """

    def snapshot(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaExecutorSnapshotServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'snapshot': grpc.unary_unary_rpc_method_handler(
                    servicer.snapshot,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=jina__pb2.SnapshotStatusProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaExecutorSnapshot', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaExecutorSnapshot', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaExecutorSnapshot(object):
    """*
    jina gRPC service to trigger a snapshot at the Executor Runtime.
    """

    @staticmethod
    def snapshot(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/jina.JinaExecutorSnapshot/snapshot',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            jina__pb2.SnapshotStatusProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaExecutorSnapshotProgressStub(object):
    """*
    jina gRPC service to trigger a snapshot at the Executor Runtime.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.snapshot_status = channel.unary_unary(
                '/jina.JinaExecutorSnapshotProgress/snapshot_status',
                request_serializer=jina__pb2.SnapshotId.SerializeToString,
                response_deserializer=jina__pb2.SnapshotStatusProto.FromString,
                _registered_method=True)


class JinaExecutorSnapshotProgressServicer(object):
    """*
    jina gRPC service to trigger a snapshot at the Executor Runtime.
    """

    def snapshot_status(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaExecutorSnapshotProgressServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'snapshot_status': grpc.unary_unary_rpc_method_handler(
                    servicer.snapshot_status,
                    request_deserializer=jina__pb2.SnapshotId.FromString,
                    response_serializer=jina__pb2.SnapshotStatusProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaExecutorSnapshotProgress', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaExecutorSnapshotProgress', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaExecutorSnapshotProgress(object):
    """*
    jina gRPC service to trigger a snapshot at the Executor Runtime.
    """

    @staticmethod
    def snapshot_status(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/jina.JinaExecutorSnapshotProgress/snapshot_status',
            jina__pb2.SnapshotId.SerializeToString,
            jina__pb2.SnapshotStatusProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaExecutorRestoreStub(object):
    """*
    jina gRPC service to trigger a restore at the Executor Runtime.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.restore = channel.unary_unary(
                '/jina.JinaExecutorRestore/restore',
                request_serializer=jina__pb2.RestoreSnapshotCommand.SerializeToString,
                response_deserializer=jina__pb2.RestoreSnapshotStatusProto.FromString,
                _registered_method=True)


class JinaExecutorRestoreServicer(object):
    """*
    jina gRPC service to trigger a restore at the Executor Runtime.
    """

    def restore(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaExecutorRestoreServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'restore': grpc.unary_unary_rpc_method_handler(
                    servicer.restore,
                    request_deserializer=jina__pb2.RestoreSnapshotCommand.FromString,
                    response_serializer=jina__pb2.RestoreSnapshotStatusProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaExecutorRestore', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaExecutorRestore', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaExecutorRestore(object):
    """*
    jina gRPC service to trigger a restore at the Executor Runtime.
    """

    @staticmethod
    def restore(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/jina.JinaExecutorRestore/restore',
            jina__pb2.RestoreSnapshotCommand.SerializeToString,
            jina__pb2.RestoreSnapshotStatusProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JinaExecutorRestoreProgressStub(object):
    """*
    jina gRPC service to trigger a snapshot at the Executor Runtime.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.restore_status = channel.unary_unary(
                '/jina.JinaExecutorRestoreProgress/restore_status',
                request_serializer=jina__pb2.RestoreId.SerializeToString,
                response_deserializer=jina__pb2.RestoreSnapshotStatusProto.FromString,
                _registered_method=True)


class JinaExecutorRestoreProgressServicer(object):
    """*
    jina gRPC service to trigger a snapshot at the Executor Runtime.
    """

    def restore_status(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JinaExecutorRestoreProgressServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'restore_status': grpc.unary_unary_rpc_method_handler(
                    servicer.restore_status,
                    request_deserializer=jina__pb2.RestoreId.FromString,
                    response_serializer=jina__pb2.RestoreSnapshotStatusProto.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'jina.JinaExecutorRestoreProgress', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('jina.JinaExecutorRestoreProgress', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JinaExecutorRestoreProgress(object):
    """*
    jina gRPC service to trigger a snapshot at the Executor Runtime.
    """

    @staticmethod
    def restore_status(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/jina.JinaExecutorRestoreProgress/restore_status',
            jina__pb2.RestoreId.SerializeToString,
            jina__pb2.RestoreSnapshotStatusProto.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
