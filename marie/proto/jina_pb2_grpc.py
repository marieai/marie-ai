from google.protobuf import __version__ as __pb__version__

if int(__pb__version__.split('.')[0]) >= 4:
    from marie.proto.docarray_v2.pb.jina_pb2_grpc import *
else:
    from marie.proto.docarray_v2.pb2.jina_pb2_grpc import *
