from google.protobuf import __version__ as __pb__version__

if __pb__version__.startswith('4'):
    from marie.proto.docarray_v2.pb.jina_pb2 import *
else:
    from marie.proto.docarray_v2.pb2.jina_pb2 import *
