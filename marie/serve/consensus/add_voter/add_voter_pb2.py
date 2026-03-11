from google.protobuf import __version__ as __pb__version__

if int(__pb__version__.split('.')[0]) >= 4:
    from marie.serve.consensus.add_voter.pb.add_voter_pb2 import *
else:
    from marie.serve.consensus.add_voter.pb2.add_voter_pb2 import *
