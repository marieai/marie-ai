# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: add_voter.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0f\x61\x64\x64_voter.proto\"-\n\rAwaitResponse\x12\r\n\x05\x65rror\x18\x01 \x01(\t\x12\r\n\x05index\x18\x02 \x01(\x04\"\x10\n\x0e\x46orgetResponse\"!\n\x06\x46uture\x12\x17\n\x0foperation_token\x18\x01 \x01(\t\"F\n\x0f\x41\x64\x64VoterRequest\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07\x61\x64\x64ress\x18\x02 \x01(\t\x12\x16\n\x0eprevious_index\x18\x03 \x01(\x04\x32~\n\tRaftAdmin\x12\'\n\x08\x41\x64\x64Voter\x12\x10.AddVoterRequest\x1a\x07.Future\"\x00\x12\"\n\x05\x41wait\x12\x07.Future\x1a\x0e.AwaitResponse\"\x00\x12$\n\x06\x46orget\x12\x07.Future\x1a\x0f.ForgetResponse\"\x00\x62\x06proto3'
)


_AWAITRESPONSE = DESCRIPTOR.message_types_by_name['AwaitResponse']
_FORGETRESPONSE = DESCRIPTOR.message_types_by_name['ForgetResponse']
_FUTURE = DESCRIPTOR.message_types_by_name['Future']
_ADDVOTERREQUEST = DESCRIPTOR.message_types_by_name['AddVoterRequest']
AwaitResponse = _reflection.GeneratedProtocolMessageType(
    'AwaitResponse',
    (_message.Message,),
    {
        'DESCRIPTOR': _AWAITRESPONSE,
        '__module__': 'add_voter_pb2'
        # @@protoc_insertion_point(class_scope:AwaitResponse)
    },
)
_sym_db.RegisterMessage(AwaitResponse)

ForgetResponse = _reflection.GeneratedProtocolMessageType(
    'ForgetResponse',
    (_message.Message,),
    {
        'DESCRIPTOR': _FORGETRESPONSE,
        '__module__': 'add_voter_pb2'
        # @@protoc_insertion_point(class_scope:ForgetResponse)
    },
)
_sym_db.RegisterMessage(ForgetResponse)

Future = _reflection.GeneratedProtocolMessageType(
    'Future',
    (_message.Message,),
    {
        'DESCRIPTOR': _FUTURE,
        '__module__': 'add_voter_pb2'
        # @@protoc_insertion_point(class_scope:Future)
    },
)
_sym_db.RegisterMessage(Future)

AddVoterRequest = _reflection.GeneratedProtocolMessageType(
    'AddVoterRequest',
    (_message.Message,),
    {
        'DESCRIPTOR': _ADDVOTERREQUEST,
        '__module__': 'add_voter_pb2'
        # @@protoc_insertion_point(class_scope:AddVoterRequest)
    },
)
_sym_db.RegisterMessage(AddVoterRequest)

_RAFTADMIN = DESCRIPTOR.services_by_name['RaftAdmin']
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _AWAITRESPONSE._serialized_start = 19
    _AWAITRESPONSE._serialized_end = 64
    _FORGETRESPONSE._serialized_start = 66
    _FORGETRESPONSE._serialized_end = 82
    _FUTURE._serialized_start = 84
    _FUTURE._serialized_end = 117
    _ADDVOTERREQUEST._serialized_start = 119
    _ADDVOTERREQUEST._serialized_end = 189
    _RAFTADMIN._serialized_start = 191
    _RAFTADMIN._serialized_end = 317
# @@protoc_insertion_point(module_scope)