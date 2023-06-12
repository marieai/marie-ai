# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: jina.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import docarray.proto.pb.docarray_pb2 as docarray__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\njina.proto\x12\x04jina\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1cgoogle/protobuf/struct.proto\x1a\x1bgoogle/protobuf/empty.proto\x1a\x0e\x64ocarray.proto\"\x9f\x01\n\nRouteProto\x12\x10\n\x08\x65xecutor\x18\x01 \x01(\t\x12.\n\nstart_time\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12,\n\x08\x65nd_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12!\n\x06status\x18\x04 \x01(\x0b\x32\x11.jina.StatusProto\"\xc3\x01\n\rJinaInfoProto\x12+\n\x04jina\x18\x01 \x03(\x0b\x32\x1d.jina.JinaInfoProto.JinaEntry\x12+\n\x04\x65nvs\x18\x02 \x03(\x0b\x32\x1d.jina.JinaInfoProto.EnvsEntry\x1a+\n\tJinaEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a+\n\tEnvsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xc6\x01\n\x0bHeaderProto\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12!\n\x06status\x18\x02 \x01(\x0b\x32\x11.jina.StatusProto\x12\x1a\n\rexec_endpoint\x18\x03 \x01(\tH\x00\x88\x01\x01\x12\x1c\n\x0ftarget_executor\x18\x04 \x01(\tH\x01\x88\x01\x01\x12\x14\n\x07timeout\x18\x05 \x01(\rH\x02\x88\x01\x01\x42\x10\n\x0e_exec_endpointB\x12\n\x10_target_executorB\n\n\x08_timeout\"f\n\x0e\x45ndpointsProto\x12\x11\n\tendpoints\x18\x01 \x03(\t\x12\x17\n\x0fwrite_endpoints\x18\x02 \x03(\t\x12(\n\x07schemas\x18\x03 \x01(\x0b\x32\x17.google.protobuf.Struct\"\xf9\x01\n\x0bStatusProto\x12*\n\x04\x63ode\x18\x01 \x01(\x0e\x32\x1c.jina.StatusProto.StatusCode\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12\x33\n\texception\x18\x03 \x01(\x0b\x32 .jina.StatusProto.ExceptionProto\x1aN\n\x0e\x45xceptionProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04\x61rgs\x18\x02 \x03(\t\x12\x0e\n\x06stacks\x18\x03 \x03(\t\x12\x10\n\x08\x65xecutor\x18\x04 \x01(\t\"$\n\nStatusCode\x12\x0b\n\x07SUCCESS\x10\x00\x12\t\n\x05\x45RROR\x10\x01\"^\n\rRelatedEntity\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07\x61\x64\x64ress\x18\x02 \x01(\t\x12\x0c\n\x04port\x18\x03 \x01(\r\x12\x15\n\x08shard_id\x18\x04 \x01(\rH\x00\x88\x01\x01\x42\x0b\n\t_shard_id\"\x9a\x02\n\x10\x44\x61taRequestProto\x12!\n\x06header\x18\x01 \x01(\x0b\x32\x11.jina.HeaderProto\x12+\n\nparameters\x18\x02 \x01(\x0b\x32\x17.google.protobuf.Struct\x12 \n\x06routes\x18\x03 \x03(\x0b\x32\x10.jina.RouteProto\x12\x35\n\x04\x64\x61ta\x18\x04 \x01(\x0b\x32\'.jina.DataRequestProto.DataContentProto\x1a]\n\x10\x44\x61taContentProto\x12&\n\x04\x64ocs\x18\x01 \x01(\x0b\x32\x16.docarray.DocListProtoH\x00\x12\x14\n\ndocs_bytes\x18\x02 \x01(\x0cH\x00\x42\x0b\n\tdocuments\"\x8a\x01\n\x16\x44\x61taRequestProtoWoData\x12!\n\x06header\x18\x01 \x01(\x0b\x32\x11.jina.HeaderProto\x12+\n\nparameters\x18\x02 \x01(\x0b\x32\x17.google.protobuf.Struct\x12 \n\x06routes\x18\x03 \x03(\x0b\x32\x10.jina.RouteProto\"@\n\x14\x44\x61taRequestListProto\x12(\n\x08requests\x18\x01 \x03(\x0b\x32\x16.jina.DataRequestProto\"\x1b\n\nSnapshotId\x12\r\n\x05value\x18\x01 \x01(\t\"\x1a\n\tRestoreId\x12\r\n\x05value\x18\x01 \x01(\t\"\xef\x01\n\x13SnapshotStatusProto\x12\x1c\n\x02id\x18\x01 \x01(\x0b\x32\x10.jina.SnapshotId\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .jina.SnapshotStatusProto.Status\x12\x15\n\rsnapshot_file\x18\x03 \x01(\t\"q\n\x06Status\x12\x15\n\x11STATE_UNSPECIFIED\x10\x00\x12\n\n\x06QUEUED\x10\x01\x12\r\n\tSCHEDULED\x10\x02\x12\x0b\n\x07RUNNING\x10\x03\x12\r\n\tSUCCEEDED\x10\x04\x12\n\n\x06\x46\x41ILED\x10\x05\x12\r\n\tNOT_FOUND\x10\x06\"\xca\x01\n\x1aRestoreSnapshotStatusProto\x12\x1b\n\x02id\x18\x01 \x01(\x0b\x32\x0f.jina.RestoreId\x12\x37\n\x06status\x18\x02 \x01(\x0e\x32\'.jina.RestoreSnapshotStatusProto.Status\"V\n\x06Status\x12\x15\n\x11STATE_UNSPECIFIED\x10\x00\x12\x0b\n\x07RUNNING\x10\x01\x12\r\n\tSUCCEEDED\x10\x02\x12\n\n\x06\x46\x41ILED\x10\x03\x12\r\n\tNOT_FOUND\x10\x06\"/\n\x16RestoreSnapshotCommand\x12\x15\n\rsnapshot_file\x18\x01 \x01(\t2Z\n\x12JinaDataRequestRPC\x12\x44\n\x0cprocess_data\x12\x1a.jina.DataRequestListProto\x1a\x16.jina.DataRequestProto\"\x00\x32\x63\n\x18JinaSingleDataRequestRPC\x12G\n\x13process_single_data\x12\x16.jina.DataRequestProto\x1a\x16.jina.DataRequestProto\"\x00\x32G\n\x07JinaRPC\x12<\n\x04\x43\x61ll\x12\x16.jina.DataRequestProto\x1a\x16.jina.DataRequestProto\"\x00(\x01\x30\x01\x32`\n\x18JinaDiscoverEndpointsRPC\x12\x44\n\x12\x65ndpoint_discovery\x12\x16.google.protobuf.Empty\x1a\x14.jina.EndpointsProto\"\x00\x32N\n\x14JinaGatewayDryRunRPC\x12\x36\n\x07\x64ry_run\x12\x16.google.protobuf.Empty\x1a\x11.jina.StatusProto\"\x00\x32G\n\x0bJinaInfoRPC\x12\x38\n\x07_status\x12\x16.google.protobuf.Empty\x1a\x13.jina.JinaInfoProto\"\x00\x32W\n\x14JinaExecutorSnapshot\x12?\n\x08snapshot\x12\x16.google.protobuf.Empty\x1a\x19.jina.SnapshotStatusProto\"\x00\x32`\n\x1cJinaExecutorSnapshotProgress\x12@\n\x0fsnapshot_status\x12\x10.jina.SnapshotId\x1a\x19.jina.SnapshotStatusProto\"\x00\x32\x62\n\x13JinaExecutorRestore\x12K\n\x07restore\x12\x1c.jina.RestoreSnapshotCommand\x1a .jina.RestoreSnapshotStatusProto\"\x00\x32\x64\n\x1bJinaExecutorRestoreProgress\x12\x45\n\x0erestore_status\x12\x0f.jina.RestoreId\x1a .jina.RestoreSnapshotStatusProto\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'jina_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _JINAINFOPROTO_JINAENTRY._options = None
  _JINAINFOPROTO_JINAENTRY._serialized_options = b'8\001'
  _JINAINFOPROTO_ENVSENTRY._options = None
  _JINAINFOPROTO_ENVSENTRY._serialized_options = b'8\001'
  _ROUTEPROTO._serialized_start=129
  _ROUTEPROTO._serialized_end=288
  _JINAINFOPROTO._serialized_start=291
  _JINAINFOPROTO._serialized_end=486
  _JINAINFOPROTO_JINAENTRY._serialized_start=398
  _JINAINFOPROTO_JINAENTRY._serialized_end=441
  _JINAINFOPROTO_ENVSENTRY._serialized_start=443
  _JINAINFOPROTO_ENVSENTRY._serialized_end=486
  _HEADERPROTO._serialized_start=489
  _HEADERPROTO._serialized_end=687
  _ENDPOINTSPROTO._serialized_start=689
  _ENDPOINTSPROTO._serialized_end=791
  _STATUSPROTO._serialized_start=794
  _STATUSPROTO._serialized_end=1043
  _STATUSPROTO_EXCEPTIONPROTO._serialized_start=927
  _STATUSPROTO_EXCEPTIONPROTO._serialized_end=1005
  _STATUSPROTO_STATUSCODE._serialized_start=1007
  _STATUSPROTO_STATUSCODE._serialized_end=1043
  _RELATEDENTITY._serialized_start=1045
  _RELATEDENTITY._serialized_end=1139
  _DATAREQUESTPROTO._serialized_start=1142
  _DATAREQUESTPROTO._serialized_end=1424
  _DATAREQUESTPROTO_DATACONTENTPROTO._serialized_start=1331
  _DATAREQUESTPROTO_DATACONTENTPROTO._serialized_end=1424
  _DATAREQUESTPROTOWODATA._serialized_start=1427
  _DATAREQUESTPROTOWODATA._serialized_end=1565
  _DATAREQUESTLISTPROTO._serialized_start=1567
  _DATAREQUESTLISTPROTO._serialized_end=1631
  _SNAPSHOTID._serialized_start=1633
  _SNAPSHOTID._serialized_end=1660
  _RESTOREID._serialized_start=1662
  _RESTOREID._serialized_end=1688
  _SNAPSHOTSTATUSPROTO._serialized_start=1691
  _SNAPSHOTSTATUSPROTO._serialized_end=1930
  _SNAPSHOTSTATUSPROTO_STATUS._serialized_start=1817
  _SNAPSHOTSTATUSPROTO_STATUS._serialized_end=1930
  _RESTORESNAPSHOTSTATUSPROTO._serialized_start=1933
  _RESTORESNAPSHOTSTATUSPROTO._serialized_end=2135
  _RESTORESNAPSHOTSTATUSPROTO_STATUS._serialized_start=2049
  _RESTORESNAPSHOTSTATUSPROTO_STATUS._serialized_end=2135
  _RESTORESNAPSHOTCOMMAND._serialized_start=2137
  _RESTORESNAPSHOTCOMMAND._serialized_end=2184
  _JINADATAREQUESTRPC._serialized_start=2186
  _JINADATAREQUESTRPC._serialized_end=2276
  _JINASINGLEDATAREQUESTRPC._serialized_start=2278
  _JINASINGLEDATAREQUESTRPC._serialized_end=2377
  _JINARPC._serialized_start=2379
  _JINARPC._serialized_end=2450
  _JINADISCOVERENDPOINTSRPC._serialized_start=2452
  _JINADISCOVERENDPOINTSRPC._serialized_end=2548
  _JINAGATEWAYDRYRUNRPC._serialized_start=2550
  _JINAGATEWAYDRYRUNRPC._serialized_end=2628
  _JINAINFORPC._serialized_start=2630
  _JINAINFORPC._serialized_end=2701
  _JINAEXECUTORSNAPSHOT._serialized_start=2703
  _JINAEXECUTORSNAPSHOT._serialized_end=2790
  _JINAEXECUTORSNAPSHOTPROGRESS._serialized_start=2792
  _JINAEXECUTORSNAPSHOTPROGRESS._serialized_end=2888
  _JINAEXECUTORRESTORE._serialized_start=2890
  _JINAEXECUTORRESTORE._serialized_end=2988
  _JINAEXECUTORRESTOREPROGRESS._serialized_start=2990
  _JINAEXECUTORRESTOREPROGRESS._serialized_end=3090
# @@protoc_insertion_point(module_scope)