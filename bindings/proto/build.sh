
Create Python GRPC client and server stubs from proto file satysfying the following requirements:

-  seperate client and server in separate packages 
-  seperate build scripts for client and server
-  server pushing streaming events to the client 
-  server should be able to push events to multiple clients that are connected to the server and match the api_key
-  GRPC server needs to support authentication using api_key
-  GRPC server needs to validate api_key on connection and reject connection if api_key is invalid
-  client should not know about server and vice versa
-  client should only be able to call server using a single function call
-  client receiving streaming events using a callback function
-  client need to authenticate using api_key
-  api_key validation on connection to the server
-  Ensure re-connection is handled in both client and server implementation

Generate client in Python, Java and Go

# Path: bindings/proto/marieai.proto

syntax = "proto3";
package marieai;

message EventMessage {
  string api_key = 1;
  string job_id = 2;
  string event = 3;
  string job_tag = 4;
  string status = 5;
  int64 timestamp = 6;
}

# Path: bindings/proto/server.proto

syntax = "proto3";
package marieai.server;

import "marieai.proto";

message ApiKey {
  string key = 1;
}

service EventService {
  rpc Connect(EventMessage) returns (stream EventMessage) {}
}

# Path: bindings/proto/client.proto
syntax = "proto3";
package marieai.client;

import "marieai.proto";

service EventService {
  rpc Connect(AuthRequest) returns (stream EventMessage) {}
}

message AuthRequest {
  string api_key = 1;
}