import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

channel = grpc.insecure_channel('localhost:52000')
stub = health_pb2_grpc.HealthStub(channel)

# Set a longer deadline (e.g., 10 seconds)
timeout = 1
health_check_req = health_pb2.HealthCheckRequest()
health_check_req.service = ''
stub = health_pb2_grpc.HealthStub(channel)

status = stub.Check(health_check_req, timeout=timeout)
print(status)
