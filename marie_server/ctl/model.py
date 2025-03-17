from pydantic import BaseModel


class ConnectionStatus(BaseModel):
    connected: str = "CONNECTED"
    connecting: str = "CONNECTING"
    disconnected: str = "DISCONNECTED"


class Config(BaseModel):
    user: str = None
    password: str = None
    etcd_host: str = "localhost"
    etcd_port: int = 2379
    service_name: str = "gateway/marie"
