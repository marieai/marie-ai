from pydantic import BaseModel


class ConnectionStatus(BaseModel):
    connecting = "CONNECTING"
    disconnected = "DISCONNECTED"
    read_write = "R/W"
    read_only = "RO"


class Config(BaseModel):
    user: str = None
    password: str = None
    etcd_host: str = "localhost"
    etcd_port: int = 2379
