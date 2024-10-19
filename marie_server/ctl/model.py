from pydantic import BaseModel


class ConnectionStatus(BaseModel):
    connecting: str = "CONNECTING"
    disconnected: str = "DISCONNECTED"
    read_write: str = "R/W"
    read_only: str = "RO"


class Config(BaseModel):
    user: str = None
    password: str = None
    etcd_host: str = "localhost"
    etcd_port: int = 2379
