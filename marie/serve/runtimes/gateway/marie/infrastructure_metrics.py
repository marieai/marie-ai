import hashlib
import re


def configured_metric_fabric(*candidates: str | None) -> str:
    """Require an explicit, consistent fabric identity for exported telemetry."""
    values = [str(value) for value in candidates if value is not None and value != ""]
    if not values or len(set(values)) != 1:
        return ""
    return values[0] if re.fullmatch(r"[a-z0-9_-]{1,128}", values[0]) else ""


def gateway_metric_attributes(
    fabric_group_id: str,
    gateway_id: str | None,
    instance_id: str,
    started_at: float,
) -> dict[str, str]:
    """Identify a configured gateway and its current process in metric samples."""
    return {
        "fabric_group_id": (
            fabric_group_id
            if re.fullmatch(r"[a-z0-9_-]{1,128}", fabric_group_id)
            else ""
        ),
        "gateway_id": (
            gateway_id
            if gateway_id and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", gateway_id)
            else ""
        ),
        "gateway_instance_id": instance_id,
        "gateway_started_at": str(started_at),
    }


def capacity_pool_identity(cluster_id: int, namespace: str) -> str:
    """Identify the shared semaphore store without exporting connection details."""
    if cluster_id <= 0:
        raise ValueError("Etcd cluster identity is unavailable")
    return hashlib.sha256(f"{cluster_id}:/{namespace}/".encode()).hexdigest()
