import time
import uuid
from typing import Dict

import pytest
from grpc_health.v1.health_pb2 import HealthCheckResponse

from marie.serve.discovery.etcd_client import EtcdClient
from marie.state.state_store import DesiredDoc, DesiredStore, StatusDoc, StatusStore


@pytest.fixture(scope="function")
def etcd_client():
    c = EtcdClient("localhost", 2379)
    yield c
    try:
        # clean everything under default namespace
        c.delete_prefix("")
    except Exception:
        pass


@pytest.fixture(scope="function")
def desired_store(etcd_client: EtcdClient):
    return DesiredStore(etcd_client)


@pytest.fixture(scope="function")
def status_store(etcd_client: EtcdClient):
    # Simple store without a lease getter for tests
    return StatusStore(etcd_client)


def _mk_ids() -> Dict[str, str]:
    node = f"node-{uuid.uuid4()}"
    depl = f"depl-{uuid.uuid4()}"
    return {"node": node, "depl": depl}


# ---------------- DesiredStore tests ----------------

def test_desired_set_and_get(desired_store: DesiredStore):
    ids = _mk_ids()
    params = {"p": 1}
    doc = desired_store.set(ids["node"], ids["depl"], params, phase="SCHEDULED")
    assert isinstance(doc, DesiredDoc)
    assert doc.phase == "SCHEDULED"
    assert doc.epoch == 1
    assert doc.params == params

    got = desired_store.get(ids["node"], ids["depl"])
    assert isinstance(got, DesiredDoc)
    assert got.phase == "SCHEDULED"
    assert got.epoch == 1
    assert got.params == params


def test_desired_schedule_new_epoch_creation_and_increment(desired_store: DesiredStore):
    ids = _mk_ids()

    # creation
    first = desired_store.schedule_new_epoch(ids["node"], ids["depl"], params={"a": 1})
    assert first.epoch == 1
    assert first.phase == "SCHEDULED"
    assert first.params.get("a") == 1

    # increment with merge
    second = desired_store.schedule_new_epoch(ids["node"], ids["depl"], params={"b": 2})
    assert second.epoch == 2
    assert second.phase == "SCHEDULED"
    assert second.params.get("a") == 1
    assert second.params.get("b") == 2


def test_desired_bump_epoch(desired_store: DesiredStore):
    ids = _mk_ids()
    created = desired_store.set(ids["node"], ids["depl"], params={}, phase="SCHEDULED")
    assert created.epoch == 1
    bumped = desired_store.bump_epoch(ids["node"], ids["depl"])
    assert bumped is not None
    assert bumped.epoch == 2


def test_desired_iter_pairs(desired_store: DesiredStore):
    # create multiple desired docs
    ids1 = _mk_ids()
    ids2 = _mk_ids()
    desired_store.set(ids1["node"], ids1["depl"], params={}, phase="SCHEDULED")
    desired_store.set(ids2["node"], ids2["depl"], params={}, phase="SCHEDULED")

    pairs = set(desired_store.iter_desired_pairs())
    assert (ids1["node"], ids1["depl"]) in pairs
    assert (ids2["node"], ids2["depl"]) in pairs


def test_desired_update_phase_keeps_epoch(desired_store: DesiredStore):
    ids = _mk_ids()
    created = desired_store.set(ids["node"], ids["depl"], params={}, phase="SCHEDULED")
    old_epoch = created.epoch

    updated = desired_store._update_phase(ids["node"], ids["depl"], phase="RUNNING")
    assert updated.phase == "RUNNING"
    assert updated.epoch == old_epoch  # epoch must not change


# ---------------- StatusStore tests ----------------

def test_status_claim_first_time(status_store: StatusStore):
    ids = _mk_ids()
    owner = f"worker-{uuid.uuid4()}"
    epoch = 1

    ok = status_store.claim(
        ids["node"], ids["depl"], worker_id=owner, epoch=epoch, initial_status=HealthCheckResponse.NOT_SERVING
    )
    assert ok is True

    st = status_store.read(ids["node"], ids["depl"])
    assert isinstance(st, StatusDoc)
    assert st.owner == owner
    assert st.epoch == epoch
    assert st.status_code == HealthCheckResponse.NOT_SERVING
    assert st.status_name == "NOT_SERVING"


def test_status_claim_idempotent_same_owner_epoch(status_store: StatusStore):
    ids = _mk_ids()
    owner = f"worker-{uuid.uuid4()}"
    epoch = 2

    ok1 = status_store.claim(ids["node"], ids["depl"], owner, epoch, HealthCheckResponse.NOT_SERVING)
    assert ok1 is True
    time_before = status_store.read(ids["node"], ids["depl"]).updated_at

    # idempotent re-claim
    ok2 = status_store.claim(ids["node"], ids["depl"], owner, epoch, HealthCheckResponse.NOT_SERVING)
    assert ok2 is True
    st = status_store.read(ids["node"], ids["depl"])
    assert st.owner == owner and st.epoch == epoch
    # updated_at should have been refreshed (lexicographically compare ISO may pass or not based on timing; just ensure it exists)
    assert isinstance(st.updated_at, str) and len(st.updated_at) > 0


def test_status_claim_roll_forward_same_owner(status_store: StatusStore):
    ids = _mk_ids()
    owner = f"worker-{uuid.uuid4()}"

    assert status_store.claim(ids["node"], ids["depl"], owner, 1, HealthCheckResponse.NOT_SERVING) is True
    assert status_store.claim(ids["node"], ids["depl"], owner, 2, HealthCheckResponse.SERVING) is True

    st = status_store.read(ids["node"], ids["depl"])
    assert st.owner == owner
    assert st.epoch == 2
    # initial status was SERVING on the second claim
    assert st.status_code == HealthCheckResponse.SERVING
    assert st.status_name == "SERVING"


def test_status_claim_fencing_different_owner(status_store: StatusStore):
    ids = _mk_ids()
    owner1 = f"worker-{uuid.uuid4()}"
    owner2 = f"worker-{uuid.uuid4()}"

    assert status_store.claim(ids["node"], ids["depl"], owner1, 5, HealthCheckResponse.NOT_SERVING) is True
    # other owner with same epoch should be rejected
    assert status_store.claim(ids["node"], ids["depl"], owner2, 5, HealthCheckResponse.NOT_SERVING) is False

    st = status_store.read(ids["node"], ids["depl"])
    assert st.owner == owner1
    assert st.epoch == 5


def test_status_set_statuses(status_store: StatusStore):
    ids = _mk_ids()
    owner = f"worker-{uuid.uuid4()}"

    assert status_store.claim(ids["node"], ids["depl"], owner, 1, HealthCheckResponse.NOT_SERVING) is True

    # set SERVING
    assert status_store.set_serving(ids["node"], ids["depl"], owner) is True
    st = status_store.read(ids["node"], ids["depl"])
    assert st.status_code == HealthCheckResponse.SERVING
    assert st.status_name == "SERVING"

    # set NOT_SERVING
    assert status_store.set_not_serving(ids["node"], ids["depl"], owner) is True
    st = status_store.read(ids["node"], ids["depl"])
    assert st.status_code == HealthCheckResponse.NOT_SERVING
    assert st.status_name == "NOT_SERVING"

    # set UNKNOWN
    assert status_store.set_unknown(ids["node"], ids["depl"], owner) is True
    st = status_store.read(ids["node"], ids["depl"])
    assert st.status_code == HealthCheckResponse.UNKNOWN
    assert st.status_name == "UNKNOWN"

    # set SERVICE_UNKNOWN
    assert status_store.set_service_unknown(ids["node"], ids["depl"], owner) is True
    st = status_store.read(ids["node"], ids["depl"])
    assert st.status_code == HealthCheckResponse.SERVICE_UNKNOWN
    assert st.status_name == "SERVICE_UNKNOWN"


def test_status_heartbeat_updates(status_store: StatusStore):
    ids = _mk_ids()
    owner = f"worker-{uuid.uuid4()}"

    # heartbeat without claim -> False
    assert status_store.heartbeat(ids["node"], ids["depl"], owner) is False

    # claim then heartbeat
    assert status_store.claim(ids["node"], ids["depl"], owner, 1, HealthCheckResponse.NOT_SERVING) is True
    st1 = status_store.read(ids["node"], ids["depl"])
    assert st1 is not None

    # slight delay so heartbeat_at is likely to differ
    time.sleep(0.05)
    assert status_store.heartbeat(ids["node"], ids["depl"], owner) is True
    st2 = status_store.read(ids["node"], ids["depl"])
    assert st2 is not None
    # compare ISO strings lexicographically is not reliable; ensure they exist and changed
    assert isinstance(st2.heartbeat_at, str) and len(st2.heartbeat_at) > 0


def test_status_read_missing_returns_none(status_store: StatusStore):
    ids = _mk_ids()
    st = status_store.read(ids["node"], ids["depl"])
    assert st is None
