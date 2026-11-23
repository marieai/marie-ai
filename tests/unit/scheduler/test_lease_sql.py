from pathlib import Path


def test_lease_jobs_disables_jit() -> None:
    project_root = Path(__file__).resolve().parents[3]
    schema_sql = project_root.joinpath(
        "config/psql/schema/lease/001_lease_jobs_by_id.sql"
    ).read_text()

    assert "SET jit = off" in schema_sql


def test_each_bundle_ships_one_matching_activation_definition() -> None:
    root = Path(__file__).resolve().parents[3]
    canonical = root / "config/psql/schema/lease/002_activate_from_lease.sql"
    helm = root / "deploy/helm/charts/marie/charts/server/files/psql/schema/lease"
    for directory in (canonical.parent, helm):
        assert list(directory.glob("*_activate_from_lease.sql")) == [
            directory / canonical.name
        ]
    assert (helm / canonical.name).read_bytes() == canonical.read_bytes()
