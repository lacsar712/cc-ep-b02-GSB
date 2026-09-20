import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# JSONB is PostgreSQL-only; compile it as plain JSON for the SQLite test DB.
@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(_type, _compiler, **_kw):
    return "JSON"

from app.auth import create_access_token  # noqa: E402
from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session):
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    # Deliberately not used as a context manager: that would trigger the
    # lifespan, which runs create_all against PostgreSQL. The SQLite session
    # from db_session serves every request via the get_db override.
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def researcher_headers():
    return auth(create_access_token("researcher", "researcher"))


@pytest.fixture()
def auditor_headers():
    return auth(create_access_token("auditor", "auditor"))


def make_run(client, headers, *, name: str, project: str = "p1"):
    resp = client.post(
        "/api/runs",
        headers=headers,
        json={
            "project": project,
            "name": name,
            "dataset_content_sha256": sha(name),
            "code_commit_sha": "abc1234",
            "description": None,
            "expected_version": 0,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def abort(client, headers, run_id, reason, expected_version):
    return client.post(
        f"/api/runs/{run_id}/abort",
        headers=headers,
        json={"reason": reason, "expected_version": expected_version},
    )


# --- Requirement 1: blank / whitespace-only reason is rejected -------------

@pytest.mark.parametrize("blank", ["", "   ", "\t\n  \t"])
def test_abort_rejects_blank_reason(client, researcher_headers, blank):
    run = make_run(client, researcher_headers, name="blank-run")
    resp = abort(client, researcher_headers, run["id"], blank, 1)
    assert resp.status_code == 422, resp.text

    # Run must remain running and carry no abort reason.
    fetched = client.get(f"/api/runs/{run['id']}", headers=researcher_headers).json()
    assert fetched["status"] == "running"
    assert fetched["abort_reason"] is None
    assert fetched["finished_at"] is None


def test_abort_strips_surrounding_whitespace_and_records_time(client, researcher_headers):
    run = make_run(client, researcher_headers, name="trim-run")
    resp = abort(client, researcher_headers, run["id"], "  GPU OOM killed  ", 1)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "aborted"
    assert body["abort_reason"] == "GPU OOM killed"
    assert body["finished_at"] is not None  # abort time displayed

    # Reason is persisted into the event payload (traceability).
    events = client.get(f"/api/runs/{run['id']}/events", headers=researcher_headers).json()
    abort_events = [e for e in events if e["event_type"] == "RunAborted"]
    assert len(abort_events) == 1
    assert abort_events[0]["payload_json"]["reason"] == "GPU OOM killed"
    assert abort_events[0]["occurred_at"] is not None


# --- Requirement 2: server-side status + reason keyword narrowing ----------

def test_keyword_filter_returns_only_matching_aborted_run(client, researcher_headers):
    target = make_run(client, researcher_headers, name="target")
    other_aborted = make_run(client, researcher_headers, name="other-aborted")
    running = make_run(client, researcher_headers, name="still-running")

    assert abort(client, researcher_headers, target["id"], "GPU OOM during training", 1).status_code == 200
    assert abort(client, researcher_headers, other_aborted["id"], "training data corrupted", 1).status_code == 200
    # `running` left in progress on purpose.

    # Step 1: filter to aborted runs only.
    aborted = client.get("/api/runs", headers=researcher_headers, params={"status": "aborted"}).json()
    aborted_ids = {r["id"] for r in aborted}
    assert target["id"] in aborted_ids
    assert other_aborted["id"] in aborted_ids
    assert running["id"] not in aborted_ids

    # Step 2: narrow by a keyword from the target's reason, server-side.
    narrowed = client.get(
        "/api/runs",
        headers=researcher_headers,
        params={"status": "aborted", "reason_kw": "oom"},
    ).json()
    narrowed_ids = {r["id"] for r in narrowed}
    assert narrowed_ids == {target["id"]}

    # Case-insensitive and whitespace tolerant.
    upper = client.get(
        "/api/runs",
        headers=researcher_headers,
        params={"status": "aborted", "reason_kw": "  OOM  "},
    ).json()
    assert {r["id"] for r in upper} == {target["id"]}


def test_keyword_without_status_still_filters(client, researcher_headers):
    r1 = make_run(client, researcher_headers, name="k1")
    r2 = make_run(client, researcher_headers, name="k2")
    abort(client, researcher_headers, r1["id"], "nan loss explosion", 1)
    abort(client, researcher_headers, r2["id"], "manual cancellation", 1)

    hits = client.get("/api/runs", headers=researcher_headers, params={"reason_kw": "nan"}).json()
    assert {r["id"] for r in hits} == {r1["id"]}


# --- Requirement 3: auditor sees reason but cannot abort -------------------

def test_auditor_reads_reason_but_abort_is_forbidden(client, researcher_headers, auditor_headers):
    run = make_run(client, researcher_headers, name="audit-me")
    abort(client, researcher_headers, run["id"], "confidential audit reason", 1)

    seen = client.get(f"/api/runs/{run['id']}", headers=auditor_headers)
    assert seen.status_code == 200
    assert seen.json()["abort_reason"] == "confidential audit reason"

    denied = abort(client, auditor_headers, run["id"], "auditor tries", 1)
    assert denied.status_code == 403


# --- Requirement 4: a finished Run cannot be aborted -----------------------

def test_completed_run_cannot_be_aborted(client, researcher_headers):
    run = make_run(client, researcher_headers, name="done-run")
    done = client.post(
        f"/api/runs/{run['id']}/complete",
        headers=researcher_headers,
        json={"result_summary": "finished cleanly", "expected_version": 1},
    )
    assert done.status_code == 200, done.text

    resp = abort(client, researcher_headers, run["id"], "too late", 2)
    assert resp.status_code == 409, resp.text

    fetched = client.get(f"/api/runs/{run['id']}", headers=researcher_headers).json()
    assert fetched["status"] == "completed"
    assert fetched["abort_reason"] is None


def test_aborted_run_cannot_be_aborted_again(client, researcher_headers):
    run = make_run(client, researcher_headers, name="double-abort")
    assert abort(client, researcher_headers, run["id"], "first reason", 1).status_code == 200
    resp = abort(client, researcher_headers, run["id"], "second reason", 2)
    assert resp.status_code == 409, resp.text
