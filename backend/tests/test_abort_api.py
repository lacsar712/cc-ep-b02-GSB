import hashlib

import pytest
from fastapi.testclient import TestClient


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def login(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture()
def auth_headers(client):
    researcher = login(client, "researcher", "lab123456")
    auditor = login(client, "auditor", "audit123456")
    return {
        "researcher": {"Authorization": f"Bearer {researcher}"},
        "auditor": {"Authorization": f"Bearer {auditor}"},
    }


def create_run(client: TestClient, headers: dict, *, project: str = "p1", name: str = "n1") -> dict:
    resp = client.post(
        "/api/runs",
        headers=headers,
        json={
            "project": project,
            "name": name,
            "dataset_content_sha256": sha(name),
            "code_commit_sha": "abc1234",
            "description": "test run",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def abort(client: TestClient, run_id: str, headers: dict, reason: str, expected_version: int = 1):
    return client.post(
        f"/api/runs/{run_id}/abort",
        headers=headers,
        json={"reason": reason, "expected_version": expected_version},
    )


def test_abort_rejects_blank_reason(client, auth_headers):
    headers = auth_headers["researcher"]
    run = create_run(client, headers, name="blank-reason-run")

    # 纯空白原因必须被拒绝
    resp = abort(client, run["id"], headers, "   ")
    assert resp.status_code == 422, resp.text

    # 空字符串同样拒绝
    resp = abort(client, run["id"], headers, "")
    assert resp.status_code == 422, resp.text

    # 拒绝后 Run 仍是 running，且没有产生任何 RunAborted 事件
    detail = client.get(f"/api/runs/{run['id']}", headers=headers).json()
    assert detail["status"] == "running"
    assert detail["abort_reason"] is None

    events = client.get(f"/api/runs/{run['id']}/events", headers=headers).json()
    assert all(e["event_type"] != "RunAborted" for e in events)


def test_abort_keyword_filter_server_side(client, auth_headers):
    headers = auth_headers["researcher"]
    run_oom = create_run(client, headers, project="gpu", name="oom-run")
    run_data = create_run(client, headers, project="data", name="data-run")
    run_running = create_run(client, headers, project="gpu", name="still-running")

    assert abort(client, run_oom["id"], headers, "GPU 显存溢出导致训练中断").status_code == 200
    assert abort(client, run_data["id"], headers, "数据校验失败，样本标签缺失").status_code == 200

    # 先筛“已中止”：返回两条已中止的，进行中的不在内
    resp = client.get("/api/runs", headers=headers, params={"status": "aborted"})
    assert resp.status_code == 200
    aborted = resp.json()
    aborted_ids = {r["id"] for r in aborted}
    assert run_oom["id"] in aborted_ids
    assert run_data["id"] in aborted_ids
    assert run_running["id"] not in aborted_ids

    # 再用原因里的词在服务端收窄：只留下 OOM 这一条
    resp = client.get(
        "/api/runs",
        headers=headers,
        params={"status": "aborted", "abort_reason": "显存溢出"},
    )
    assert resp.status_code == 200
    narrowed = resp.json()
    assert [r["id"] for r in narrowed] == [run_oom["id"]]

    # 不传状态、只传关键字同样只命中这一条
    resp = client.get("/api/runs", headers=headers, params={"abort_reason": "训练中断"})
    assert [r["id"] for r in resp.json()] == [run_oom["id"]]

    # 匹配不到时返回空列表
    resp = client.get(
        "/api/runs",
        headers=headers,
        params={"status": "aborted", "abort_reason": "不存在的原因关键字"},
    )
    assert resp.json() == []

    # LIKE 通配符被转义：% 不应匹配全部
    resp = client.get(
        "/api/runs",
        headers=headers,
        params={"status": "aborted", "abort_reason": "%"},
    )
    assert resp.json() == []


def test_auditor_sees_reason_but_cannot_abort(client, auth_headers):
    researcher = auth_headers["researcher"]
    auditor = auth_headers["auditor"]
    run = create_run(client, researcher, name="auditor-visibility")
    reason = "人工复核发现指标异常，主动中止"
    assert abort(client, run["id"], researcher, reason).status_code == 200

    # 审计员在列表与详情中看得见中止原因
    resp = client.get(
        "/api/runs",
        headers=auditor,
        params={"status": "aborted", "abort_reason": "指标异常"},
    )
    assert resp.status_code == 200
    visible = resp.json()
    assert len(visible) == 1
    assert visible[0]["id"] == run["id"]
    assert visible[0]["abort_reason"] == reason

    detail = client.get(f"/api/runs/{run['id']}", headers=auditor)
    assert detail.status_code == 200
    assert detail.json()["abort_reason"] == reason

    # 审计员没有中止权限（前端不显示按钮，后端同样拒绝）
    resp = client.post(
        f"/api/runs/{run['id']}/abort",
        headers=auditor,
        json={"reason": "审计员尝试中止", "expected_version": 2},
    )
    assert resp.status_code == 403


def test_cannot_abort_finished_run(client, auth_headers):
    headers = auth_headers["researcher"]
    run = create_run(client, headers, name="finished-run")

    resp = client.post(
        f"/api/runs/{run['id']}/complete",
        headers=headers,
        json={"result_summary": "正常完成", "expected_version": 1},
    )
    assert resp.status_code == 200, resp.text

    # 已结束（completed）的 Run 不能再中止
    resp = abort(client, run["id"], headers, "结束后反悔", expected_version=2)
    assert resp.status_code == 409, resp.text

    detail = client.get(f"/api/runs/{run['id']}", headers=headers).json()
    assert detail["status"] == "completed"
    assert detail["abort_reason"] is None


def test_abort_writes_reason_and_time_to_event_and_detail(client, auth_headers):
    headers = auth_headers["researcher"]
    run = create_run(client, headers, name="traceable-abort")
    reason = "NaN loss at step 42"

    resp = abort(client, run["id"], headers, reason)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "aborted"
    assert body["abort_reason"] == reason
    # 详情中可见中止时间
    assert body["finished_at"] is not None

    events = client.get(f"/api/runs/{run['id']}/events", headers=headers).json()
    abort_events = [e for e in events if e["event_type"] == "RunAborted"]
    assert len(abort_events) == 1
    event = abort_events[0]
    # 原因写入事件 payload，事件携带发生时间
    assert event["payload_json"]["reason"] == reason
    assert event["occurred_at"] is not None
