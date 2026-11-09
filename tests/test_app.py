"""The whole app over HTTP with a stand-in for the language model: references, instant builders, a generation with
review rounds, editing, versions, exports and the agent tools."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from .test_core import sprite_png


class FakeLink:
    """Answers like the model: first round builds, second round corrects, third says done."""

    def __init__(self):
        self.calls = []
        self.sync = SimpleNamespace(chat=self.chat, resolve=self.resolve)

    def resolve(self, capability):
        return SimpleNamespace(state="resolved", url="http://spark:8888/v1/chat/completions", model="deepseek-v4.1-flash",
                               provider="test", api="link", reason="test")

    def chat(self, messages, images=None, **kw):
        self.calls.append({"messages": messages, "images": images or [], **kw})
        n = len(self.calls)
        if n == 1:
            answer = {"notes": "body and eyes", "palette_add": ["#123456"],
                      "ops": [{"op": "sphere", "center": [6.5, 6, 3], "radii": [6, 5, 3], "c": 1},
                              {"op": "box", "from": [99, 0, 0], "to": [100, 1, 1], "c": 2},          # outside: clipped, fine
                              {"op": "nonsense"}]}
            text = "Here it is:\n```json\n" + json.dumps(answer) + "\n```"
        elif n == 2:
            text = json.dumps({"notes": "feet", "ops": [{"op": "box", "from": [3, 0, 2], "to": [10, 1, 4], "c": 3}]})
        else:
            text = json.dumps({"done": True, "notes": "good", "ops": []})
        return SimpleNamespace(text=text, model="deepseek-v4.1-flash")


@pytest.fixture
def app_client(tmp_path):
    from fastapi.testclient import TestClient

    from seurat_hoard.config import Config
    from seurat_hoard.main import create_app
    from seurat_hoard.services import Services

    cfg = Config(data_dir=tmp_path / "data", port=5250, data_dir_configured=True, background=False)
    link = FakeLink()
    svc = Services(cfg, link_factory=lambda: link)
    app = create_app(cfg, services=svc)
    with TestClient(app, base_url="http://127.0.0.1:5250", client=("127.0.0.1", 50000)) as c:
        yield c, svc, link


def test_health_and_status(app_client):
    c, svc, _ = app_client
    assert c.get("/api/health").json()["service"] == "seurats-hoard"
    st = c.get("/api/status").json()
    assert st["llm"]["available"] is True and st["llm"]["model"] == "deepseek-v4.1-flash"
    assert c.get("/").status_code == 200


def test_reference_builders_and_generation(app_client):
    c, svc, link = app_client
    ref = c.post("/api/references?name=slime", content=sprite_png(), headers={"Content-Type": "image/png"}).json()
    assert ref["width"] == 64 and ref["name"] == "slime"
    assert c.get(f"/references/{ref['id']}").headers["content-type"] == "image/png"
    grid = c.get(f"/api/references/{ref['id']}/grid?max_side=32").json()
    assert grid["width"] == 14 and grid["rows"][0] == "....222222...."
    assert c.get(f"/api/references/{ref['id']}/indexed.png").status_code == 200

    built = c.post("/api/build", json={"reference_id": ref["id"], "mode": "inflate"}).json()
    assert built["count"] > 100 and built["size"][:2] == [14, 13]

    job = c.post("/api/generate", json={"references": [{"id": ref["id"], "role": "front"}], "prompt": "a cute slime",
                                        "rounds": 2, "name": "Slime voxel"}).json()
    st = c.get(f"/api/jobs/{job['job_id']}").json()
    assert st["status"] == "done", st
    assert [r["round"] for r in st["rounds"]] == [0, 1, 2, 3] and st["rounds"][1]["errors"]           # the nonsense op reported
    assert st["llm"] == "deepseek-v4.1-flash" and any("da el resultado por bueno" in line["text"] for line in st["log"])
    model = c.get(f"/api/models/{st['model_id']}").json()
    assert model["name"] == "Slime voxel" and "#123456" in model["palette"] and model["source"]["kind"] == "generated"
    assert c.get(f"/renders/{st['rounds'][1]['render']}").status_code == 200
    first, second = link.calls[0], link.calls[1]
    user = first["messages"][1]["content"]
    assert "plane front, origin [0,0]" in user and "....222222...." in user and "Palette (index(character)" in user
    assert len(first["images"]) == 1                                  # auto with the model: sculpted from an empty grid
    assert "Review round 2" in second["messages"][1]["content"] and "FAILED" in second["messages"][1]["content"]
    assert first["capability"] == "vision"


def test_generation_without_model_and_refine_current(app_client):
    c, svc, link = app_client
    ref = c.post("/api/references", content=sprite_png(), headers={"Content-Type": "image/png"}).json()
    job = c.post("/api/generate", json={"references": [{"id": ref["id"]}], "use_model": False, "start": "extrude"}).json()
    st = c.get(f"/api/jobs/{job['job_id']}").json()
    assert st["status"] == "done" and not link.calls and len(st["rounds"]) == 1
    mid = st["model_id"]
    job2 = c.post("/api/generate", json={"references": [{"id": ref["id"]}], "into": mid, "start": "current", "rounds": 0}).json()
    st2 = c.get(f"/api/jobs/{job2['job_id']}").json()
    assert st2["status"] == "done" and st2["model_id"] == mid
    assert c.get(f"/api/models/{mid}/versions").json()["versions"]                                 # the extrude is a version now
    bad = c.post("/api/generate", json={"references": [{"id": "r_nope"}]})
    assert bad.status_code == 404


def test_editor_round_trip_versions_and_exports(app_client, tmp_path):
    c, svc, _ = app_client
    m = c.post("/api/models", json={"name": "casa", "size": [8, 8, 8]}).json()
    mid = m["id"]
    c.put(f"/api/models/{mid}", json={"voxels": [[1, 0, 1, 3], [2, 0, 1, 4]], "palette": m["palette"][:10], "name": "casa"})
    got = c.get(f"/api/models/{mid}").json()
    assert got["voxels"] == [[1, 0, 1, 3], [2, 0, 1, 4]] and len(got["palette"]) == 10
    r = c.post(f"/api/models/{mid}/ops", json={"ops": [{"op": "box", "from": [0, 0, 0], "to": [7, 0, 7], "c": 2}]}).json()
    assert r["count"] == 64
    bad = c.post(f"/api/models/{mid}/ops", json={"ops": [{"op": "box", "from": [0, 0, 0]}]})
    assert bad.status_code == 400 and "Operation 0" in bad.json()["error"]
    versions = c.get(f"/api/models/{mid}/versions").json()["versions"]
    assert len(versions) >= 2
    restored = c.post(f"/api/models/{mid}/restore", json={"version": versions[0]["version"]}).json()
    assert restored["count"] == 2
    for fmt, magic in (("vox", b"VOX "), ("glb", b"glTF"), ("png", b"\x89PNG"), ("sprites", b"\x89PNG")):
        res = c.get(f"/api/models/{mid}/export/{fmt}")
        assert res.status_code == 200 and res.content[:4] == magic[:4] and "attachment" in res.headers["content-disposition"]
    assert c.get(f"/api/models/{mid}/export/stl").content[:6] == b"Seurat"
    imported = c.post("/api/models/import-vox?name=back", content=c.get(f"/api/models/{mid}/export/vox").content).json()
    assert imported["count"] == 2
    dup = c.post(f"/api/models/{mid}/duplicate").json()
    assert dup["id"] != mid
    assert c.delete(f"/api/models/{dup['id']}").json()["deleted"] == dup["id"]
    assert c.get(f"/api/models/{dup['id']}").status_code == 404


def test_agent_tools(app_client, tmp_path):
    c, svc, _ = app_client
    head = {"Authorization": "Bearer " + svc.token}
    tools = {t["name"] for t in c.get("/api/agent/tools").json()["tools"]}
    assert {"model_ops", "model_look", "generate_start", "model_export", "reference_add", "build_from_image"} <= tools
    pic = tmp_path / "ref.png"
    pic.write_bytes(sprite_png())
    call = lambda name, args: c.post("/api/agent/call", json={"name": name, "arguments": args}, headers=head).json()
    ref = call("reference_add", {"path": str(pic)})
    built = call("build_from_image", {"reference_id": ref["id"], "mode": "extrude", "depth": 2})
    look = call("model_look", {"model_id": built["id"], "view": "sheet"})
    assert look["_image"]["mime"] == "image/png" and len(look["_image"]["data"]) > 1000
    out = call("model_export", {"model_id": built["id"], "format": "stl", "folder": str(tmp_path / "out")})
    assert out["path"].endswith(".stl") and (tmp_path / "out").exists()
    ops = call("model_ops", {"model_id": built["id"], "ops": [{"op": "clear"}], "note": "wipe"})
    assert ops["count"] == 0
    denied = c.post("/api/agent/call", json={"name": "model_delete", "arguments": {"model_id": built["id"]}}, headers=head)
    assert denied.status_code == 400 and denied.json()["code"] == "confirm_required"
    assert c.post("/api/agent/call", json={"name": "status", "arguments": {}}).status_code == 401


def test_editor_autosaves_share_one_version(app_client):
    c, svc, _ = app_client
    mid = c.post("/api/models", json={"name": "v", "size": [4, 4, 4]}).json()["id"]
    c.post(f"/api/models/{mid}/ops", json={"ops": [{"op": "voxel", "at": [0, 0, 0], "c": 1}], "note": "before editing"})
    for i in range(5):
        c.put(f"/api/models/{mid}", json={"voxels": [[i % 4, 0, 0, 2]], "version_note": "editor"})
    notes = [v["note"] for v in c.get(f"/api/models/{mid}/versions").json()["versions"]]
    assert len(notes) <= 3, notes                       # empty start, the ops result, one editor run


def test_a_model_failure_keeps_the_starting_model(app_client):
    c, svc, link = app_client

    def broken(*a, **kw):
        raise RuntimeError("spark down")

    link.sync.chat = broken
    ref = c.post("/api/references", content=sprite_png(), headers={"Content-Type": "image/png"}).json()
    job = c.post("/api/generate", json={"references": [{"id": ref["id"]}], "rounds": 1}).json()
    st = c.get(f"/api/jobs/{job['job_id']}").json()
    assert st["status"] == "done" and st["model_id"] and "spark down" in st["warning"]
    assert c.get(f"/api/models/{st['model_id']}").json()["count"] > 100
