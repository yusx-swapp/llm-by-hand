from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app


def test_public_entry_and_local_assets(tmp_path):
    with TestClient(create_app(tmp_path / "progress.sqlite3")) as client:
        assert "LLM by Hand" in client.get("/").text
        assert client.get("/api/health").json()["local_only"]
        for path in ("/static/course/main.js", "/static/ui.js", "/static/vendor/monaco/vs/loader.js"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/javascript")
        assert client.get("/docs").status_code == 404
        assert client.get("/legacy").status_code == 404
        assert client.get("/prototype").status_code == 404


def test_local_write_boundary(tmp_path):
    with TestClient(create_app(tmp_path / "progress.sqlite3")) as client:
        payload = {"stage": "follow", "source": "# local draft"}
        assert client.post("/api/course/lesson/mha_core/draft", json=payload,
                           headers={"Origin": "https://untrusted.example"}).status_code == 403
        assert client.post("/api/course/lesson/mha_core/draft", json=payload,
                           headers={"Origin": "http://testserver"}).status_code == 200
        assert client.get("/api/health", headers={"Host": "untrusted.example"}).status_code == 400


def test_release_keeps_licenses_and_excludes_historical_material():
    root = Path(__file__).resolve().parents[1]
    assert (root / "LICENSE").is_file()
    assert (root / "NOTICE").is_file()
    assert (root / "web/vendor/monaco/LICENSE").is_file()
    assert (root / "web/vendor/monaco/ThirdPartyNotices.txt").is_file()
    assert not (root / "questions").exists()
    assert not (root / "docs/research").exists()
    assert not (root / "web/learn").exists()
