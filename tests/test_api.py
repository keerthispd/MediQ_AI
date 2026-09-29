from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_root():
    r = client.get('/')
    assert r.status_code == 200
    assert 'service' in r.json()


def test_health():
    r = client.get('/api/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'ok'
