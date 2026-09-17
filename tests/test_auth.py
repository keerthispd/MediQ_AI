from datetime import datetime

import pytest

from backend.models.db import SessionLocal
from backend.models.models import AuthToken, User
from backend.services.auth import hash_password, verify_password
from conftest import PASSWORD, get_history, register


def test_root_and_health(client):
    assert 'service' in client.get('/').json()
    assert client.get('/api/health').json() == {'status': 'ok'}


def test_register_login_logout(client):
    username, headers = register(client)
    assert client.get("/api/auth/me", headers=headers).json() == {"username": username}

    wrong = client.post("/api/auth/login", json={"username": username, "password": "wrong-password"})
    assert wrong.status_code == 401
    unknown = client.post("/api/auth/login", json={"username": "nobody-here", "password": PASSWORD})
    assert unknown.status_code == 401

    login = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    assert login.status_code == 200
    second = {"Authorization": f"Bearer {login.json()['token']}"}

    assert client.post("/api/auth/logout", headers=second).status_code == 200
    assert client.get("/api/auth/me", headers=second).status_code == 401
    # Logging out one session leaves the others alone
    assert client.get("/api/auth/me", headers=headers).status_code == 200


def test_register_rejects_taken_names_and_weak_input(client):
    username, _ = register(client)
    taken = client.post("/api/auth/register", json={"username": username, "password": PASSWORD})
    assert taken.status_code == 409

    short_password = client.post("/api/auth/register", json={"username": "newperson", "password": "short"})
    assert short_password.status_code == 422
    bad_name = client.post("/api/auth/register", json={"username": "../admin", "password": PASSWORD})
    assert bad_name.status_code == 422


@pytest.mark.parametrize("method, path", [
    ("post", "/api/chat"),
    ("post", "/api/upload"),
    ("get", "/api/history"),
    ("delete", "/api/history"),
    ("get", "/api/auth/me"),
])
def test_endpoints_require_login(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    bogus = {"Authorization": "Bearer not-a-real-token"}
    assert getattr(client, method)(path, headers=bogus).status_code == 401


def test_expired_token_is_rejected(client):
    username, headers = register(client)
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).one()
        db.query(AuthToken).filter(AuthToken.user_id == user.id).update({AuthToken.expires_at: datetime(2000, 1, 1)})
        db.commit()
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_history_is_private_to_each_user(client, fake_llm):
    _, alice = register(client)
    _, bob = register(client)
    client.post("/api/chat", data={"message": "alice's question"}, headers=alice)
    client.post("/api/chat", data={"message": "bob's question"}, headers=bob)

    assert [it["user_message"] for it in get_history(client, alice)] == ["alice's question"]

    assert client.delete("/api/history", headers=alice).json() == {"status": "cleared"}
    assert get_history(client, alice) == []
    assert [it["user_message"] for it in get_history(client, bob)] == ["bob's question"]


def test_password_hashing():
    stored = hash_password("s3cret-pass")
    assert stored != "s3cret-pass"
    assert verify_password("s3cret-pass", stored)
    assert not verify_password("other-pass", stored)
    assert not verify_password("s3cret-pass", "garbage")
