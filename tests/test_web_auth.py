from app.web_auth import create_session, hash_password, verify_password, verify_session


def test_password_hashes_are_saltied_and_verifiable() -> None:
    first = hash_password("secret")
    second = hash_password("secret")
    assert first != second
    assert verify_password("secret", first)
    assert verify_password("secret", second)
    assert not verify_password("wrong", first)


def test_sessions_round_trip_and_expire() -> None:
    token = create_session("bob", "secret", ttl_seconds=100)
    assert verify_session(token, "secret") == "bob"
    assert verify_session(token, "wrong") is None
    expired = create_session("bob", "secret", ttl_seconds=-1)
    assert verify_session(expired, "secret") is None


def test_invalid_session_shapes_are_rejected() -> None:
    assert verify_session("not-a-token", "secret") is None
    assert verify_session(None, "secret") is None
