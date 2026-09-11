import pytest


@pytest.fixture(autouse=True)
def disable_default_access_policy(monkeypatch, tmp_path):
    """Point Settings at a missing policy so ACL tests opt in explicitly."""
    monkeypatch.setenv("ACCESS_POLICY_FILE", str(tmp_path / "no-policy.json"))
