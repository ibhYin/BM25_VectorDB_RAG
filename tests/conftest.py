import os

import pytest

# Never read a developer's real runtime overrides during collection; tests that
# exercise persistence point this at tmp_path themselves.
os.environ.setdefault("RUNTIME_CONFIG_FILE", os.devnull)


@pytest.fixture(autouse=True)
def disable_default_access_policy(monkeypatch, tmp_path):
    """Point Settings at a missing policy so ACL tests opt in explicitly."""
    monkeypatch.setenv("ACCESS_POLICY_FILE", str(tmp_path / "no-policy.json"))
    monkeypatch.setenv("RUNTIME_CONFIG_FILE", str(tmp_path / "runtime_config.json"))
