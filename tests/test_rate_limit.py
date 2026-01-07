import time
from types import SimpleNamespace

import main


def mk_request(ip="127.0.0.1"):
    return SimpleNamespace(client=SimpleNamespace(host=ip))


def reset_inmemory_buckets():
    main.rate_buckets.clear()


def test_inmemory_rate_limit_exceed(monkeypatch):
    # Force in-memory by removing redis client
    monkeypatch.setattr(main, "redis_client", None, raising=False)
    # small limits
    monkeypatch.setenv("RL_PER_IP_PER_MIN", "3")
    monkeypatch.setenv("RL_PER_USER_PER_MIN", "5")
    reset_inmemory_buckets()

    req = mk_request("10.0.0.1")
    # 3 allowed, 4th should raise 429 for IP
    for i in range(3):
        main.enforce_rate_limit(req, user_id="u1")
    try:
        main.enforce_rate_limit(req, user_id="u1")
        assert False, "Expected rate limit exceed"
    except Exception as e:
        from fastapi import HTTPException
        assert isinstance(e, HTTPException)
        assert e.status_code == 429


def test_redis_rate_limit(monkeypatch):
    # Fake redis client and pipeline
    class FakePipe:
        def __init__(self, store):
            self.store = store
            self.cmds = []
        def incr(self, key):
            self.cmds.append(("incr", key))
            return self
        def expire(self, key, ttl):
            self.cmds.append(("expire", key, ttl))
            return self
        def execute(self):
            out = []
            for c in self.cmds:
                if c[0] == "incr":
                    key = c[1]
                    self.store[key] = self.store.get(key, 0) + 1
                    out.append(self.store[key])
                elif c[0] == "expire":
                    out.append(True)
            self.cmds = []
            return out
    class FakeRedis:
        def __init__(self):
            self.store = {}
        def pipeline(self):
            return FakePipe(self.store)
        def ping(self):
            return True

    fake = FakeRedis()
    monkeypatch.setattr(main, "redis_client", fake, raising=False)
    # Set tight limits
    monkeypatch.setenv("RL_PER_IP_PER_MIN", "2")
    monkeypatch.setenv("RL_PER_USER_PER_MIN", "2")

    req = mk_request("10.0.0.2")
    # First two pass
    main.enforce_rate_limit(req, user_id="u2")
    main.enforce_rate_limit(req, user_id="u2")
    # Third should exceed IP limit
    try:
        main.enforce_rate_limit(req, user_id="u2")
        assert False, "Expected redis rate limit exceed"
    except Exception as e:
        from fastapi import HTTPException
        assert isinstance(e, HTTPException)
        assert e.status_code == 429
