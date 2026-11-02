import http.server
import json
import threading
import typing
import urllib.error
import urllib.parse

import pytest

from arena_assets.impl.fetch import Bucket, outermost

API = "https://storage.googleapis.com/storage/v1/b/bkt/o"


@pytest.fixture(autouse=True)
def _default_endpoint(monkeypatch):
    monkeypatch.delenv("ARENA_ASSETS_ENDPOINT", raising=False)


def test_endpoint_override_reaches_every_url(monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_ENDPOINT", "http://127.0.0.1:9")
    bucket = Bucket("bkt")
    assert bucket._object_url("") == "http://127.0.0.1:9/storage/v1/b/bkt/o"
    assert bucket._upload_url("a").startswith("http://127.0.0.1:9/upload/")


def test_object_url_bare():
    assert Bucket("bkt")._object_url("") == API


def test_object_url_params():
    assert Bucket("bkt")._object_url("", prefix="a/") == f"{API}?prefix=a%2F"


def test_object_url_encodes_object():
    assert Bucket("bkt")._object_url("dir/file.yaml", fields="name") == f"{API}/dir%2Ffile.yaml?fields=name"


def test_upload_url_encodes_object():
    assert Bucket("bkt")._upload_url("dir/file.yaml") == "https://storage.googleapis.com/upload/storage/v1/b/bkt/o?uploadType=media&name=dir%2Ffile.yaml"


def test_request_carries_token():
    req = Bucket("bkt", token="tok")._request(API)
    assert req.get_header("Authorization") == "Bearer tok"


def test_request_anonymous_without_token():
    req = Bucket("bkt")._request(API)
    assert not req.has_header("Authorization")


def test_outermost_drops_link_bundles_of_articulated_objects():
    listed = ["Common/Object/Fridge", "Common/Object/Fridge/links/door", "Common/Object/Fridge/links/base", "Common/Object/Chair", "Common/Object/Fridge_B"]
    assert outermost(listed) == ["Common/Object/Fridge", "Common/Object/Chair", "Common/Object/Fridge_B"]


class _PrivateBucket(http.server.BaseHTTPRequestHandler):
    """GCS JSON API stand-in that only answers the token `good`."""

    names: typing.ClassVar[list[str]] = ["w1/world.yaml", "w2/0/world.yaml", "w2/1/world.yaml", "world.yaml", "Common/Object/a/annotation.yaml", "recordings/take.mp4"]
    seen: typing.ClassVar[list[str | None]] = []

    def _reply(self) -> None:
        auth = self.headers.get("Authorization")
        type(self).seen.append(auth)
        if auth != "Bearer good":
            self.send_error(401 if auth else 403)
            return
        url = urllib.parse.urlparse(self.path)
        if "/o/" in url.path or "download" in url.path:
            self.send_error(404)
            return
        query = urllib.parse.parse_qs(url.query)
        names = self.names
        if "matchGlob" in query:
            suffix = query["matchGlob"][0].removeprefix("**/")
            names = [n for n in names if n.rsplit("/", 1)[-1] == suffix]
        body = json.dumps({"items": [{"name": n} for n in names]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _reply
    do_HEAD = _reply

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def private_bucket(monkeypatch, tmp_path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PrivateBucket)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _PrivateBucket.seen = []
    monkeypatch.setenv("ARENA_ASSETS_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.delenv("GCS_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    yield _PrivateBucket
    server.shutdown()
    server.server_close()


def test_refused_read_retries_with_the_environment_token(private_bucket, monkeypatch):
    monkeypatch.setenv("GCS_ACCESS_TOKEN", "good")
    assert len(Bucket("bkt").listdir("")) == len(private_bucket.names)
    assert private_bucket.seen == [None, "Bearer good"]


def test_token_sticks_after_the_first_refusal(private_bucket, monkeypatch):
    monkeypatch.setenv("GCS_ACCESS_TOKEN", "good")
    bucket = Bucket("bkt")
    bucket.listdir("")
    bucket.listdir("")
    assert private_bucket.seen == [None, "Bearer good", "Bearer good"]


def test_refused_read_without_a_token_raises_the_refusal(private_bucket):
    with pytest.raises(urllib.error.HTTPError) as refused:
        Bucket("bkt").listdir("")
    assert refused.value.code == 403


def test_invalid_token_raises_after_one_retry(private_bucket, monkeypatch):
    monkeypatch.setenv("GCS_ACCESS_TOKEN", "stale")
    with pytest.raises(urllib.error.HTTPError) as refused:
        Bucket("bkt").listdir("")
    assert refused.value.code == 401
    assert private_bucket.seen == [None, "Bearer stale"]


def test_explicit_token_is_not_replaced_on_refusal(private_bucket, monkeypatch):
    monkeypatch.setenv("GCS_ACCESS_TOKEN", "good")
    with pytest.raises(urllib.error.HTTPError):
        Bucket("bkt", token="stale").listdir("")
    assert private_bucket.seen == ["Bearer stale"]


def test_missing_object_stays_false_on_a_private_bucket(private_bucket, monkeypatch):
    monkeypatch.setenv("GCS_ACCESS_TOKEN", "good")
    assert Bucket("bkt").object_exists("absent.yaml") is False


def test_refused_download_retries_with_the_environment_token(private_bucket, monkeypatch, tmp_path):
    monkeypatch.setenv("GCS_ACCESS_TOKEN", "good")
    with pytest.raises(urllib.error.HTTPError) as missing:
        Bucket("bkt").download("absent.bin", str(tmp_path / "out" / "absent.bin"))
    assert missing.value.code == 404
    assert private_bucket.seen == [None, "Bearer good"]


def test_listchildren_containing_keeps_only_children_with_that_file(private_bucket, monkeypatch):
    monkeypatch.setenv("GCS_ACCESS_TOKEN", "good")
    assert Bucket("bkt").listchildren("", containing="world.yaml") == ["w1", "w2"]
