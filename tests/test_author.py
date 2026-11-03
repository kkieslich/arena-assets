import pytest

from arena_assets.impl.author import author_database
from arena_assets.impl.fetch import StoreError
from arena_assets.utils.logging import initialize


@pytest.fixture
def bucket(store, monkeypatch):
    initialize(silent=True)
    store.buckets["bkt"] = {}
    store.tokens["token"] = "alice"
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "token")
    return store


def uploads(store) -> list[str]:
    return [path.removeprefix("/bkt/") for method, path, _ in store.requests if method == "PUT"]


def make_database(tmp_path, files):
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_files_land_under_destination(bucket, tmp_path):
    make_database(tmp_path, {"a/annotation.yaml": b"a: 1", "a/model.obj": b"obj-data"})
    author_database("bkt", str(tmp_path), "assets")
    assert bucket.buckets["bkt"] == {"assets/a/annotation.yaml": b"a: 1", "assets/a/model.obj": b"obj-data"}


def test_public_folder_source_lands_in_its_folder(bucket, tmp_path):
    make_database(tmp_path, {"w1/world.yaml": b"w"})
    author_database("public/worlds", str(tmp_path))
    assert bucket.buckets["public"] == {"worlds/w1/world.yaml": b"w"}


def test_no_destination_uploads_to_root(bucket, tmp_path):
    make_database(tmp_path, {"a/model.obj": b"1"})
    author_database("bkt", str(tmp_path))
    assert uploads(bucket) == ["a/model.obj"]


def test_skip_existing_with_same_content(bucket, tmp_path):
    make_database(tmp_path, {"a/model.obj": b"12345"})
    bucket.put("bkt", "a/model.obj", b"12345")
    author_database("bkt", str(tmp_path))
    assert uploads(bucket) == []


def test_reupload_on_same_size_other_content(bucket, tmp_path):
    make_database(tmp_path, {"a/model.obj": b"12345"})
    bucket.put("bkt", "a/model.obj", b"54321")
    author_database("bkt", str(tmp_path))
    assert bucket.buckets["bkt"]["a/model.obj"] == b"12345"


def test_reupload_on_size_mismatch(bucket, tmp_path):
    make_database(tmp_path, {"a/model.obj": b"123456"})
    bucket.put("bkt", "a/model.obj", b"12")
    author_database("bkt", str(tmp_path))
    assert bucket.buckets["bkt"]["a/model.obj"] == b"123456"


def test_no_overwrite_refuses_changed_objects(bucket, tmp_path):
    make_database(tmp_path, {"a/model.obj": b"mine"})
    bucket.put("bkt", "a/model.obj", b"theirs")
    with pytest.raises(StoreError, match="412"):
        author_database("bkt", str(tmp_path), overwrite=False)
    assert bucket.buckets["bkt"]["a/model.obj"] == b"theirs"


def test_empty_source_uploads_nothing(bucket, tmp_path):
    author_database("bkt", str(tmp_path))
    assert uploads(bucket) == []


def test_upload_error_propagates(bucket, tmp_path):
    make_database(tmp_path, {"a/x.obj": b"1", "a/y.obj": b"22"})
    bucket.failing.add("a/y.obj")
    with pytest.raises(StoreError, match="500"):
        author_database("bkt", str(tmp_path))


def test_explicit_token_skips_resolution(bucket, tmp_path, monkeypatch):
    monkeypatch.delenv("ARENA_ASSETS_TOKEN")
    bucket.tokens["explicit"] = "alice"
    make_database(tmp_path, {"a/model.obj": b"1"})
    author_database("bkt", str(tmp_path), token="explicit")
    assert bucket.seen("PUT", "/bkt/a/model.obj")[0]["Authorization"] == "Bearer explicit"


def test_missing_token_refuses_before_any_request(store, tmp_path):
    initialize(silent=True)
    make_database(tmp_path, {"a/model.obj": b"1"})
    with pytest.raises(RuntimeError, match="ARENA_ASSETS_TOKEN"):
        author_database("bkt", str(tmp_path))
    assert store.requests == []
