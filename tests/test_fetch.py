import pytest

from arena_assets.impl.fetch import Bucket, StoreError, fetch_database
from arena_assets.utils.logging import initialize


@pytest.fixture
def bucket(store):
    initialize(silent=True)
    store.buckets["bkt"] = {}
    return store


def downloads(store) -> list[str]:
    return [path.removeprefix("/bkt/") for method, path, _ in store.requests if method == "GET" and path != "/bkt/index.json"]


def test_files_land_under_destination(bucket, tmp_path):
    bucket.put("bkt", "assets/a/annotation.yaml", b"a: 1")
    bucket.put("bkt", "assets/a/model.obj", b"obj-data")
    fetch_database("bkt", ["assets/a"], str(tmp_path))
    assert (tmp_path / "assets/a/annotation.yaml").read_bytes() == b"a: 1"
    assert (tmp_path / "assets/a/model.obj").read_bytes() == b"obj-data"


def test_public_folder_source_lands_without_the_folder(store, tmp_path):
    initialize(silent=True)
    store.put("public", "worlds/w1/world.yaml", b"w")
    fetch_database("public/worlds", ["w1"], str(tmp_path))
    assert (tmp_path / "w1/world.yaml").read_bytes() == b"w"


def test_absolute_blob_name_stays_contained(bucket, tmp_path):
    bucket.put("bkt", "/etc/passwd", b"pwned")
    fetch_database("bkt", [""], str(tmp_path))
    assert (tmp_path / "etc/passwd").read_bytes() == b"pwned"


def test_format_filter(bucket, tmp_path):
    bucket.put("bkt", "a/model.obj", b"1")
    bucket.put("bkt", "a/model.fbx", b"2")
    fetch_database("bkt", ["a"], str(tmp_path), model_formats=["obj"])
    assert downloads(bucket) == ["a/model.obj"]


def test_no_annotations(bucket, tmp_path):
    bucket.put("bkt", "a/annotation.yaml", b"a: 1")
    bucket.put("bkt", "a/model.obj", b"1")
    fetch_database("bkt", ["a"], str(tmp_path), annotations=False)
    assert downloads(bucket) == ["a/model.obj"]


def test_skip_existing_with_same_content(bucket, tmp_path):
    bucket.put("bkt", "a/model.obj", b"12345")
    target = tmp_path / "a/model.obj"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"12345")
    fetch_database("bkt", ["a"], str(tmp_path))
    assert downloads(bucket) == []


def test_redownload_on_same_size_other_content(bucket, tmp_path):
    bucket.put("bkt", "a/model.obj", b"12345")
    target = tmp_path / "a/model.obj"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"54321")
    fetch_database("bkt", ["a"], str(tmp_path))
    assert target.read_bytes() == b"12345"


def test_redownload_on_size_mismatch(bucket, tmp_path):
    bucket.put("bkt", "a/model.obj", b"123456")
    target = tmp_path / "a/model.obj"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"12")
    fetch_database("bkt", ["a"], str(tmp_path))
    assert target.read_bytes() == b"123456"


def test_dedup_overlapping_paths(bucket, tmp_path):
    bucket.put("bkt", "a/b/model.obj", b"1")
    fetch_database("bkt", ["a", "a/b"], str(tmp_path))
    assert downloads(bucket) == ["a/b/model.obj"]


def test_missing_path_downloads_nothing(bucket, tmp_path):
    bucket.put("bkt", "a/model.obj", b"1")
    fetch_database("bkt", ["b"], str(tmp_path))
    assert downloads(bucket) == []


def test_download_error_propagates(bucket, tmp_path):
    bucket.put("bkt", "a/x.obj", b"1")
    bucket.put("bkt", "a/y.obj", b"22")
    bucket.failing.add("a/y.obj")
    with pytest.raises(StoreError, match="GET bkt/a/y.obj: 500 broken"):
        fetch_database("bkt", ["a"], str(tmp_path))


def test_unknown_source_raises(store, tmp_path):
    initialize(silent=True)
    with pytest.raises(StoreError):
        fetch_database("arena-models-prod-public", ["a"], str(tmp_path))


def test_object_listed_but_gone_from_the_store_is_skipped(bucket, tmp_path):
    bucket.put("bkt", "a/x.obj", b"1")
    bucket.put("bkt", "a/y.obj", b"22")
    Bucket("bkt").objects()
    del bucket.buckets["bkt"]["a/y.obj"]
    fetch_database("bkt", ["a"], str(tmp_path))
    assert (tmp_path / "a/x.obj").read_bytes() == b"1"
    assert not list((tmp_path / "a").glob("y.obj*"))
