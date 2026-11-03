import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from arena_assets.impl.fetch import MAX_UPLOAD, Bucket, StoreError, resolve_token, sha256_of, store_url
from arena_assets.utils.logging import format_file_size, get_logger, get_manager


def author_database(
    bucket: str,
    source: str,
    destination: str = "",
    token: str | None = None,
    overwrite: bool = True,
) -> dict[str, int]:
    """Upload a built database directory to an asset store source. Returns the counts of files, sent, skipped and bytes sent.

    Args:
        bucket: The source to upload to, `<bucket>[/<folder>]`
        source: The local built database directory to upload
        destination: Path inside the source to upload under (default: its root)
        token: Bearer token. If None, resolved from ARENA_ASSETS_TOKEN or gcloud
        overwrite: Replace objects that exist with other content, else the store refuses them
    """
    logger = get_logger("author")
    try:
        b = Bucket(bucket, token=token if token is not None else resolve_token())

        prefix = destination.strip("/")
        if prefix:
            prefix += "/"

        local_files = []
        for root, dirs, files in os.walk(source):
            dirs.sort()
            for file in sorted(files):
                local_path = os.path.join(root, file)
                blob_name = prefix + os.path.relpath(local_path, source).replace(os.sep, "/")
                local_files.append((local_path, blob_name, os.path.getsize(local_path), sha256_of(local_path)))

        if not local_files:
            logger.warning("No files found in '%s'", source)
            return {"files": 0, "sent": 0, "skipped": 0, "bytes": 0}

        too_large = [f"{path} ({size} bytes)" for path, _, size, _ in local_files if size > MAX_UPLOAD]
        if too_large:
            raise StoreError(f"the store takes files up to {MAX_UPLOAD} bytes: {', '.join(too_large)}")

        total_files = len(local_files)
        total_size = sum(size for _, _, size, _ in local_files)
        logger.info("Found %d files (%s) to upload", total_files, format_file_size(total_size))

        remote_sha256 = {blob["name"]: blob.get("sha256") for blob in b.listdir(destination)}

        uploaded_count = 0
        uploaded_size = 0
        skipped_size = 0

        with get_manager() as manager:
            status_bar = manager.status_bar(
                status_format="Uploading: {current_file}{fill}",
                current_file="Initializing...",
                color="cyan",
            )

            data_progress = manager.counter(
                total=total_size,
                desc=f"[{1:>{len(str(total_files))}d}/{total_files}]",
                color="blue",
                counter_format="{desc}{desc_pad}{count_formatted}/{total_formatted} [{elapsed}, {rate_formatted}/s]{fill}",
                bar_format="{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count_formatted}/{total_formatted} [{elapsed}<{eta}, {rate_formatted}/s]",
                fields={
                    "count_formatted": format_file_size(0),
                    "total_formatted": format_file_size(total_size),
                    "rate_formatted": format_file_size(0),
                },
            )

            pending = []

            for local_path, blob_name, size, sha256 in local_files:
                if remote_sha256.get(blob_name) == sha256:
                    logger.info(
                        "Skipping '%s' - already exists with matching sha256",
                        blob_name,
                    )
                    uploaded_count += 1
                    skipped_size += size

                    new_total = total_size - skipped_size
                    data_progress.total = new_total

                    elapsed_time = data_progress.elapsed
                    current_rate = uploaded_size / elapsed_time if elapsed_time > 0 else 0

                    data_progress.desc = f"[{uploaded_count:>{len(str(total_files))}d}/{total_files}]"
                    data_progress.update(
                        0,
                        count_formatted=format_file_size(uploaded_size),
                        total_formatted=format_file_size(new_total),
                        rate_formatted=format_file_size(current_rate),
                    )
                    continue

                pending.append((local_path, blob_name, size, sha256))

            with ThreadPoolExecutor(max_workers=20) as pool:
                futures = {pool.submit(b.upload, local_path, blob_name, sha256, overwrite): (blob_name, size) for local_path, blob_name, size, sha256 in pending}
                try:
                    for future in as_completed(futures):
                        blob_name, size = futures[future]
                        future.result()

                        uploaded_count += 1
                        uploaded_size += size

                        status_bar.update(current_file=f"{blob_name} ({format_file_size(size)})")

                        elapsed_time = data_progress.elapsed
                        current_rate = uploaded_size / elapsed_time if elapsed_time > 0 else 0

                        data_progress.desc = f"[{uploaded_count:>{len(str(total_files))}d}/{total_files}]"
                        data_progress.update(
                            size,
                            count_formatted=format_file_size(uploaded_size),
                            total_formatted=format_file_size(total_size - skipped_size),
                            rate_formatted=format_file_size(current_rate),
                        )
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise

            status_bar.update(current_file="Complete!")

        logger.info(
            "Uploaded %d files (%s) successfully to: %s/%s/%s",
            uploaded_count,
            format_file_size(uploaded_size),
            store_url(),
            bucket,
            prefix,
        )
        return {"files": total_files, "sent": len(pending), "skipped": total_files - len(pending), "bytes": uploaded_size}

    except Exception as e:
        logger.error("Author error: %s", e)
        raise
