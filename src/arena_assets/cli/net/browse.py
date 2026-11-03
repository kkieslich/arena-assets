import json

import typer

from arena_assets.cli.utils import store_errors


def browse_command(
    ctx: typer.Context,
    path: str = typer.Argument("", help="Folder inside the source, empty for its top level"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable folders and files"),
):
    """Show the folders and files directly inside a folder of the source, with the object count and size below each folder."""
    source = ctx.obj["source"]

    from arena_assets.impl.fetch import Bucket, StoreError
    from arena_assets.utils.logging import format_file_size

    folders, files = Bucket(source).browse(path)
    if not folders and not files:
        raise StoreError(f"{source}/{path.strip('/')} holds nothing")
    if as_json:
        rows = [{"name": name, "files": count, "bytes": size} for name, (count, size) in folders.items()]
        print(json.dumps({"source": source, "path": path.strip("/"), "folders": rows, "files": files}))
        return
    width = max(len(name) + 1 for name in [*folders, *(entry["name"] for entry in files)])
    for name, (count, size) in folders.items():
        print(f"{name + '/':<{width}}  {format_file_size(size):>10}  {count} file(s)")
    for entry in files:
        print(f"{entry['name']:<{width}}  {format_file_size(int(entry.get('size', 0))):>10}  {entry.get('updated', '')}")


def add_to_cmd(cmd: typer.Typer):
    """Add browse command to the network command group."""
    cmd.command("browse")(store_errors(browse_command))


if __name__ == "__main__":
    typer.run(browse_command)
