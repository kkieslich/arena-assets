import typer

from arena_assets.cli.utils import safe_echo, store_errors


def delete_command(
    ctx: typer.Context,
    names: list[str] = typer.Argument(..., help="Object names inside the source to delete"),
    recursive: bool = typer.Option(False, "-r", "--recursive", help="Treat each name as a folder and delete every object below it"),
    token: str | None = typer.Option(None, "--token", help="Store bearer token (defaults to ARENA_ASSETS_TOKEN or gcloud)"),
):
    """Delete objects from the source, printing each deleted name. The store allows it to their uploader or an admin."""
    source = ctx.obj["source"]

    from arena_assets.impl.fetch import Bucket, StoreError, resolve_token

    b = Bucket(source, token=token if token is not None else resolve_token())
    targets = [found for name in names for found in b.under(name)] if recursive else names
    if not targets:
        raise StoreError(f"nothing under {', '.join(names)} in {source}")
    for name in dict.fromkeys(targets):
        b.delete(name)
        print(name)
    safe_echo(f"Deleted {len(set(targets))} object(s) from {source}", ctx)


def add_to_cmd(cmd: typer.Typer):
    """Add delete command to the network command group."""
    cmd.command("delete")(store_errors(delete_command))


if __name__ == "__main__":
    typer.run(delete_command)
