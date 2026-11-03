import json
from pathlib import Path

import typer

from arena_assets.cli.utils import safe_echo, store_errors


def author_command(
    ctx: typer.Context,
    database_dir: Path = typer.Argument(
        ...,
        help="Built database directory to upload",
        file_okay=False,
        dir_okay=True,
        readable=True,
        exists=True,
    ),
    destination: str = typer.Option("", "--destination", "-d", help="Path inside the bucket to upload under"),
    token: str | None = typer.Option(None, "--token", help="Store bearer token (defaults to ARENA_ASSETS_TOKEN or gcloud)"),
    no_overwrite: bool = typer.Option(False, "--no-overwrite", help="Refuse to replace objects that exist with other content", is_flag=True),
):
    """Upload a built database to the source, skipping files the store already holds. Prints a JSON summary of files, sent, skipped and bytes."""
    source = ctx.obj["source"]

    safe_echo(f"Uploading database from {database_dir} to bucket: {source}", ctx)

    from arena_assets.impl.author import author_database

    summary = author_database(
        bucket=source,
        source=str(database_dir),
        destination=destination,
        token=token,
        overwrite=not no_overwrite,
    )

    print(json.dumps(summary))
    safe_echo(f"Sent {summary['sent']} of {summary['files']} file(s), {summary['skipped']} already on the store", ctx)


def add_to_cmd(cmd: typer.Typer):
    """Add author command to the network command group."""
    cmd.command("author")(store_errors(author_command))


if __name__ == "__main__":
    typer.run(author_command)
