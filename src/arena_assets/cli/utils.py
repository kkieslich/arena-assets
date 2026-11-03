"""Utility functions for CLI operations."""

import functools
import typing

import typer


def safe_echo(message: str, ctx: typer.Context | None = None):
    """Echo a message only if not in silent mode."""
    if ctx and ctx.obj and ctx.obj.get("silent", False):
        return
    typer.echo(message, err=True)


def get_silent_flag(ctx: typer.Context) -> bool:
    """Get the silent flag from context."""
    return ctx.obj.get("silent", False) if ctx and ctx.obj else False


def store_errors(fn: typing.Callable[..., object]) -> typing.Callable[..., object]:
    """Report asset store and network failures as one line and exit 1."""

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        from arena_assets.impl.fetch import StoreError

        try:
            return fn(*args, **kwargs)
        except (StoreError, OSError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from None

    return wrapper
