"""Network command group for arena_assets CLI."""

import typer

from .author import add_to_cmd as add_author_cmd
from .browse import add_to_cmd as add_browse_cmd
from .delete import add_to_cmd as add_delete_cmd
from .exists import add_to_cmd as add_exists_cmd
from .fetch import add_to_cmd as add_fetch_cmd
from .list import add_to_cmd as add_list_cmd


def add_to_cmd(cmd: typer.Typer):
    """Add network command group to the main CLI."""

    net_app = typer.Typer(
        name="net",
        help="Network operations for arena assets",
        rich_markup_mode="rich",
        no_args_is_help=True,
    )

    @net_app.callback()
    def net_callback(
        ctx: typer.Context,
        source: str = typer.Argument(help="Asset store source, <bucket>[/<folder>] such as public/models or arena-internal, `default` for public/models"),
    ):
        """Network operations with specified source."""
        if ctx:
            ctx.ensure_object(dict)
            if source == "default":
                from arena_assets.impl.fetch import DEFAULT_SOURCE

                source = DEFAULT_SOURCE
            ctx.obj["source"] = source
            if ctx.parent and ctx.parent.obj:
                ctx.obj["silent"] = ctx.parent.obj.get("silent", False)

    add_fetch_cmd(net_app)
    add_exists_cmd(net_app)
    add_list_cmd(net_app)
    add_browse_cmd(net_app)
    add_author_cmd(net_app)
    add_delete_cmd(net_app)

    cmd.add_typer(net_app, name="net")
