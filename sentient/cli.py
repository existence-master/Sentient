"""Developer commands for the backend the desktop app launches.

The product is the desktop application. This CLI exists to run the backend on
its own (``serve``), check an install (``doctor``) and inspect configuration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sentient import __version__, paths
from sentient.config import config_json_schema, load_config, save_config
from sentient.config.schema import SentientConfig

app = typer.Typer(help="Sentient backend (launched by the desktop app).", no_args_is_help=True)
config_app = typer.Typer(help="Inspect and edit configuration.")
app.add_typer(config_app, name="config")

console = Console()


def _setup_logging(verbose: bool) -> None:
    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("LiteLLM", "httpx", "httpcore", "aiosqlite", "multipart", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")):
    _setup_logging(verbose)


@app.command()
def version():
    """Print the version."""
    console.print(f"sentient {__version__}")


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Override gateway.host"),
    port: int | None = typer.Option(None, help="Override gateway.port"),
    reload: bool = typer.Option(False, help="Dev auto-reload"),
):
    """Run the backend (REST + WebSocket). The desktop app starts this for you."""
    import uvicorn

    from sentient.gateway.auth import ENV_TOKEN, load_or_create_token

    cfg = load_config()
    h = host or cfg.gateway.host
    p = port if port is not None else cfg.gateway.port
    if h not in {"127.0.0.1", "localhost", "::1"}:
        console.print("[yellow]WARNING: binding a non-loopback address exposes your assistant to the network.[/yellow]")
    token = load_or_create_token()
    if not os.environ.get(ENV_TOKEN):
        # started by hand: show how to open the UI in a browser
        console.print(f"Sentient backend on http://{h}:{p}/  (dev UI: http://{h}:{p}/?api=http://{h}:{p}&token={token})")
    uvicorn.run(
        "sentient.gateway.app:create_app", factory=True, host=h, port=p, reload=reload,
        log_level="warning", ws_ping_interval=20, ws_ping_timeout=60,
    )


@app.command()
def node(
    url: str = typer.Option(..., help="Engine address, e.g. wss://192.168.1.20:7778/ws/node"),
    code: str | None = typer.Option(None, help="Pairing code shown in Sentient (first connection only)"),
    name: str = typer.Option("Reference node", help="Name shown in Sentient"),
    kind: str = typer.Option("glasses", help="phone, glasses, watch or custom"),
    camera: int | None = typer.Option(None, help="Webcam index to offer as the device camera"),
    fingerprint: str | None = typer.Option(
        None, help="Certificate fingerprint shown in Sentient next to the pairing code (remembered after pairing)"
    ),
    insecure: bool = typer.Option(False, "--insecure", help="Accept any certificate (development only)"),
    speech: bool = typer.Option(True, "--speech/--no-speech", help="Speak 'speak' requests aloud with pyttsx3"),
):
    """Run a reference device node that simulates a phone or smart glasses (owner: nodes agent).

    --url also accepts the pairing link: "sentient://pair?url=...&code=...&fp=...".
    """
    from sentient.nodes.reference import main as node_main

    node_main(
        url=url, code=code, name=name, kind=kind, camera=camera, fingerprint=fingerprint, insecure=insecure,
        speech=speech,
    )


@app.command()
def doctor(
    models: bool = typer.Option(
        False, "--models", help="Also run the model check-up: a reply, a tool call, JSON, context and GPU per role."
    ),
):
    """Check that everything needed to run is in place."""
    asyncio.run(_doctor(models))


async def _doctor(models: bool = False) -> None:
    from sentient.app import SentientApp

    table = Table(title="sentient doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")

    def row(name: str, ok: bool | None, detail: str = ""):
        mark = "[green]ok[/green]" if ok else ("[yellow]warn[/yellow]" if ok is None else "[red]fail[/red]")
        table.add_row(name, mark, detail)

    row("home", paths.home().exists(), str(paths.home()))
    try:
        cfg = load_config()
        row("config", True, str(paths.config_file()) if paths.config_file().exists() else "defaults (finish onboarding in the app)")
    except Exception as exc:
        row("config", False, str(exc))
        console.print(table)
        return
    s = SentientApp(cfg, enable_background=False)
    checkup_table: Table | None = None
    try:
        await s.start()
        row("database", True, str(s.store.path))
        row("sqlite-vec", s.store.vec_available, "semantic memory " + ("enabled" if s.store.vec_available else "DISABLED"))
        row("tools", True, f"{len(s.registry.tools())} tools from {len(s.registry.plugins())} plugins")
        row("skills", True, f"{len(s.skills.list())} active, {len(s.skills.list_pending())} pending review")
        try:
            text = ""
            async for chunk in s.llm.stream("primary", [{"role": "user", "content": "Reply with the single word: ready"}]):
                text += chunk.text
            row("primary model", True, f"{s.llm.model_for('primary')} -> {text.strip()[:40]!r}")
        except Exception as exc:
            row("primary model", False, f"{s.llm.model_for('primary')}: {exc}")
        sizes: dict[int, list[str]] = {}
        for role in ("primary", "fast", "planner", "executor", "vision", "voice"):
            n = await s.llm.context_length(role)
            if n:
                sizes.setdefault(n, []).append(role)
        if sizes:
            row("context length", True, "; ".join(f"{n} tokens: {', '.join(roles)}" for n, roles in sizes.items()))
        try:
            [vec] = await s.llm.embed(["hello"])
            row("embedding model", True, f"{s.llm.model_for('embedding')} (dim {len(vec)})")
        except Exception as exc:
            row("embedding model", False, f"{s.llm.model_for('embedding')}: {exc}")
        if models:
            checkup_table = await _model_checkup(s)
    except Exception as exc:
        row("startup", False, str(exc))
    finally:
        await s.stop()
    console.print(table)
    if checkup_table is not None:
        console.print(checkup_table)


async def _model_checkup(s) -> Table:
    from sentient.llm.checkup import run_checkup

    marks = {"pass": "[green]ok[/green]", "warn": "[yellow]warn[/yellow]", "fail": "[red]fail[/red]", "skip": "skip"}
    table = Table(title="model check-up")
    table.add_column("role")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    with console.status("Checking models...") as status:
        async for event in run_checkup(s.config, s.llm):
            if event["type"] == "step":
                status.update(f"{event['role']}: {event['label']}...")
            elif event["type"] == "role":
                name = f"{event['role']}\n[dim]{event['model'] or 'uses primary'}[/dim]"
                if not event["checks"]:
                    table.add_row(name, "", marks["skip"], "Uses the primary model.")
                for i, c in enumerate(event["checks"]):
                    detail = c["detail"] + (f"\n[bold]Fix:[/bold] {c['fix']}" if c.get("fix") else "")
                    table.add_row(name if i == 0 else "", c["label"], marks[c["status"]], detail)
                table.add_section()
    return table


@config_app.command("path")
def config_path():
    console.print(str(paths.config_file()))


@config_app.command("show")
def config_show(defaults: bool = typer.Option(False, help="Show defaults instead of the saved file.")):
    cfg = SentientConfig() if defaults else load_config()
    console.print_json(cfg.model_dump_json(indent=2))


@config_app.command("schema")
def config_schema(out: Path | None = typer.Option(None, help="Write JSON schema to a file.")):
    schema = config_json_schema()
    if out:
        out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        console.print(f"wrote {out}")
    else:
        console.print_json(json.dumps(schema))


@config_app.command("set")
def config_set(key: str = typer.Argument(..., help="Dotted key, e.g. models.roles.primary"), value: str = typer.Argument(...)):
    cfg = load_config()
    data = cfg.model_dump(mode="json")
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    node[parts[-1]] = parsed
    save_config(SentientConfig.model_validate(data))
    console.print(f"[green]set[/green] {key} = {parsed!r}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
