from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from fast_agents.agent import Agent
from fast_agents.tui import FastAgentsTUI


def _ensure_cwd_on_sys_path() -> None:
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _locate_symbol(path: str) -> Any:
    if ":" in path:
        module_path, symbol_name = path.split(":", 1)
    else:
        parts = path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError("Import path must include module and symbol, e.g. pkg.mod:agent")
        module_path, symbol_name = parts

    _ensure_cwd_on_sys_path()
    module: ModuleType = importlib.import_module(module_path)
    if not hasattr(module, symbol_name):
        raise AttributeError(f"Symbol '{symbol_name}' not found in module '{module_path}'")
    return getattr(module, symbol_name)


def _resolve_agent(obj: Any) -> Agent:
    if isinstance(obj, Agent):
        return obj
    if isinstance(obj, type) and issubclass(obj, Agent):
        return obj()  # type: ignore[call-arg]
    if callable(obj):
        maybe_agent = obj()  # type: ignore[misc]
        if isinstance(maybe_agent, Agent):
            return maybe_agent
    raise TypeError("Provided symbol must be an Agent instance, Agent subclass, or a factory returning Agent")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fast-agents")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a TUI chat with an Agent")
    run_p.add_argument("import_path", help="Import path to Agent (e.g. pkg.module:agent)")

    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            symbol = _locate_symbol(args.import_path)
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None)
            print(
                "Could not import module. Ensure you are running from your project root "
                "or add it to PYTHONPATH. Error:",
                e,
            )
            raise

        agent = _resolve_agent(symbol)

        app = FastAgentsTUI(
            agent=agent
        )
        app.run()


if __name__ == "__main__":
    main()


