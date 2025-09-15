from __future__ import annotations

from typing import Any, Optional
import asyncio
import contextlib

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.reactive import var
from textual.widgets import Header, Footer, Input, RichLog, Static
from textual import events

from fast_agents.agent import Agent
from fast_agents.thread import Thread
from fast_agents.helpers.function_helper import string_to_user_message


class FastAgentsTUI(App):
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]
    CSS_PATH = None

    def __init__(
        self,
        agent: Agent,
    ) -> None:
        super().__init__()
        self.agent = agent

        self.input_history: list[dict[str, Any]] = []
        
        self.title = f"fast-agents"
        self.thinking = var(False)
        self.current_task: asyncio.Task | None = None
        self.available_commands: list[str] = ["/exit", "/reasoning", "/max-turns"]
        self.suggest_index: int = 0
        self.suggest_visible: bool = False
        # Keep a persistent thread so handoffs update agent live
        self.thread = Thread(agent=self.agent)
        # Spinner frames
        self.thinking_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index = 0
        self._spinner_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            self.meta = Static("", id="meta")
            yield self.meta
            self.chat = RichLog(markup=True, wrap=True, id="chat")
            yield self.chat
            # Thinking indicator lives just above the input box
            self.status = Static("", id="status")
            yield self.status
            # Command suggestions (shown when typing '/')
            self.suggestions = Static("", id="suggestions")
            yield self.suggestions
            self.user_input = Input(placeholder="> Send a message…", id="input")
            yield self.user_input
        yield Footer()

    async def on_mount(self) -> None:
        self.user_input.focus()
        self._update_meta()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return
        # slash commands
        if message.startswith("/"):
            cmd = message[1:].strip().lower()
            if cmd in {"exit", "quit"}:
                await self.action_quit()
                return
            if cmd.startswith("reasoning"):
                parts = cmd.split()
                level = parts[1] if len(parts) > 1 else ""
                valid, value = self._map_reasoning(level)
                if not valid:
                    self.status.update("[red]Usage: /reasoning off|minimal|low|medium|high[/red]")
                else:
                    self.thread.agent.reasoning_effort = value  # type: ignore[assignment]
                    shown = value if value is not None else "off"
                    self.status.update(f"[grey62]reasoning set to {shown}[/grey62]")
                    self._update_meta()
                # Clear command and suggestions
                event.input.value = ""
                self.suggest_visible = False
                self.suggestions.update("")
                return
            if cmd.startswith("max-turns"):
                parts = cmd.split()
                if len(parts) != 2:
                    self.status.update("[red]Usage: /max-turns <positive-int>[/red]")
                    event.input.value = ""
                    self.suggest_visible = False
                    self.suggestions.update("")
                    return
                try:
                    value = int(parts[1])
                    if value <= 0:
                        raise ValueError
                except ValueError:
                    self.status.update("[red]max-turns must be a positive integer[/red]")
                    event.input.value = ""
                    self.suggest_visible = False
                    self.suggestions.update("")
                    return
                self.thread.max_turns = value
                self._update_meta()
                self.status.update(f"[grey62]max_turns set to {value}[/grey62]")
                event.input.value = ""
                self.suggest_visible = False
                self.suggestions.update("")
                return
        event.input.value = ""
        await self._append_user(message)
        await self._run_thread_turn(message)

    def on_input_changed(self, event: Input.Changed) -> None:  # type: ignore[override]
        value = event.value
        if value.startswith("/"):
            self.suggest_visible = True
            # filter commands by prefix
            filtered = [c for c in self.available_commands if c.startswith(value)] or self.available_commands
            self.suggest_index = min(self.suggest_index, len(filtered) - 1)
            rendered = []
            for i, cmd in enumerate(filtered):
                if i == self.suggest_index:
                    rendered.append(f"[reverse]{cmd}[/reverse]")
                else:
                    rendered.append(cmd)
            self.suggestions.update(" ".join(rendered))
        else:
            self.suggest_visible = False
            self.suggestions.update("")

    async def _append_user(self, text: str) -> None:
        # light gray user message
        self.chat.write(f"[grey70]> {self._escape(text)}[/grey70]")

    async def _append_assistant(self, text: str) -> None:
        # assistant with bullet indicator
        self.chat.write(f"[b][white]●[/white][/b] {self._escape(text)}")

    def _message_to_history(self, text: str) -> None:
        self.input_history.append(string_to_user_message(text))

    def _escape(self, text: str) -> str:
        # basic escaping for Rich markup
        return text.replace("[", "[[").replace("]", "]] ")

    def _extract_text(self, item: Any) -> str:
        t = getattr(item, "type", None)
        if t == "message":
            parts = []
            for c in getattr(item, "content", []) or []:
                ct = getattr(c, "type", None)
                text = getattr(c, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts) or "[empty]"
        if t == "function_call":
            name = getattr(item, "name", "<fn>")
            args = getattr(item, "arguments", "{}")
            return f"[bold cyan][tool][/bold cyan] [cyan]{name}[/cyan]{args}"
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            return f"[green][tool output][/green] {item.get('output')}"

    async def _run_thread_turn(self, user_text: str) -> None:
        self._message_to_history(user_text)
        # Update thread input each turn
        self.thread.input = self.input_history.copy()
        self.thinking = True
        self._show_thinking()

        async def _runner() -> None:
            try:
                self._update_meta()
                async for output in self.thread.run():
                    extracted = self._extract_text(output)
                    if extracted:
                        self.chat.write(extracted)
            except asyncio.CancelledError:
                self.chat.write("[grey62][interrupted][/grey62]")
                raise
            finally:
                self.thinking = False
                self._hide_thinking()
                # Reset counter after each run
                self.thread.turn_count = 0
                self._update_meta()

        # Cancel previous task if somehow still running
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.current_task

        self.current_task = asyncio.create_task(_runner())
        # Don't await; let it stream while UI remains responsive

    def _show_thinking(self) -> None:
        self.set_focus(self.user_input)
        # Start spinner
        if self._spinner_task and not self._spinner_task.done():
            self._spinner_task.cancel()
        self._spinner_task = asyncio.create_task(self._spin())
        self.status.update("[grey62]thinking… (esc to interrupt)[/grey62]")

    def _hide_thinking(self) -> None:
        # Stop spinner
        if self._spinner_task and not self._spinner_task.done():
            self._spinner_task.cancel()
        self.status.update("")

    def action_interrupt(self) -> None:
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        else:
            # if no running task, just clear indicator if any
            self.thinking = False
            self._hide_thinking()

    def on_key(self, event: events.Key) -> None:  # type: ignore[override]
        # Ensure ESC works even when Input has focus
        if event.key == "escape":
            self.action_interrupt()
            return
        if self.suggest_visible:
            if event.key in {"down", "right"}:
                self.suggest_index += 1
                self.refresh_suggestions()
                return
            if event.key in {"up", "left"}:
                self.suggest_index = max(0, self.suggest_index - 1)
                self.refresh_suggestions()
                return
            if event.key == "tab":
                # apply current suggestion
                options = [c for c in self.available_commands if c.startswith(self.user_input.value)] or self.available_commands
                if options:
                    cmd = options[min(self.suggest_index, len(options)-1)]
                    self.user_input.value = cmd + " "
                self.suggest_visible = False
                self.suggestions.update("")
                return

    def refresh_suggestions(self) -> None:
        value = self.user_input.value
        options = [c for c in self.available_commands if c.startswith(value)] or self.available_commands
        self.suggest_index = max(0, min(self.suggest_index, len(options) - 1))
        rendered = []
        for i, cmd in enumerate(options):
            if i == self.suggest_index:
                rendered.append(f"[reverse]{cmd}[/reverse]")
            else:
                rendered.append(cmd)
        self.suggestions.update(" ".join(rendered))

    async def _spin(self) -> None:
        try:
            while True:
                frame = self.thinking_frames[self._spinner_index % len(self.thinking_frames)]
                self._spinner_index += 1
                self.status.update(f"[grey62]{frame} thinking… (esc to interrupt)[/grey62]")
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            pass

    def _update_meta(self) -> None:
        reasoning = getattr(self.thread.agent, "reasoning_effort", None) or "-"
        self.meta.update(
            f"Agent: {self.thread.agent.name} | Model: {self.thread.agent.model} | Reasoning: {reasoning} | Turn: {self.thread.turn_count}/{self.thread.max_turns}"
        )

    def _map_reasoning(self, level: str) -> tuple[bool, Optional[str]]:
        normalized = level.strip().lower()
        if normalized in {"off", "none", "null", "0", "no"}:
            return True, None
        if normalized in {"min", "minimal"}:
            return True, "minimal"
        if normalized in {"low", "l"}:
            return True, "low"
        if normalized in {"medium", "med", "m"}:
            return True, "medium"
        if normalized in {"high", "h"}:
            return True, "high"
        return False, None
