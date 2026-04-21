from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable


def _truncate_text(text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3] + "..."


@dataclass(slots=True)
class MessageUpdatePlan:
    content: str
    requires_new_message: bool
    rolled_over: bool = False


class StatusMessagePlanner:
    ROOT_PREFIX = "> "
    SUBSTEP_PREFIX = "> ↳ "
    FALLBACK_ROOT = "Progress update"

    def __init__(
        self,
        *,
        max_length: int = 1900,
        max_substeps_per_message: int = 8,
        max_edit_window_seconds: float = 90.0,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.max_length = max_length
        self.max_substeps_per_message = max_substeps_per_message
        self.max_edit_window_seconds = max_edit_window_seconds
        self._time_func = time_func or time.monotonic
        self._root_message = ""
        self._lines: list[str] = []
        self._started_at = self._time_func()
        self._substep_count = 0
        self._continuation_count = 0

    def plan(self, message: str, is_sub_step: bool = False) -> MessageUpdatePlan:
        text = str(message).strip()
        now = self._time_func()

        if not is_sub_step:
            self._root_message = text or self.FALLBACK_ROOT
            self._continuation_count = 0
            self._lines = [self._format_root(self._root_message)]
            self._started_at = now
            self._substep_count = 0
            return MessageUpdatePlan(
                content=self._render(self._lines),
                requires_new_message=True,
            )

        if not self._lines:
            self._root_message = self.FALLBACK_ROOT
            self._continuation_count = 0
            self._lines = [self._format_root(self._root_message)]
            self._started_at = now
            self._substep_count = 0

        candidate_lines = [*self._lines, self._format_substep(text)]
        if self._should_roll_over(candidate_lines, now):
            self._continuation_count += 1
            self._started_at = now
            self._substep_count = 1
            self._lines = self._build_rollover_lines(text)
            return MessageUpdatePlan(
                content=self._render(self._lines),
                requires_new_message=True,
                rolled_over=True,
            )

        self._lines = candidate_lines
        self._substep_count += 1
        return MessageUpdatePlan(
            content=self._render(self._lines),
            requires_new_message=False,
        )

    def _should_roll_over(self, candidate_lines: list[str], now: float) -> bool:
        if len(self._render(candidate_lines)) > self.max_length:
            return True
        if self._substep_count >= self.max_substeps_per_message:
            return True
        if self._substep_count > 0 and (now - self._started_at) >= self.max_edit_window_seconds:
            return True
        return False

    def _build_rollover_lines(self, substep_message: str) -> list[str]:
        header = self._format_root(self._continued_root())
        remaining = self.max_length - len(header) - 1 - len(self.SUBSTEP_PREFIX)
        substep = self.SUBSTEP_PREFIX + _truncate_text(substep_message, remaining)
        return [header, substep]

    def _continued_root(self) -> str:
        if self._continuation_count <= 1:
            return f"{self._root_message} (continued)"
        return f"{self._root_message} (continued {self._continuation_count})"

    def _format_root(self, message: str) -> str:
        return self.ROOT_PREFIX + _truncate_text(message, self.max_length - len(self.ROOT_PREFIX))

    def _format_substep(self, message: str) -> str:
        return self.SUBSTEP_PREFIX + _truncate_text(message, self.max_length - len(self.SUBSTEP_PREFIX))

    @staticmethod
    def _render(lines: list[str]) -> str:
        return "\n".join(lines)


class SharedProgressLogger:
    def __init__(
        self,
        interaction: Any,
        *,
        console_writer: Callable[[str], None] | None = None,
        planner: StatusMessagePlanner | None = None,
    ) -> None:
        self.interaction = interaction
        self.console_writer = console_writer or print
        self.planner = planner or StatusMessagePlanner()
        self._lock = asyncio.Lock()
        self._status_message: Any = None

    async def __call__(self, message: str, is_sub_step: bool = False) -> Any:
        return await self.log(message, is_sub_step=is_sub_step)

    async def acknowledge(self, content: str) -> Any:
        self._mirror_console(content, is_sub_step=False)
        async with self._lock:
            response = getattr(self.interaction, "response", None)
            if response is not None and not self._response_is_done():
                try:
                    return await response.send_message(content=content)
                except Exception as exc:
                    self._write_console(f"[ProgressLogger] Initial response failed: {exc}")
            return await self._send_out_of_band(content=content)

    async def log(self, message: str, is_sub_step: bool = False) -> Any:
        self._mirror_console(message, is_sub_step=is_sub_step)
        channel = getattr(self.interaction, "channel", None)
        if channel is None:
            return None

        async with self._lock:
            plan = self.planner.plan(message, is_sub_step=is_sub_step)
            try:
                if self._status_message is None or plan.requires_new_message:
                    self._status_message = await channel.send(content=plan.content)
                    return self._status_message

                await self._status_message.edit(content=plan.content)
                return self._status_message
            except Exception as exc:
                self._write_console(f"[ProgressLogger] Status update failed: {exc}")
                try:
                    self._status_message = await channel.send(content=plan.content)
                    return self._status_message
                except Exception as send_exc:
                    self._write_console(f"[ProgressLogger] Status send failed: {send_exc}")
                    return None

    async def send_message(
        self,
        content: str | None = None,
        *,
        prefer_channel: bool = False,
        mention_on_fallback: bool = False,
        **kwargs: Any,
    ) -> Any:
        if content:
            self._mirror_console(content, is_sub_step=False)
        async with self._lock:
            return await self._send_out_of_band(
                content=content,
                prefer_channel=prefer_channel,
                mention_on_fallback=mention_on_fallback,
                **kwargs,
            )

    async def try_edit_acknowledgement(self, content: str) -> bool:
        self._mirror_console(content, is_sub_step=False)
        async with self._lock:
            try:
                await self.interaction.edit_original_response(content=content)
                return True
            except Exception as exc:
                self._write_console(f"[ProgressLogger] Original response edit skipped: {exc}")
                return False

    async def _send_out_of_band(
        self,
        *,
        content: str | None = None,
        prefer_channel: bool = False,
        mention_on_fallback: bool = False,
        **kwargs: Any,
    ) -> Any:
        ordered_targets = ["channel", "followup"] if prefer_channel else ["followup", "channel"]
        last_error: Exception | None = None

        for index, target_name in enumerate(ordered_targets):
            target = getattr(self.interaction, target_name, None)
            if target is None:
                continue
            payload = dict(kwargs)
            if content is not None:
                payload["content"] = content
            if target_name == "channel" and index > 0 and mention_on_fallback:
                payload["content"] = self._with_user_mention(payload.get("content"))
            try:
                return await target.send(**payload)
            except Exception as exc:
                last_error = exc
                self._write_console(f"[ProgressLogger] {target_name} send failed: {exc}")

        if last_error:
            raise last_error
        return None

    def _response_is_done(self) -> bool:
        response = getattr(self.interaction, "response", None)
        if response is None:
            return True
        is_done = getattr(response, "is_done", None)
        if callable(is_done):
            return bool(is_done())
        return bool(is_done)

    def _with_user_mention(self, content: str | None) -> str | None:
        user_id = getattr(getattr(self.interaction, "user", None), "id", None)
        if user_id is None or content is None:
            return content
        return f"<@{user_id}> {content}"

    def _mirror_console(self, message: str, *, is_sub_step: bool) -> None:
        prefix = "  ↳ " if is_sub_step else ""
        self._write_console(f"{prefix}{message}")

    def _write_console(self, line: str) -> None:
        try:
            self.console_writer(line)
        except Exception:
            pass
