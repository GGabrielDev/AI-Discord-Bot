import unittest

from progress_logger import SharedProgressLogger, StatusMessagePlanner


class StatusMessagePlannerTests(unittest.TestCase):
    def test_rolls_over_when_message_length_would_overflow(self):
        planner = StatusMessagePlanner(max_length=80, max_substeps_per_message=10, max_edit_window_seconds=999)

        root = planner.plan("Main task")
        first_substep = planner.plan("short detail", is_sub_step=True)
        rollover = planner.plan("x" * 80, is_sub_step=True)

        self.assertTrue(root.requires_new_message)
        self.assertFalse(first_substep.requires_new_message)
        self.assertTrue(rollover.requires_new_message)
        self.assertTrue(rollover.rolled_over)
        self.assertIn("continued", rollover.content)

    def test_rolls_over_when_edit_window_expires(self):
        now = [10.0]
        planner = StatusMessagePlanner(
            max_length=200,
            max_substeps_per_message=10,
            max_edit_window_seconds=5,
            time_func=lambda: now[0],
        )

        planner.plan("Main task")
        now[0] = 12.0
        planner.plan("first detail", is_sub_step=True)
        now[0] = 18.0
        rollover = planner.plan("second detail", is_sub_step=True)

        self.assertTrue(rollover.requires_new_message)
        self.assertTrue(rollover.rolled_over)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.edits: list[str] = []

    async def edit(self, *, content: str):
        self.content = content
        self.edits.append(content)
        return self


class _FakeChannel:
    def __init__(self):
        self.sent_payloads: list[dict] = []
        self.messages: list[_FakeMessage] = []

    async def send(self, **kwargs):
        self.sent_payloads.append(kwargs)
        message = _FakeMessage(kwargs.get("content", ""))
        self.messages.append(message)
        return message


class _FakeFollowup:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.sent_payloads: list[dict] = []

    async def send(self, **kwargs):
        if self.should_fail:
            raise RuntimeError("token expired")
        self.sent_payloads.append(kwargs)
        return kwargs


class _FakeResponse:
    def __init__(self):
        self._done = False
        self.sent_payloads: list[dict] = []

    def is_done(self):
        return self._done

    async def send_message(self, **kwargs):
        self._done = True
        self.sent_payloads.append(kwargs)
        return kwargs


class _FakeInteraction:
    def __init__(self, *, followup_should_fail: bool = False):
        self.channel = _FakeChannel()
        self.followup = _FakeFollowup(should_fail=followup_should_fail)
        self.response = _FakeResponse()
        self.user = type("User", (), {"id": 42})()
        self.original_edits: list[dict] = []

    async def edit_original_response(self, **kwargs):
        self.original_edits.append(kwargs)
        return kwargs


class SharedProgressLoggerTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_updates_edit_then_roll_over(self):
        interaction = _FakeInteraction()
        console_lines: list[str] = []
        logger = SharedProgressLogger(
            interaction,
            console_writer=console_lines.append,
            planner=StatusMessagePlanner(max_length=200, max_substeps_per_message=1, max_edit_window_seconds=999),
        )

        await logger.log("Main task")
        await logger.log("first detail", is_sub_step=True)
        await logger.log("second detail", is_sub_step=True)

        self.assertEqual(len(interaction.channel.messages), 2)
        self.assertEqual(len(interaction.channel.messages[0].edits), 1)
        self.assertIn("continued", interaction.channel.messages[1].content)
        self.assertIn("Main task", console_lines[0])
        self.assertTrue(any("↳ first detail" in line for line in console_lines))

    async def test_send_message_falls_back_to_channel_and_mentions_user(self):
        interaction = _FakeInteraction(followup_should_fail=True)
        logger = SharedProgressLogger(interaction, console_writer=lambda *_: None)

        await logger.send_message("Finished!", mention_on_fallback=True)

        self.assertEqual(interaction.followup.sent_payloads, [])
        self.assertEqual(interaction.channel.sent_payloads[-1]["content"], "<@42> Finished!")


if __name__ == "__main__":
    unittest.main()
