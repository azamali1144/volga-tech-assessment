import io
import json
import logging
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from app.config import Settings
from app.logging_config import configure_logging


class LoggingConfigTest(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        self._saved = (root.handlers[:], root.level)

    def tearDown(self):
        root = logging.getLogger()
        root.handlers[:], level = self._saved
        root.setLevel(level)

    def capture(self, fmt, emit, level="INFO"):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            configure_logging(level, fmt)  # handler binds to the redirected stdout
            emit(logging.getLogger("app.test"))
        return buffer.getvalue().splitlines()

    def test_json_lines_carry_extra_fields(self):
        (line,) = self.capture(
            "json", lambda log: log.info("job completed", extra={"job_id": "abc", "segments": 12})
        )
        entry = json.loads(line)
        self.assertEqual(entry["message"], "job completed")
        self.assertEqual(entry["level"], "INFO")
        self.assertEqual(entry["logger"], "app.test")
        self.assertEqual((entry["job_id"], entry["segments"]), ("abc", 12))
        self.assertIn("ts", entry)

    def test_exceptions_are_included(self):
        def emit(log):
            try:
                raise ValueError("boom")
            except ValueError:
                log.exception("failed")

        (line,) = self.capture("json", emit)
        self.assertIn("ValueError: boom", json.loads(line)["exc_info"])

    def test_non_serializable_extras_do_not_break_logging(self):
        (line,) = self.capture("json", lambda log: log.info("x", extra={"path": Path("a/b")}))
        self.assertIn("path", json.loads(line))

    def test_level_filters_output(self):
        lines = self.capture("json", lambda log: (log.debug("hidden"), log.info("shown")), "INFO")
        self.assertEqual([json.loads(line)["message"] for line in lines], ["shown"])

    def test_text_format_is_human_readable_with_extras(self):
        (line,) = self.capture("text", lambda log: log.info("job done", extra={"job_id": "abc"}))
        self.assertIn("INFO app.test: job done job_id=abc", line)

    def test_configure_is_idempotent(self):
        configure_logging("INFO", "json")
        configure_logging("INFO", "json")
        self.assertEqual(len(logging.getLogger().handlers), 1)

    def test_uvicorn_loggers_route_through_root(self):
        configure_logging("INFO", "json")
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logger = logging.getLogger(name)
            self.assertEqual(logger.handlers, [])
            self.assertTrue(logger.propagate)

    def test_settings_validate_log_options(self):
        with self.assertRaises(ValueError):
            Settings(log_format="xml")
        with self.assertRaises(ValueError):
            Settings(log_level="LOUD")


if __name__ == "__main__":
    unittest.main()
