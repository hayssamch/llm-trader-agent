"""
live/notifier.py — Telegram delivery for the daily signal digest.

Credentials come from the environment only (TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID) and are never logged: every exception text is scrubbed of the
token before it can reach a log line, because `requests` embeds the full request
URL (which contains the token) in its error messages.
"""
import logging
import os
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

API = "https://api.telegram.org"
MAX_LEN = 4000          # Telegram hard limit is 4096 characters per message


class NotifyError(RuntimeError):
    pass


class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None,
                 dry_run: bool = False):
        self.token = (token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
        self.dry_run = dry_run
        if not dry_run and not (self.token and self.chat_id):
            raise NotifyError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set "
                "(or run with --dry-run to print instead of sending).")

    # ── internals ────────────────────────────────────────────────────────────
    def _scrub(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text

    def _post(self, method: str, **kwargs) -> None:
        url = f"{API}/bot{self.token}/{method}"
        last = ""
        for attempt in range(3):
            try:
                r = requests.post(url, timeout=30, **kwargs)
                if r.status_code == 429:                      # rate limited
                    time.sleep(2 * (attempt + 1))
                    last = "HTTP 429 (rate limited)"
                    continue
                if r.status_code != 200:
                    desc = ""
                    try:
                        desc = r.json().get("description", "")
                    except Exception:
                        pass
                    raise NotifyError(f"Telegram {method} failed: HTTP "
                                      f"{r.status_code} {desc}".strip())
                return
            except NotifyError:
                raise
            except Exception as exc:                          # network errors
                last = self._scrub(f"{type(exc).__name__}: {exc}")
                time.sleep(2)
        raise NotifyError(f"Telegram {method} failed after retries: {last}")

    @staticmethod
    def split(text: str, limit: int = MAX_LEN) -> List[str]:
        """Split on line boundaries so no message exceeds Telegram's limit."""
        if len(text) <= limit:
            return [text]
        parts, cur = [], ""
        for line in text.split("\n"):
            if len(cur) + len(line) + 1 > limit and cur:
                parts.append(cur)
                cur = ""
            cur += line + "\n"
        if cur.strip():
            parts.append(cur)
        return parts

    # ── public API ───────────────────────────────────────────────────────────
    def send(self, html_text: str, silent: bool = False) -> None:
        for part in self.split(html_text):
            if self.dry_run:
                print("\n----- TELEGRAM (dry run) -----\n" + part + "\n------------------------------")
                continue
            self._post("sendMessage", json={
                "chat_id": self.chat_id, "text": part, "parse_mode": "HTML",
                "disable_web_page_preview": True, "disable_notification": silent,
            })

    def send_photo(self, path: str, caption: str = "", silent: bool = True) -> None:
        if self.dry_run:
            print(f"[dry run] would send chart image {path}")
            return
        with open(path, "rb") as fh:
            self._post("sendPhoto",
                       data={"chat_id": self.chat_id, "caption": caption[:1000],
                             "parse_mode": "HTML", "disable_notification": silent},
                       files={"photo": fh})
