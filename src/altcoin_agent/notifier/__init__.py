"""Notifier package — out-of-band channels for human-readable alerts.

Currently includes:

  * Telegram (telegram.TelegramNotifier)

All notifiers share the same ``Notifier`` Protocol so main.py can fan-out
events to multiple channels without caring which is enabled.
"""

from altcoin_agent.notifier.telegram import (
    Notifier,
    NullNotifier,
    TelegramNotifier,
    build_default_notifier,
)

__all__ = [
    "Notifier",
    "NullNotifier",
    "TelegramNotifier",
    "build_default_notifier",
]
