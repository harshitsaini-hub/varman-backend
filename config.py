"""Backward-compatible import shim for configuration values.

New code should import from ``core.config``. This module remains so existing scripts
and deployments that use ``from config import ...`` continue to work.
"""

from core.config import *  # noqa: F403
