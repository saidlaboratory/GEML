#!/usr/bin/env python3
"""Perform a credential-presence preflight without network or paid model calls."""

from __future__ import annotations

import argparse
import os

KEY_ENVIRONMENT = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=tuple(sorted(KEY_ENVIRONMENT)), required=True)
    parser.add_argument("--require-key", action="store_true")
    arguments = parser.parse_args()
    variable = KEY_ENVIRONMENT[arguments.provider]
    configured = bool(os.environ.get(variable))
    print(f"provider={arguments.provider}")
    print(f"credential_environment={variable}")
    print(f"credential_configured={configured}")
    print("network_calls=0")
    print("paid_calls=0")
    return 0 if configured or not arguments.require_key else 2


if __name__ == "__main__":
    raise SystemExit(main())
