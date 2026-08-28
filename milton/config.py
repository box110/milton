"""Environment configuration. Milton reads shrub; it does not write to it."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("MILTON_DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("MILTON_DB_PORT", "3384"))
DB_USER = os.getenv("MILTON_DB_USER", "milton_ro")
DB_PASSWORD = os.getenv("MILTON_DB_PASSWORD", "")
DB_NAME = os.getenv("MILTON_DB_NAME", "stonks2")

TENANT_ID = int(os.getenv("MILTON_TENANT_ID", "4"))

APPROVER = os.getenv("MILTON_APPROVER", "")
SUBJECT_TAG = os.getenv("MILTON_SUBJECT_TAG", "milton")

# Milton owns no mail credentials. shrub already has a working mailbox, so it
# sends on milton's behalf and forwards the replies back.
SHRUB_URL = os.getenv("MILTON_SHRUB_URL", "http://app:8000")
INTERNAL_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")

# Milton's own state (proposals and their decisions) — shrub's database is
# read-only to us by design.
STATE_DB = os.getenv("MILTON_STATE_DB", "/app/data/milton.sqlite3")

# Horizons milton scores against. 20d is the primary objective (it is the only
# one where shrub's incumbent weights show a real edge); 5d is carried as a
# secondary check that a fit hasn't simply overfit one horizon.
PRIMARY_HORIZON = 20
SECONDARY_HORIZON = 5
