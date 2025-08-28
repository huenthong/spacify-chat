import sqlite3

DB_NAME = "belive.db"

schema = """
PRAGMA foreign_keys = ON;

-- 1) Canonical customers (optional but useful to de-duplicate across chats)
CREATE TABLE IF NOT EXISTS customers (
  customer_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  phone_e164     TEXT UNIQUE NOT NULL,
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2) Chats (one row per lead/chat session; store demographics captured at start)
CREATE TABLE IF NOT EXISTS chats (
  chat_id               INTEGER PRIMARY KEY AUTOINCREMENT,
  customer_id           INTEGER,                       -- link to canonical person if you want
  -- demographics / preferences captured by the form at chat start:
  customer_phone_number TEXT,
  customer_journey      TEXT,
  location_search       TEXT,
  selected_property     TEXT,
  initial_contact_date  DATE,
  last_action_date      DATE,
  lead_source           TEXT,
  budget                REAL,
  move_in_date          DATE,
  tenancy_period        INTEGER,
  no_of_pax             INTEGER,
  gender                TEXT,
  transportation        TEXT,
  parking               TEXT,
  nationality           TEXT,
  source_from           TEXT,
  combined_lead_source  TEXT,
  reply_within_1h       INTEGER,       -- boolean 0/1
  reply_rate_hour       REAL,
  clean_phone           TEXT,
  room_type             TEXT,
  contact_dayofweek     TEXT,
  contact_hour          INTEGER,
  contact_month         INTEGER,
  recencydays           INTEGER,
  created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_chats_customer ON chats(customer_id);
CREATE INDEX IF NOT EXISTS ix_chats_initial ON chats(initial_contact_date);
CREATE INDEX IF NOT EXISTS ix_chats_last    ON chats(last_action_date);

-- 3) Messages (each row is ONE message under a chat)
CREATE TABLE IF NOT EXISTS messages (
  message_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id        INTEGER NOT NULL,
  sender_type    TEXT NOT NULL,             -- 'customer' | 'agent' | 'bot'
  sent_at        TIMESTAMP NOT NULL,
  text           TEXT,
  char_len       INTEGER,                   -- fill from app as len(text)
  token_len      INTEGER,                   -- optional if you compute tokens
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_msg_chat_time ON messages(chat_id, sent_at);
CREATE INDEX IF NOT EXISTS ix_msg_time      ON messages(sent_at);

-- 4) Appointments (Sheet-2)
CREATE TABLE IF NOT EXISTS appointments (
  appointment_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  customer_id           INTEGER,
  rental_proposed       REAL,
  viewing_status_main   TEXT,               -- Success / Lose/Not Interested / High Chance / Neutral
  viewing_appointment_date DATE,
  start_ts              TIMESTAMP,
  end_ts                TIMESTAMP,
  prospect_name         TEXT,
  viewing_method        TEXT,
  appointment_by        TEXT,
  room_type             TEXT,
  room_number           TEXT,
  unit_number           TEXT,
  remark                TEXT,
  inserted_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_appt_customer_time ON appointments(customer_id, start_ts);
CREATE INDEX IF NOT EXISTS ix_appt_status       ON appointments(viewing_status_main);

-- 5) Optional: link a chat to one/many appointments (probabilistic matching supported)
CREATE TABLE IF NOT EXISTS chat_appointment_links (
  chat_id        INTEGER,
  appointment_id INTEGER,
  link_score     REAL,
  linked_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (chat_id, appointment_id),
  FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE,
  FOREIGN KEY(appointment_id) REFERENCES appointments(appointment_id) ON DELETE CASCADE
);

-- 6) RFM snapshots (per customer, periodic) — where "frequency" belongs analytically
CREATE TABLE IF NOT EXISTS rfm_metrics (
  customer_id    INTEGER,
  as_of_date     DATE NOT NULL,
  recency_days   INTEGER,
  frequency_30d  INTEGER,
  frequency_90d  INTEGER,
  monetary_avg   REAL,
  monetary_max   REAL,
  PRIMARY KEY (customer_id, as_of_date),
  FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);
CREATE INDEX IF NOT EXISTS ix_rfm_asof ON rfm_metrics(as_of_date);

-- 7) Real-time rollups (optional, if you compute them on message ingest)
CREATE TABLE IF NOT EXISTS rt_chat_metrics (
  chat_id                INTEGER PRIMARY KEY,
  msg_count_total        INTEGER,
  msg_count_customer     INTEGER,
  msg_count_agent        INTEGER,
  msg_avg_char_len       REAL,
  first_msg_ts           TIMESTAMP,
  last_msg_ts            TIMESTAMP,
  avg_intermsg_seconds   REAL,
  updated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rt_customer_metrics (
  customer_id            INTEGER PRIMARY KEY,
  msg_count_1h           INTEGER,
  msg_count_24h          INTEGER,
  last_msg_ts            TIMESTAMP,
  avg_char_len_24h       REAL,
  active_chats_open      INTEGER,
  updated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- 8) Feature snapshots (exact rows you train on; versioned for retraining)
-- Keep engineered 'frequency' here if your model uses it.
CREATE TABLE IF NOT EXISTS feature_snapshots (
  feature_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  customer_id                INTEGER,
  customer_phone_number      TEXT,
  customer_journey           TEXT,
  location_search            TEXT,
  selected_property          TEXT,
  initial_contact_date       DATE,
  last_action_date           DATE,
  lead_source                TEXT,
  budget                     REAL,
  move_in_date               DATE,
  tenancy_period             INTEGER,
  no_of_pax                  INTEGER,
  gender                     TEXT,
  transportation             TEXT,
  parking                    TEXT,
  nationality                TEXT,
  source_from                TEXT,
  combined_lead_source       TEXT,
  reply_within_1_hour        INTEGER,
  reply_rate_hour            REAL,
  frequency                  INTEGER,        -- ENGINEERED feature lives here (not in chats)
  clean_phone                TEXT,
  room_type                  TEXT,
  prospect_phone_number      TEXT,
  rental_proposed            REAL,
  viewing_status_main        TEXT,
  contact_dayofweek          TEXT,
  contact_hour               INTEGER,
  contact_month              INTEGER,
  recencydays                INTEGER,
  label_hot_lead             INTEGER,        -- optional derived label (1/0)

  snapshot_date              DATE NOT NULL,  -- when this feature row was materialized
  feature_version            TEXT NOT NULL,  -- bump when feature set changes
  source_run_id              TEXT,
  source_file                TEXT,
  inserted_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_feat_customer ON feature_snapshots(customer_id);
CREATE INDEX IF NOT EXISTS ix_feat_snapshot ON feature_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS ix_feat_label    ON feature_snapshots(label_hot_lead);

-- Convenience view: messages + their chat demographics (great for Streamlit/analysis)
CREATE VIEW IF NOT EXISTS v_messages_with_chat AS
SELECT
  m.message_id, m.chat_id, m.sender_type, m.sent_at, m.text, m.char_len, m.token_len,
  c.customer_id, c.customer_phone_number, c.lead_source, c.budget, c.room_type,
  c.nationality, c.initial_contact_date, c.last_action_date
FROM messages m
JOIN chats c ON c.chat_id = m.chat_id;
"""

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.executescript(schema)
    conn.execute("PRAGMA journal_mode=WAL;")  # better read concurrency
    conn.commit()
    conn.close()
    print(f"SQLite DB '{DB_NAME}' initialized.")

if __name__ == "__main__":
    init_db()
