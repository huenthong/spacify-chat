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

-- 6) RFM snapshots (per customer, periodic) â€" where "frequency" belongs analytically
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

-- =========================================
-- ALPS Stage 2 Extensions
-- =========================================

-- 9) ALPS Conversation Scoring (links to your chats table)
CREATE TABLE IF NOT EXISTS alps_conversations (
  conversation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id            INTEGER UNIQUE NOT NULL,          -- Links to your chats table
  customer_id        INTEGER,                          -- Links to your customers table
  initial_score      REAL NOT NULL,                    -- Stage 1 ALPS score
  current_score      REAL NOT NULL,                    -- Current Stage 2 score
  threshold          REAL DEFAULT 70.0,                -- Routing threshold
  current_handler    TEXT DEFAULT 'bot',               -- 'bot' | 'agent' 
  status            TEXT DEFAULT 'active',             -- 'active' | 'completed' | 'abandoned'
  created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE,
  FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_alps_conv_chat ON alps_conversations(chat_id);
CREATE INDEX IF NOT EXISTS ix_alps_conv_customer ON alps_conversations(customer_id);
CREATE INDEX IF NOT EXISTS ix_alps_conv_score ON alps_conversations(current_score);
CREATE INDEX IF NOT EXISTS ix_alps_conv_handler ON alps_conversations(current_handler);

-- 10) ALPS Score History (tracks score changes over time)
CREATE TABLE IF NOT EXISTS alps_score_history (
  score_id           INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id    INTEGER NOT NULL,
  message_id         INTEGER,                          -- Links to your messages table
  old_score          REAL,
  new_score          REAL NOT NULL,
  score_change       REAL,
  trigger_type       TEXT,                            -- 'initial' | 'message_analysis' | 'manual'
  confidence_level   TEXT,                            -- 'high' | 'medium' | 'low'
  created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(conversation_id) REFERENCES alps_conversations(conversation_id) ON DELETE CASCADE,
  FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_alps_score_conv ON alps_score_history(conversation_id);
CREATE INDEX IF NOT EXISTS ix_alps_score_time ON alps_score_history(created_at);

-- 11) ALPS Signal Analysis (detailed breakdown of detected signals)
CREATE TABLE IF NOT EXISTS alps_signal_analysis (
  analysis_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id         INTEGER NOT NULL,
  conversation_id    INTEGER NOT NULL,
  signal_category    TEXT NOT NULL,                   -- 'timeline' | 'budget' | 'intent' | 'property' | 'qualification'
  signal_name        TEXT NOT NULL,                   -- 'high_urgency' | 'viewing_intent' | etc.
  signal_description TEXT,
  matches_found      INTEGER DEFAULT 0,
  matched_text       TEXT,                            -- JSON array of matched phrases
  weight_applied     REAL,
  score_impact       REAL,
  success_rate       REAL,                            -- Expected success rate for this signal
  created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE CASCADE,
  FOREIGN KEY(conversation_id) REFERENCES alps_conversations(conversation_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_alps_signals_msg ON alps_signal_analysis(message_id);
CREATE INDEX IF NOT EXISTS ix_alps_signals_conv ON alps_signal_analysis(conversation_id);
CREATE INDEX IF NOT EXISTS ix_alps_signals_category ON alps_signal_analysis(signal_category);

-- 12) ALPS Routing Actions (log of bot/agent handoffs)
CREATE TABLE IF NOT EXISTS alps_routing_actions (
  routing_id         INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id    INTEGER NOT NULL,
  message_id         INTEGER,                          -- Message that triggered the routing
  old_handler        TEXT,                            -- 'bot' | 'agent'
  new_handler        TEXT,                            -- 'bot' | 'agent'
  routing_reason     TEXT,                            -- Why the routing happened
  score_at_routing   REAL,
  threshold_used     REAL,
  agent_id           TEXT,                            -- If assigned to specific agent
  created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(conversation_id) REFERENCES alps_conversations(conversation_id) ON DELETE CASCADE,
  FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS ix_alps_routing_conv ON alps_routing_actions(conversation_id);
CREATE INDEX IF NOT EXISTS ix_alps_routing_time ON alps_routing_actions(created_at);

-- =========================================
-- Enhanced Views with ALPS Integration
-- =========================================

-- Convenience view: messages + their chat demographics (great for Streamlit/analysis)
CREATE VIEW IF NOT EXISTS v_messages_with_chat AS
SELECT
  m.message_id, m.chat_id, m.sender_type, m.sent_at, m.text, m.char_len, m.token_len,
  c.customer_id, c.customer_phone_number, c.lead_source, c.budget, c.room_type,
  c.nationality, c.initial_contact_date, c.last_action_date
FROM messages m
JOIN chats c ON c.chat_id = m.chat_id;

-- Enhanced view: Messages with ALPS context
CREATE VIEW IF NOT EXISTS v_messages_with_alps AS
SELECT 
  m.message_id, m.chat_id, m.sender_type, m.sent_at, m.text, m.char_len,
  c.customer_id, c.customer_phone_number, c.lead_source, c.budget, c.room_type, c.nationality,
  ac.conversation_id, ac.initial_score, ac.current_score, ac.current_handler, ac.threshold,
  ash.score_change, ash.confidence_level,
  -- Latest routing action
  (SELECT new_handler FROM alps_routing_actions ara 
   WHERE ara.conversation_id = ac.conversation_id 
   ORDER BY ara.created_at DESC LIMIT 1) as latest_handler_assigned
FROM messages m
JOIN chats c ON c.chat_id = m.chat_id
LEFT JOIN alps_conversations ac ON ac.chat_id = m.chat_id
LEFT JOIN alps_score_history ash ON ash.message_id = m.message_id
ORDER BY m.sent_at DESC;

-- Analytics view: Conversation performance
CREATE VIEW IF NOT EXISTS v_alps_conversation_analytics AS
SELECT 
  ac.conversation_id,
  ac.chat_id,
  ac.customer_id,
  ac.initial_score,
  ac.current_score,
  (ac.current_score - ac.initial_score) as score_improvement,
  ac.current_handler,
  ac.status,
  c.budget,
  c.nationality,
  c.room_type,
  c.lead_source,
  -- Message stats
  COUNT(m.message_id) as total_messages,
  COUNT(CASE WHEN m.sender_type = 'customer' THEN 1 END) as customer_messages,
  COUNT(CASE WHEN m.sender_type = 'agent' THEN 1 END) as agent_messages,
  COUNT(CASE WHEN m.sender_type = 'bot' THEN 1 END) as bot_messages,
  -- Score changes
  (SELECT COUNT(*) FROM alps_score_history ash WHERE ash.conversation_id = ac.conversation_id) as score_changes,
  (SELECT MAX(new_score) FROM alps_score_history ash WHERE ash.conversation_id = ac.conversation_id) as peak_score,
  (SELECT MIN(new_score) FROM alps_score_history ash WHERE ash.conversation_id = ac.conversation_id) as lowest_score,
  -- Routing stats
  (SELECT COUNT(*) FROM alps_routing_actions ara WHERE ara.conversation_id = ac.conversation_id) as routing_changes,
  -- Timing
  ac.created_at as conversation_started,
  ac.updated_at as last_activity,
  (julianday('now') - julianday(ac.created_at)) * 24 * 60 as duration_minutes
FROM alps_conversations ac
JOIN chats c ON c.chat_id = ac.chat_id
LEFT JOIN messages m ON m.chat_id = ac.chat_id
GROUP BY ac.conversation_id, ac.chat_id, ac.customer_id, ac.initial_score, ac.current_score, 
         ac.current_handler, ac.status, c.budget, c.nationality, c.room_type, c.lead_source,
         ac.created_at, ac.updated_at;

-- Signal performance view (for tuning the analyzer)
CREATE VIEW IF NOT EXISTS v_alps_signal_performance AS
SELECT 
  asa.signal_category,
  asa.signal_name,
  asa.signal_description,
  COUNT(*) as total_detections,
  AVG(asa.score_impact) as avg_score_impact,
  AVG(asa.success_rate) as expected_success_rate,
  COUNT(DISTINCT asa.conversation_id) as unique_conversations,
  MIN(asa.created_at) as first_detected,
  MAX(asa.created_at) as last_detected
FROM alps_signal_analysis asa
GROUP BY asa.signal_category, asa.signal_name, asa.signal_description
ORDER BY total_detections DESC, avg_score_impact DESC;

-- Routing efficiency view
CREATE VIEW IF NOT EXISTS v_alps_routing_efficiency AS
SELECT 
  ara.old_handler + ' -> ' + ara.new_handler as routing_type,
  COUNT(*) as routing_count,
  AVG(ara.score_at_routing) as avg_triggering_score,
  AVG(ara.threshold_used) as avg_threshold,
  -- Score improvement after routing
  AVG(
    (SELECT ac.current_score FROM alps_conversations ac WHERE ac.conversation_id = ara.conversation_id)
    - ara.score_at_routing
  ) as avg_score_improvement_post_routing,
  MIN(ara.created_at) as first_routing,
  MAX(ara.created_at) as last_routing
FROM alps_routing_actions ara
GROUP BY ara.old_handler, ara.new_handler
ORDER BY routing_count DESC;
"""

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.executescript(schema)
    conn.execute("PRAGMA journal_mode=WAL;")  # better read concurrency
    conn.commit()
    conn.close()
    print(f"SQLite DB '{DB_NAME}' initialized with ALPS Stage 2 extensions.")
    print("Tables created:")
    print("  - Original: customers, chats, messages, appointments, etc.")
    print("  - ALPS: alps_conversations, alps_score_history, alps_signal_analysis, alps_routing_actions")
    print("  - Views: v_messages_with_alps, v_alps_conversation_analytics, v_alps_signal_performance")

if __name__ == "__main__":
    init_db()
