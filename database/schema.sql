-- GymMind 360 metadata table. Data tables are created by pandas.to_sql
-- from the cleaned DataFrames (see utils/cleaning.py SCHEMAS).
CREATE TABLE IF NOT EXISTS _metadata (
    table_name  TEXT PRIMARY KEY,
    row_count   INTEGER NOT NULL,
    source      TEXT NOT NULL,       -- 'demo' or 'import:<filename>'
    updated_at  TEXT NOT NULL
);
