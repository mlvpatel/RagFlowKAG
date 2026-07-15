-- Extensions enabled the first time the Postgres data directory is
-- initialized (files in /docker-entrypoint-initdb.d run on init).
-- vector: dense similarity search. pg_trgm: trigram indexes that serve the
-- knowledge graph's substring entity matching.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
