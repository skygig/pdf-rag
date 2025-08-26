-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 CHAR(64) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Document versions table (for handling updates)
CREATE TABLE document_versions (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL DEFAULT 1,
    file_size BIGINT,
    page_count INTEGER,
    processing_status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(document_id, version_number)
);

-- Chunks table
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_version_id INTEGER REFERENCES document_versions(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_clean TEXT NOT NULL, -- cleaned version for embeddings
    chunk_hash CHAR(64) NOT NULL, -- sha256(text_clean + document_version_id)
    page_start INTEGER,
    page_end INTEGER,
    heading_path TEXT[], -- array of headings hierarchy
    char_count INTEGER NOT NULL,
    embedding vector(384),
    tsv tsvector, -- for full-text search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(document_version_id, chunk_index)
);

-- Create indexes
CREATE INDEX idx_documents_sha256 ON documents(source_sha256);
CREATE INDEX idx_document_versions_status ON document_versions(processing_status);
CREATE INDEX idx_chunks_document_version ON chunks(document_version_id);
CREATE INDEX idx_chunks_embedding_null ON chunks(id) WHERE embedding IS NULL;

-- Full-text search index
CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv);

-- Vector search index (will be created after data is loaded)
-- CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Trigger to automatically update tsv column
CREATE OR REPLACE FUNCTION update_chunks_tsv() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', NEW.text_clean);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chunks_tsv_trigger
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunks_tsv();
