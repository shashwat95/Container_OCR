\connect template1

DO $$
BEGIN
    -- Create database if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = current_setting('POSTGRES_DB')) THEN
        EXECUTE 'CREATE DATABASE ' || quote_ident(current_setting('POSTGRES_DB'));
    END IF;
END
$$;

\connect :POSTGRES_DB

-- Create the container_data table
CREATE TABLE container_data (
    id SERIAL PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    ocr_output TEXT NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    image_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    track_id VARCHAR(50),
    confidence FLOAT,
    CONSTRAINT chk_camera_id CHECK (camera_id ~ '^CAM_\d+$')
);

-- Create the ocr_results table
CREATE TABLE ocr_results (
    id SERIAL PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    ocr_output TEXT NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    image_path VARCHAR(255),
    track_id INT,
    confidence FLOAT
);

-- Create indexes
CREATE INDEX idx_datetime ON container_data(datetime);
CREATE INDEX idx_camera_id ON container_data(camera_id);
CREATE INDEX idx_ocr_output ON container_data(ocr_output);
CREATE INDEX idx_composite ON container_data(camera_id, datetime);

-- Create view for recent data
CREATE OR REPLACE VIEW recent_container_data AS
SELECT 
    datetime,
    ocr_output,
    camera_id,
    image_path
FROM container_data
WHERE datetime >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY datetime DESC;

-- Create admin user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'admin') THEN
        CREATE USER admin WITH PASSWORD 'admin123' SUPERUSER;
    END IF;
END
$$;

-- Create read-only user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'ocr_readonly') THEN
        CREATE USER ocr_readonly WITH PASSWORD 'readonly_pass';
    END IF;
END
$$;

-- Grant permissions to admin
GRANT ALL PRIVILEGES ON DATABASE :"POSTGRES_DB" TO admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO admin;

-- Grant permissions to read-only user
GRANT CONNECT ON DATABASE :"POSTGRES_DB" TO ocr_readonly;
GRANT USAGE ON SCHEMA public TO ocr_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ocr_readonly;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO ocr_readonly; 