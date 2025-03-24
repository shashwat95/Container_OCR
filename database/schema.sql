-- First connect to postgres and create database
\c postgres
DROP DATABASE IF EXISTS ocr_system;
CREATE DATABASE ocr_system;

-- Connect to our database
\c ocr_system

-- Create the container_data table (without partitioning)
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

-- Create read-only user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'ocr_readonly') THEN
        CREATE USER ocr_readonly WITH PASSWORD 'readonly_pass';
    END IF;
END
$$;

-- Grant permissions
GRANT CONNECT ON DATABASE ocr_system TO ocr_readonly;
GRANT USAGE ON SCHEMA public TO ocr_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ocr_readonly;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO ocr_readonly; 