-- SAR Predictor Database Schema
-- Create database and tables for the SAR prediction system

-- Create database
CREATE DATABASE IF NOT EXISTS sar_predictor_db;
USE sar_predictor_db;

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    organization VARCHAR(255),
    role ENUM('admin', 'engineer', 'researcher') DEFAULT 'engineer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Frequency bands table
CREATE TABLE IF NOT EXISTS frequency_bands (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    frequency_range VARCHAR(50) NOT NULL,
    center_frequency DECIMAL(10,3) NOT NULL,
    color_code VARCHAR(7) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Antenna parameters table
CREATE TABLE IF NOT EXISTS antenna_parameters (
    id INT PRIMARY KEY AUTO_INCREMENT,
    substrate_thickness DECIMAL(5,2) NOT NULL,
    permittivity DECIMAL(5,2) NOT NULL,
    patch_width DECIMAL(5,2) NOT NULL,
    patch_length DECIMAL(5,2) NOT NULL,
    feed_position DECIMAL(4,3) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SAR predictions table
CREATE TABLE IF NOT EXISTS sar_predictions (
    id VARCHAR(100) PRIMARY KEY,
    user_id INT,
    frequency_band_id VARCHAR(50),
    antenna_parameters_id INT,
    sar_value DECIMAL(8,5) NOT NULL,
    gain_dbi DECIMAL(6,2) NOT NULL,
    efficiency DECIMAL(5,4) NOT NULL,
    bandwidth_mhz DECIMAL(8,2) NOT NULL,
    safety_status ENUM('safe', 'warning', 'unsafe') NOT NULL,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INT,
    model_version VARCHAR(20) DEFAULT 'v2.1.0',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (frequency_band_id) REFERENCES frequency_bands(id),
    FOREIGN KEY (antenna_parameters_id) REFERENCES antenna_parameters(id),
    INDEX idx_user_timestamp (user_id, prediction_timestamp),
    INDEX idx_band_timestamp (frequency_band_id, prediction_timestamp),
    INDEX idx_safety_status (safety_status)
);

-- S-parameters data table
CREATE TABLE IF NOT EXISTS s_parameters (
    id INT PRIMARY KEY AUTO_INCREMENT,
    prediction_id VARCHAR(100),
    frequency_hz BIGINT NOT NULL,
    s11_db DECIMAL(6,3) NOT NULL,
    s21_db DECIMAL(6,3) DEFAULT NULL,
    s12_db DECIMAL(6,3) DEFAULT NULL,
    s22_db DECIMAL(6,3) DEFAULT NULL,
    FOREIGN KEY (prediction_id) REFERENCES sar_predictions(id) ON DELETE CASCADE,
    INDEX idx_prediction_frequency (prediction_id, frequency_hz)
);

-- Radiation pattern data table
CREATE TABLE IF NOT EXISTS radiation_patterns (
    id INT PRIMARY KEY AUTO_INCREMENT,
    prediction_id VARCHAR(100),
    theta_rad DECIMAL(8,6) NOT NULL,
    phi_rad DECIMAL(8,6) NOT NULL,
    gain_db DECIMAL(6,3) NOT NULL,
    phase_deg DECIMAL(6,2) DEFAULT NULL,
    FOREIGN KEY (prediction_id) REFERENCES sar_predictions(id) ON DELETE CASCADE,
    INDEX idx_prediction_angles (prediction_id, theta_rad, phi_rad)
);

-- System settings table
CREATE TABLE IF NOT EXISTS system_settings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT NOT NULL,
    setting_type ENUM('string', 'number', 'boolean', 'json') DEFAULT 'string',
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- API usage logs table
CREATE TABLE IF NOT EXISTS api_usage_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INT NOT NULL,
    response_time_ms INT,
    request_size_bytes INT,
    response_size_bytes INT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_user_timestamp (user_id, created_at),
    INDEX idx_endpoint_timestamp (endpoint, created_at)
);

-- Create indexes for better performance
CREATE INDEX idx_predictions_timestamp ON sar_predictions(prediction_timestamp DESC);
CREATE INDEX idx_predictions_sar_value ON sar_predictions(sar_value);
CREATE INDEX idx_predictions_safety ON sar_predictions(safety_status, prediction_timestamp DESC);
