-- Seed initial data for SAR Predictor

USE sar_predictor_db;

-- Insert frequency bands
INSERT INTO frequency_bands (id, name, frequency_range, center_frequency, color_code, description) VALUES
('x-band', 'X-band', '8-12 GHz', 10.000, '#3b82f6', 'X-band frequency range commonly used for radar and satellite communications'),
('ku-band', 'Ku-band', '12-18 GHz', 15.000, '#8b5cf6', 'Ku-band frequency range used for satellite communications and broadcasting'),
('k-band', 'K-band', '18-27 GHz', 22.500, '#06b6d4', 'K-band frequency range used for radar and satellite communications'),
('ka-band', 'Ka-band', '27-40 GHz', 33.500, '#10b981', 'Ka-band frequency range used for high-speed satellite communications'),
('v-band', 'V-band', '40-75 GHz', 57.500, '#f59e0b', 'V-band frequency range used for millimeter wave communications'),
('w-band', 'W-band', '75-110 GHz', 92.500, '#ef4444', 'W-band frequency range used for automotive radar and 5G'),
('d-band', 'D-band', '110-170 GHz', 140.000, '#ec4899', 'D-band frequency range used for future 6G communications');

-- Insert system settings
INSERT INTO system_settings (setting_key, setting_value, setting_type, description) VALUES
('sar_limit_fcc', '1.6', 'number', 'FCC SAR limit in W/kg'),
('sar_limit_icnirp', '2.0', 'number', 'ICNIRP SAR limit in W/kg'),
('default_sar_standard', 'fcc', 'string', 'Default SAR safety standard'),
('max_predictions_per_user', '1000', 'number', 'Maximum predictions per user'),
('data_retention_days', '365', 'number', 'Data retention period in days'),
('api_rate_limit', '100', 'number', 'API requests per minute per user'),
('model_version', 'v2.1.0', 'string', 'Current prediction model version'),
('enable_auto_cleanup', 'true', 'boolean', 'Enable automatic data cleanup'),
('notification_email', 'admin@sarpredictor.com', 'string', 'System notification email'),
('maintenance_mode', 'false', 'boolean', 'System maintenance mode flag');

-- Insert sample user (for testing)
INSERT INTO users (email, password_hash, first_name, last_name, organization, role) VALUES
('admin@sarpredictor.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', 'Admin', 'User', 'SAR Predictor Inc.', 'admin'),
('engineer@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', 'John', 'Engineer', 'Tech Corp', 'engineer'),
('researcher@university.edu', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', 'Dr. Jane', 'Smith', 'University Research Lab', 'researcher');

-- Insert sample antenna parameters
INSERT INTO antenna_parameters (substrate_thickness, permittivity, patch_width, patch_length, feed_position) VALUES
(1.6, 4.4, 12.0, 15.0, 0.25),
(1.2, 3.8, 8.0, 10.0, 0.30),
(0.8, 3.2, 6.0, 7.5, 0.35),
(0.6, 2.8, 4.5, 5.5, 0.40),
(0.4, 2.4, 3.0, 3.8, 0.45),
(0.3, 2.2, 2.0, 2.5, 0.50);

COMMIT;
