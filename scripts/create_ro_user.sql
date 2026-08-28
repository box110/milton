-- Milton reads shrub's data and must never be able to change it. The only
-- write it ever performs is an approved weight promotion, and that goes
-- through shrub's own API rather than this connection.
CREATE USER IF NOT EXISTS 'milton_ro'@'%' IDENTIFIED BY 'CHANGE_ME';
GRANT SELECT ON stonks2.discovery_picks        TO 'milton_ro'@'%';
GRANT SELECT ON stonks2.discovery_pick_returns TO 'milton_ro'@'%';
GRANT SELECT ON stonks2.research_briefings     TO 'milton_ro'@'%';
GRANT SELECT ON stonks2.research_documents     TO 'milton_ro'@'%';
GRANT SELECT ON stonks2.research_chunks        TO 'milton_ro'@'%';
GRANT SELECT ON stonks2.tenant_settings        TO 'milton_ro'@'%';
FLUSH PRIVILEGES;
