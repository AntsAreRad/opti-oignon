PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE acquire_write (
                id INTEGER PRIMARY KEY,
                lock_status INTEGER NOT NULL
            );
INSERT INTO "acquire_write" VALUES(1,1);
INSERT INTO "acquire_write" VALUES(2,1);
CREATE TABLE collection_metadata (
    collection_id TEXT REFERENCES collections(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    str_value TEXT,
    int_value INTEGER,
    float_value REAL, bool_value INTEGER,
    PRIMARY KEY (collection_id, key)
);
INSERT INTO "collection_metadata" VALUES('de612eee-908a-4030-867c-5f4eff4cb38f','a',NULL,1,NULL,NULL);
INSERT INTO "collection_metadata" VALUES('7cbad169-9c5b-4ffd-a1f7-ea9148917105','hnsw:space','cosine',NULL,NULL,NULL);
CREATE TABLE "collections" (
    id TEXT PRIMARY KEY, -- unique globally
    name TEXT NOT NULL, -- unique per database
    dimension INTEGER,
    database_id TEXT NOT NULL REFERENCES databases(id) ON DELETE CASCADE, config_json_str TEXT, schema_str TEXT,
    UNIQUE (name, database_id)
);
INSERT INTO "collections" VALUES('12106f70-2eb5-4ad2-9e79-f1c60623e74f','legacy_nested_ip',3,'00000000-0000-0000-0000-000000000000','{"vector_index": {"hnsw": {"ef_construction": 100, "ef_search": 100, "max_neighbors": 16, "resize_factor": 1.2, "sync_threshold": 1000, "space": "ip"}}, "embedding_function": {"type": "legacy"}}',NULL);
INSERT INTO "collections" VALUES('1de6c109-210b-4480-9be9-e0d8f8b14463','legacy_top_ip',3,'00000000-0000-0000-0000-000000000000','{"hnsw": {"ef_construction": 100, "ef_search": 100, "max_neighbors": 16, "resize_factor": 1.2, "sync_threshold": 1000, "space": "ip"}, "spann": null, "embedding_function": {"type": "legacy"}}',NULL);
INSERT INTO "collections" VALUES('402b5b5f-d5f8-4809-91e2-3333246f5e73','legacy_05_ip',3,'00000000-0000-0000-0000-000000000000','{"hnsw_configuration": {"ef_construction": 100, "ef_search": 10, "num_threads": 4, "M": 16, "resize_factor": 1.2, "batch_size": 100, "sync_threshold": 1000, "_type": "HNSWConfigurationInternal", "space": "ip"}, "_type": "CollectionConfigurationInternal"}',NULL);
INSERT INTO "collections" VALUES('de612eee-908a-4030-867c-5f4eff4cb38f','legacy_segment_cosine',3,'00000000-0000-0000-0000-000000000000',NULL,NULL);
INSERT INTO "collections" VALUES('7cbad169-9c5b-4ffd-a1f7-ea9148917105','legacy_05_l2_segment_cosine',3,'00000000-0000-0000-0000-000000000000','{"hnsw_configuration": {"ef_construction": 100, "ef_search": 10, "num_threads": 4, "M": 16, "resize_factor": 1.2, "batch_size": 100, "sync_threshold": 1000, "_type": "HNSWConfigurationInternal", "space": "l2"}, "_type": "CollectionConfigurationInternal"}',NULL);
INSERT INTO "collections" VALUES('4ec1c0ba-e28c-40fa-af26-0e0b6618a1f8','legacy_schema_l2_segment_cosine',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"l2","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"l2","source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}},"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}}}}');
INSERT INTO "collections" VALUES('9d8b4154-39b7-49de-ae3f-77226df0bd78','legacy_schema_ip_segment_cosine',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"ip","embedding_function":{"type":"legacy"},"hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"ip","embedding_function":{"type":"legacy"},"source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}},"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}}}}');
INSERT INTO "collections" VALUES('6fb67c80-d5f3-4024-a839-c6cc1cea3fa9','legacy_nested_ip_segment_cosine',3,'00000000-0000-0000-0000-000000000000','{"vector_index": {"hnsw": {"ef_construction": 100, "ef_search": 100, "max_neighbors": 16, "resize_factor": 1.2, "sync_threshold": 1000, "space": "ip"}}, "embedding_function": {"type": "legacy"}}',NULL);
CREATE TABLE databases (
    id TEXT PRIMARY KEY, -- unique globally
    name TEXT NOT NULL, -- unique per tenant
    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    UNIQUE (tenant_id, name) -- Ensure that a tenant has only one database with a given name
);
INSERT INTO "databases" VALUES('00000000-0000-0000-0000-000000000000','default_database','default_tenant');
CREATE TABLE embedding_metadata (
    id INTEGER REFERENCES embeddings(id),
    key TEXT NOT NULL,
    string_value TEXT,
    int_value INTEGER,
    float_value REAL, bool_value INTEGER,
    PRIMARY KEY (id, key)
);
INSERT INTO "embedding_metadata" VALUES(1,'chroma:document','space text 90',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(2,'chroma:document','space text 91',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(3,'chroma:document','space text 92',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(4,'chroma:document','space text 100',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(5,'chroma:document','space text 101',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(6,'chroma:document','space text 102',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(7,'chroma:document','space text 110',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(8,'chroma:document','space text 111',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(9,'chroma:document','space text 112',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(10,'chroma:document','space text 120',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(11,'chroma:document','space text 121',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(12,'chroma:document','space text 122',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(13,'chroma:document','space text 130',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(14,'chroma:document','space text 131',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(15,'chroma:document','space text 132',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(16,'chroma:document','space text 140',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(17,'chroma:document','space text 141',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(18,'chroma:document','space text 142',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(19,'chroma:document','space text 150',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(20,'chroma:document','space text 151',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(21,'chroma:document','space text 152',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(22,'chroma:document','space text 160',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(23,'chroma:document','space text 161',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(24,'chroma:document','space text 162',NULL,NULL,NULL);
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    segment_id TEXT NOT NULL,
    embedding_id TEXT NOT NULL,
    seq_id BLOB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (segment_id, embedding_id)
);
INSERT INTO "embeddings" VALUES(1,'1487212a-41cc-4e4b-ad65-55bc54504f16','item-90',1,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(2,'1487212a-41cc-4e4b-ad65-55bc54504f16','item-91',2,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(3,'1487212a-41cc-4e4b-ad65-55bc54504f16','item-92',3,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(4,'78d5c073-d4e6-4871-890d-671b8be3ed8f','item-100',4,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(5,'78d5c073-d4e6-4871-890d-671b8be3ed8f','item-101',5,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(6,'78d5c073-d4e6-4871-890d-671b8be3ed8f','item-102',6,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(7,'3aa5f142-71a8-4df7-bb4c-269a820fe41f','item-110',7,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(8,'3aa5f142-71a8-4df7-bb4c-269a820fe41f','item-111',8,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(9,'3aa5f142-71a8-4df7-bb4c-269a820fe41f','item-112',9,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(10,'44caf2c2-8d0d-4a23-bc8c-d164d480274e','item-120',10,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(11,'44caf2c2-8d0d-4a23-bc8c-d164d480274e','item-121',11,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(12,'44caf2c2-8d0d-4a23-bc8c-d164d480274e','item-122',12,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(13,'dfeca8ac-b65b-4d33-8fcf-4aa0e9f45432','item-130',13,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(14,'dfeca8ac-b65b-4d33-8fcf-4aa0e9f45432','item-131',14,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(15,'dfeca8ac-b65b-4d33-8fcf-4aa0e9f45432','item-132',15,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(16,'7f4ff452-50a4-4a10-9ba6-b9f0838889ba','item-140',16,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(17,'7f4ff452-50a4-4a10-9ba6-b9f0838889ba','item-141',17,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(18,'7f4ff452-50a4-4a10-9ba6-b9f0838889ba','item-142',18,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(19,'444936a6-0e43-4fb9-8ba0-5ae292b62ecc','item-150',19,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(20,'444936a6-0e43-4fb9-8ba0-5ae292b62ecc','item-151',20,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(21,'444936a6-0e43-4fb9-8ba0-5ae292b62ecc','item-152',21,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(22,'f4297823-f6e1-4d6a-b6dc-96460e267cc7','item-160',22,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(23,'f4297823-f6e1-4d6a-b6dc-96460e267cc7','item-161',23,'2026-10-25 04:03:39');
INSERT INTO "embeddings" VALUES(24,'f4297823-f6e1-4d6a-b6dc-96460e267cc7','item-162',24,'2026-10-25 04:03:39');
CREATE TABLE embeddings_queue (
    seq_id INTEGER PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    operation INTEGER NOT NULL,
    topic TEXT NOT NULL,
    id TEXT NOT NULL,
    vector BLOB,
    encoding TEXT,
    metadata TEXT
);
INSERT INTO "embeddings_queue" VALUES(1,'2026-10-25 04:03:39',0,'persistent://default/default/12106f70-2eb5-4ad2-9e79-f1c60623e74f','item-90',X'000060BF000000BF000000BE','FLOAT32','{"chroma:document":"space text 90"}');
INSERT INTO "embeddings_queue" VALUES(2,'2026-10-25 04:03:39',0,'persistent://default/default/12106f70-2eb5-4ad2-9e79-f1c60623e74f','item-91',X'000000000000C03E0000403F','FLOAT32','{"chroma:document":"space text 91"}');
INSERT INTO "embeddings_queue" VALUES(3,'2026-10-25 04:03:39',0,'persistent://default/default/12106f70-2eb5-4ad2-9e79-f1c60623e74f','item-92',X'0000603F000060BF000000BF','FLOAT32','{"chroma:document":"space text 92"}');
INSERT INTO "embeddings_queue" VALUES(4,'2026-10-25 04:03:39',0,'persistent://default/default/1de6c109-210b-4480-9be9-e0d8f8b14463','item-100',X'000020BF000080BE0000003E','FLOAT32','{"chroma:document":"space text 100"}');
INSERT INTO "embeddings_queue" VALUES(5,'2026-10-25 04:03:39',0,'persistent://default/default/1de6c109-210b-4480-9be9-e0d8f8b14463','item-101',X'0000803E0000203F0000803F','FLOAT32','{"chroma:document":"space text 101"}');
INSERT INTO "embeddings_queue" VALUES(6,'2026-10-25 04:03:39',0,'persistent://default/default/1de6c109-210b-4480-9be9-e0d8f8b14463','item-102',X'000080BF000020BF000080BE','FLOAT32','{"chroma:document":"space text 102"}');
INSERT INTO "embeddings_queue" VALUES(7,'2026-10-25 04:03:39',0,'persistent://default/default/402b5b5f-d5f8-4809-91e2-3333246f5e73','item-110',X'0000C0BE000000000000C03E','FLOAT32','{"chroma:document":"space text 110"}');
INSERT INTO "embeddings_queue" VALUES(8,'2026-10-25 04:03:39',0,'persistent://default/default/402b5b5f-d5f8-4809-91e2-3333246f5e73','item-111',X'0000003F0000603F000060BF','FLOAT32','{"chroma:document":"space text 111"}');
INSERT INTO "embeddings_queue" VALUES(9,'2026-10-25 04:03:39',0,'persistent://default/default/402b5b5f-d5f8-4809-91e2-3333246f5e73','item-112',X'000040BF0000C0BE00000000','FLOAT32','{"chroma:document":"space text 112"}');
INSERT INTO "embeddings_queue" VALUES(10,'2026-10-25 04:03:39',0,'persistent://default/default/de612eee-908a-4030-867c-5f4eff4cb38f','item-120',X'000000BE0000803E0000203F','FLOAT32','{"chroma:document":"space text 120"}');
INSERT INTO "embeddings_queue" VALUES(11,'2026-10-25 04:03:39',0,'persistent://default/default/de612eee-908a-4030-867c-5f4eff4cb38f','item-121',X'0000403F000080BF000020BF','FLOAT32','{"chroma:document":"space text 121"}');
INSERT INTO "embeddings_queue" VALUES(12,'2026-10-25 04:03:39',0,'persistent://default/default/de612eee-908a-4030-867c-5f4eff4cb38f','item-122',X'000000BF000000BE0000803E','FLOAT32','{"chroma:document":"space text 122"}');
INSERT INTO "embeddings_queue" VALUES(13,'2026-10-25 04:03:39',0,'persistent://default/default/7cbad169-9c5b-4ffd-a1f7-ea9148917105','item-130',X'0000003E0000003F0000603F','FLOAT32','{"chroma:document":"space text 130"}');
INSERT INTO "embeddings_queue" VALUES(14,'2026-10-25 04:03:39',0,'persistent://default/default/7cbad169-9c5b-4ffd-a1f7-ea9148917105','item-131',X'0000803F000040BF0000C0BE','FLOAT32','{"chroma:document":"space text 131"}');
INSERT INTO "embeddings_queue" VALUES(15,'2026-10-25 04:03:39',0,'persistent://default/default/7cbad169-9c5b-4ffd-a1f7-ea9148917105','item-132',X'000080BE0000003E0000003F','FLOAT32','{"chroma:document":"space text 132"}');
INSERT INTO "embeddings_queue" VALUES(16,'2026-10-25 04:03:39',0,'persistent://default/default/4ec1c0ba-e28c-40fa-af26-0e0b6618a1f8','item-140',X'0000C03E0000403F000080BF','FLOAT32','{"chroma:document":"space text 140"}');
INSERT INTO "embeddings_queue" VALUES(17,'2026-10-25 04:03:39',0,'persistent://default/default/4ec1c0ba-e28c-40fa-af26-0e0b6618a1f8','item-141',X'000060BF000000BF000000BE','FLOAT32','{"chroma:document":"space text 141"}');
INSERT INTO "embeddings_queue" VALUES(18,'2026-10-25 04:03:39',0,'persistent://default/default/4ec1c0ba-e28c-40fa-af26-0e0b6618a1f8','item-142',X'000000000000C03E0000403F','FLOAT32','{"chroma:document":"space text 142"}');
INSERT INTO "embeddings_queue" VALUES(19,'2026-10-25 04:03:39',0,'persistent://default/default/9d8b4154-39b7-49de-ae3f-77226df0bd78','item-150',X'0000203F0000803F000040BF','FLOAT32','{"chroma:document":"space text 150"}');
INSERT INTO "embeddings_queue" VALUES(20,'2026-10-25 04:03:39',0,'persistent://default/default/9d8b4154-39b7-49de-ae3f-77226df0bd78','item-151',X'000020BF000080BE0000003E','FLOAT32','{"chroma:document":"space text 151"}');
INSERT INTO "embeddings_queue" VALUES(21,'2026-10-25 04:03:39',0,'persistent://default/default/9d8b4154-39b7-49de-ae3f-77226df0bd78','item-152',X'0000803E0000203F0000803F','FLOAT32','{"chroma:document":"space text 152"}');
INSERT INTO "embeddings_queue" VALUES(22,'2026-10-25 04:03:39',0,'persistent://default/default/6fb67c80-d5f3-4024-a839-c6cc1cea3fa9','item-160',X'0000603F000060BF000000BF','FLOAT32','{"chroma:document":"space text 160"}');
INSERT INTO "embeddings_queue" VALUES(23,'2026-10-25 04:03:39',0,'persistent://default/default/6fb67c80-d5f3-4024-a839-c6cc1cea3fa9','item-161',X'0000C0BE000000000000C03E','FLOAT32','{"chroma:document":"space text 161"}');
INSERT INTO "embeddings_queue" VALUES(24,'2026-10-25 04:03:39',0,'persistent://default/default/6fb67c80-d5f3-4024-a839-c6cc1cea3fa9','item-162',X'0000003F0000603F000060BF','FLOAT32','{"chroma:document":"space text 162"}');
CREATE TABLE embeddings_queue_config (
    id INTEGER PRIMARY KEY,
    config_json_str TEXT
);
INSERT INTO "embeddings_queue_config" VALUES(1,'{"automatically_purge":true,"_type":"EmbeddingsQueueConfigurationInternal"}');
CREATE TABLE maintenance_log (
  id INT PRIMARY KEY,
  timestamp INT NOT NULL,
  operation TEXT NOT NULL
);
CREATE TABLE max_seq_id (
    segment_id TEXT PRIMARY KEY,
    seq_id INTEGER);
INSERT INTO "max_seq_id" VALUES('1487212a-41cc-4e4b-ad65-55bc54504f16',3);
INSERT INTO "max_seq_id" VALUES('78d5c073-d4e6-4871-890d-671b8be3ed8f',6);
INSERT INTO "max_seq_id" VALUES('3aa5f142-71a8-4df7-bb4c-269a820fe41f',9);
INSERT INTO "max_seq_id" VALUES('44caf2c2-8d0d-4a23-bc8c-d164d480274e',12);
INSERT INTO "max_seq_id" VALUES('dfeca8ac-b65b-4d33-8fcf-4aa0e9f45432',15);
INSERT INTO "max_seq_id" VALUES('7f4ff452-50a4-4a10-9ba6-b9f0838889ba',18);
INSERT INTO "max_seq_id" VALUES('444936a6-0e43-4fb9-8ba0-5ae292b62ecc',21);
INSERT INTO "max_seq_id" VALUES('f4297823-f6e1-4d6a-b6dc-96460e267cc7',24);
CREATE TABLE migrations (
                dir TEXT NOT NULL,
                version INTEGER NOT NULL,
                filename TEXT NOT NULL,
                sql TEXT NOT NULL,
                hash TEXT NOT NULL,
                PRIMARY KEY (dir, version)
            );
INSERT INTO "migrations" VALUES('sysdb',1,'00001-collections.sqlite.sql','CREATE TABLE collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    topic TEXT NOT NULL,
    UNIQUE (name)
);

CREATE TABLE collection_metadata (
    collection_id TEXT REFERENCES collections(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    str_value TEXT,
    int_value INTEGER,
    float_value REAL,
    PRIMARY KEY (collection_id, key)
);
','38352d725ad1c16074fac420b22b4633');
INSERT INTO "migrations" VALUES('sysdb',2,'00002-segments.sqlite.sql','CREATE TABLE segments (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    topic TEXT,
    collection TEXT REFERENCES collection(id)
);

CREATE TABLE segment_metadata (
    segment_id TEXT  REFERENCES segments(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    str_value TEXT,
    int_value INTEGER,
    float_value REAL,
    PRIMARY KEY (segment_id, key)
);
','2913cb6a503055a95f625448037e8912');
INSERT INTO "migrations" VALUES('sysdb',3,'00003-collection-dimension.sqlite.sql','ALTER TABLE collections ADD COLUMN dimension INTEGER;
','42d22d0574d31d419c2a0e7f625c93aa');
INSERT INTO "migrations" VALUES('sysdb',4,'00004-tenants-databases.sqlite.sql','CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    UNIQUE (id)
);

CREATE TABLE IF NOT EXISTS databases (
    id TEXT PRIMARY KEY, -- unique globally
    name TEXT NOT NULL, -- unique per tenant
    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    UNIQUE (tenant_id, name) -- Ensure that a tenant has only one database with a given name
);

CREATE TABLE IF NOT EXISTS collections_tmp (
    id TEXT PRIMARY KEY, -- unique globally
    name TEXT NOT NULL, -- unique per database
    topic TEXT NOT NULL,
    dimension INTEGER,
    database_id TEXT NOT NULL REFERENCES databases(id) ON DELETE CASCADE,
    UNIQUE (name, database_id)
);

-- Create default tenant and database
INSERT OR REPLACE INTO tenants (id) VALUES (''default_tenant''); -- The default tenant id is ''default_tenant'' others are UUIDs
INSERT OR REPLACE INTO databases (id, name, tenant_id) VALUES (''00000000-0000-0000-0000-000000000000'', ''default_database'', ''default_tenant'');

INSERT OR REPLACE INTO collections_tmp (id, name, topic, dimension, database_id)
    SELECT id, name, topic, dimension, ''00000000-0000-0000-0000-000000000000'' FROM collections;
DROP TABLE collections;
ALTER TABLE collections_tmp RENAME TO collections;
','048867ce8fcdefe4023c7110e4433591');
INSERT INTO "migrations" VALUES('sysdb',5,'00005-remove-topic.sqlite.sql','-- Remove the topic column from the Collections and Segments tables

ALTER TABLE collections DROP COLUMN topic;
ALTER TABLE segments DROP COLUMN topic;
','b1367c826b8fba5f96f27befdc1d42d2');
INSERT INTO "migrations" VALUES('sysdb',6,'00006-collection-segment-metadata.sqlite.sql','-- SQLite does not support adding check with alter table, as a result, adding a check
-- involve creating a new table and copying the data over. It is over kill with adding
-- a boolean type column. The application write to the table needs to ensure the data
-- integrity.
ALTER TABLE collection_metadata ADD COLUMN bool_value INTEGER;
ALTER TABLE segment_metadata ADD COLUMN bool_value INTEGER;
','4eea7468935bf25d4604a0fed2366116');
INSERT INTO "migrations" VALUES('sysdb',7,'00007-collection-config.sqlite.sql','-- Stores collection configuration dictionaries.
ALTER TABLE collections ADD COLUMN config_json_str TEXT;
','1c7e63bba346a42a18b6ab7f1c989bed');
INSERT INTO "migrations" VALUES('sysdb',8,'00008-maintenance-log.sqlite.sql','-- Records when database maintenance operations are performed.
-- At time of creation, this table is only used to record vacuum operations.
CREATE TABLE maintenance_log (
  id INT PRIMARY KEY,
  timestamp INT NOT NULL,
  operation TEXT NOT NULL
);
','0a0e7e93111a01789addf64961c6127c');
INSERT INTO "migrations" VALUES('sysdb',9,'00009-segment-collection-not-null.sqlite.sql','-- This makes segments.collection non-nullable.
CREATE TABLE segments_temp (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    collection TEXT REFERENCES collection(id) NOT NULL
);

INSERT INTO segments_temp SELECT * FROM segments;
DROP TABLE segments;
ALTER TABLE segments_temp RENAME TO segments;
','054355aef9e63702bf54ea29e61563f1');
INSERT INTO "migrations" VALUES('sysdb',10,'00010-collection-schema.sqlite.sql','-- Stores collection schema as stringified json
ALTER TABLE collections ADD COLUMN schema_str TEXT;
','5c3a5ac4b79df76799b4721827ed5e1d');
INSERT INTO "migrations" VALUES('metadb',1,'00001-embedding-metadata.sqlite.sql','CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    segment_id TEXT NOT NULL,
    embedding_id TEXT NOT NULL,
    seq_id BLOB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (segment_id, embedding_id)
);

CREATE TABLE embedding_metadata (
    id INTEGER REFERENCES embeddings(id),
    key TEXT NOT NULL,
    string_value TEXT,
    int_value INTEGER,
    float_value REAL,
    PRIMARY KEY (id, key)
);

CREATE TABLE max_seq_id (
    segment_id TEXT PRIMARY KEY,
    seq_id BLOB NOT NULL
);

CREATE VIRTUAL TABLE embedding_fulltext USING fts5(id, string_value);
','2b4cf52c4bb2676e21d6860a4409f856');
INSERT INTO "migrations" VALUES('metadb',2,'00002-embedding-metadata.sqlite.sql','-- SQLite does not support adding check with alter table, as a result, adding a check
-- involve creating a new table and copying the data over. It is over kill with adding
-- a boolean type column. The application write to the table needs to ensure the data
-- integrity.
ALTER TABLE embedding_metadata ADD COLUMN bool_value INTEGER
','12a570f7121b3a8ce750a2a7c36da20f');
INSERT INTO "migrations" VALUES('metadb',3,'00003-full-text-tokenize.sqlite.sql','CREATE VIRTUAL TABLE embedding_fulltext_search USING fts5(string_value, tokenize=''trigram'');
INSERT INTO embedding_fulltext_search (rowid, string_value) SELECT rowid, string_value FROM embedding_metadata;
DROP TABLE embedding_fulltext;
','f97ad6334aeaa8f419f01110b648b97a');
INSERT INTO "migrations" VALUES('metadb',4,'00004-metadata-indices.sqlite.sql','CREATE INDEX IF NOT EXISTS embedding_metadata_int_value ON embedding_metadata (key, int_value) WHERE int_value IS NOT NULL;
CREATE INDEX IF NOT EXISTS embedding_metadata_float_value ON embedding_metadata (key, float_value) WHERE float_value IS NOT NULL;
CREATE INDEX IF NOT EXISTS embedding_metadata_string_value ON embedding_metadata (key, string_value) WHERE string_value IS NOT NULL;
','fb36603a45ee2cd0254cef3ef86585e8');
INSERT INTO "migrations" VALUES('metadb',5,'00005-max-seq-id-int.sqlite.sql','ALTER TABLE max_seq_id ADD COLUMN int_seq_id INTEGER;

-- Convert 8 byte wide big-endian integer as blob to native 64 bit integer.
-- Adapted from https://stackoverflow.com/a/70296198.
UPDATE max_seq_id SET int_seq_id = (
  SELECT (
       (instr(''123456789ABCDEF'', substr(hex(seq_id), -1 , 1)) <<  0)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -2 , 1)) <<  4)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -3 , 1)) <<  8)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -4 , 1)) << 12)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -5 , 1)) << 16)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -6 , 1)) << 20)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -7 , 1)) << 24)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -8 , 1)) << 28)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -9 , 1)) << 32)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -10, 1)) << 36)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -11, 1)) << 40)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -12, 1)) << 44)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -13, 1)) << 48)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -14, 1)) << 52)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -15, 1)) << 56)
     | (instr(''123456789ABCDEF'', substr(hex(seq_id), -16, 1)) << 60)
    )
);

ALTER TABLE max_seq_id DROP COLUMN seq_id;
ALTER TABLE max_seq_id RENAME COLUMN int_seq_id TO seq_id;
','0e9de46758761b373ce682925edcc326');
INSERT INTO "migrations" VALUES('embeddings_queue',1,'00001-embeddings.sqlite.sql','CREATE TABLE embeddings_queue (
    seq_id INTEGER PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    operation INTEGER NOT NULL,
    topic TEXT NOT NULL,
    id TEXT NOT NULL,
    vector BLOB,
    encoding TEXT,
    metadata TEXT
);
','d3755dfd232be8e8301f4d7fcfb3a486');
INSERT INTO "migrations" VALUES('embeddings_queue',2,'00002-embeddings-queue-config.sqlite.sql','CREATE TABLE embeddings_queue_config (
    id INTEGER PRIMARY KEY,
    config_json_str TEXT
);
','8fbfe4ffb3e57f1d8bfdc58510a82e85');
CREATE TABLE segment_metadata (
    segment_id TEXT  REFERENCES segments(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    str_value TEXT,
    int_value INTEGER,
    float_value REAL, bool_value INTEGER,
    PRIMARY KEY (segment_id, key)
);
INSERT INTO "segment_metadata" VALUES('7e4e5580-2de5-4d61-942e-0ea3331ae32c','hnsw:space','cosine',NULL,NULL,NULL);
INSERT INTO "segment_metadata" VALUES('f202132b-3855-47f9-ac51-dfc070d21bee','hnsw:space','cosine',NULL,NULL,NULL);
INSERT INTO "segment_metadata" VALUES('78f3d1e2-73b3-4e63-8f06-3403a24fe9c2','hnsw:space','cosine',NULL,NULL,NULL);
INSERT INTO "segment_metadata" VALUES('9a3db14f-95e6-4458-bb23-c8099677c232','hnsw:space','cosine',NULL,NULL,NULL);
INSERT INTO "segment_metadata" VALUES('c1e125a4-7a35-4433-9ad5-d224a6565e5f','hnsw:space','cosine',NULL,NULL,NULL);
CREATE TABLE "segments" (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    collection TEXT REFERENCES collection(id) NOT NULL
);
INSERT INTO "segments" VALUES('92701411-3af3-4441-ab88-4b675da138fd','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','12106f70-2eb5-4ad2-9e79-f1c60623e74f');
INSERT INTO "segments" VALUES('1487212a-41cc-4e4b-ad65-55bc54504f16','urn:chroma:segment/metadata/sqlite','METADATA','12106f70-2eb5-4ad2-9e79-f1c60623e74f');
INSERT INTO "segments" VALUES('11660dd5-e082-408a-a1ab-1c183b73ac32','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','1de6c109-210b-4480-9be9-e0d8f8b14463');
INSERT INTO "segments" VALUES('78d5c073-d4e6-4871-890d-671b8be3ed8f','urn:chroma:segment/metadata/sqlite','METADATA','1de6c109-210b-4480-9be9-e0d8f8b14463');
INSERT INTO "segments" VALUES('1ac25a97-0d63-4aa3-a5ee-bf6d51ca3bd5','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','402b5b5f-d5f8-4809-91e2-3333246f5e73');
INSERT INTO "segments" VALUES('3aa5f142-71a8-4df7-bb4c-269a820fe41f','urn:chroma:segment/metadata/sqlite','METADATA','402b5b5f-d5f8-4809-91e2-3333246f5e73');
INSERT INTO "segments" VALUES('7e4e5580-2de5-4d61-942e-0ea3331ae32c','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','de612eee-908a-4030-867c-5f4eff4cb38f');
INSERT INTO "segments" VALUES('44caf2c2-8d0d-4a23-bc8c-d164d480274e','urn:chroma:segment/metadata/sqlite','METADATA','de612eee-908a-4030-867c-5f4eff4cb38f');
INSERT INTO "segments" VALUES('f202132b-3855-47f9-ac51-dfc070d21bee','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','7cbad169-9c5b-4ffd-a1f7-ea9148917105');
INSERT INTO "segments" VALUES('dfeca8ac-b65b-4d33-8fcf-4aa0e9f45432','urn:chroma:segment/metadata/sqlite','METADATA','7cbad169-9c5b-4ffd-a1f7-ea9148917105');
INSERT INTO "segments" VALUES('78f3d1e2-73b3-4e63-8f06-3403a24fe9c2','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','4ec1c0ba-e28c-40fa-af26-0e0b6618a1f8');
INSERT INTO "segments" VALUES('7f4ff452-50a4-4a10-9ba6-b9f0838889ba','urn:chroma:segment/metadata/sqlite','METADATA','4ec1c0ba-e28c-40fa-af26-0e0b6618a1f8');
INSERT INTO "segments" VALUES('9a3db14f-95e6-4458-bb23-c8099677c232','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','9d8b4154-39b7-49de-ae3f-77226df0bd78');
INSERT INTO "segments" VALUES('444936a6-0e43-4fb9-8ba0-5ae292b62ecc','urn:chroma:segment/metadata/sqlite','METADATA','9d8b4154-39b7-49de-ae3f-77226df0bd78');
INSERT INTO "segments" VALUES('c1e125a4-7a35-4433-9ad5-d224a6565e5f','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','6fb67c80-d5f3-4024-a839-c6cc1cea3fa9');
INSERT INTO "segments" VALUES('f4297823-f6e1-4d6a-b6dc-96460e267cc7','urn:chroma:segment/metadata/sqlite','METADATA','6fb67c80-d5f3-4024-a839-c6cc1cea3fa9');
CREATE TABLE tenants (
    id TEXT PRIMARY KEY,
    UNIQUE (id)
);
INSERT INTO "tenants" VALUES('default_tenant');
CREATE INDEX embedding_metadata_int_value ON embedding_metadata (key, int_value) WHERE int_value IS NOT NULL;
CREATE INDEX embedding_metadata_float_value ON embedding_metadata (key, float_value) WHERE float_value IS NOT NULL;
CREATE INDEX embedding_metadata_string_value ON embedding_metadata (key, string_value) WHERE string_value IS NOT NULL;
CREATE VIRTUAL TABLE embedding_fulltext_search USING fts5(string_value, tokenize='trigram');
COMMIT;
