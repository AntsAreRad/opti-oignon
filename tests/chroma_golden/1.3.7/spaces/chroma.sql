PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE acquire_write (
                id INTEGER PRIMARY KEY,
                lock_status INTEGER NOT NULL
            );
INSERT INTO "acquire_write" VALUES(1,1);
CREATE TABLE collection_metadata (
    collection_id TEXT REFERENCES collections(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    str_value TEXT,
    int_value INTEGER,
    float_value REAL, bool_value INTEGER,
    PRIMARY KEY (collection_id, key)
);
INSERT INTO "collection_metadata" VALUES('52026c96-583b-4d00-90ba-2ec96259a3a5','b','x',NULL,NULL,NULL);
INSERT INTO "collection_metadata" VALUES('3f9da12a-8eee-46da-a02e-9877508a32e5','hnsw:space','cosine',NULL,NULL,NULL);
CREATE TABLE "collections" (
    id TEXT PRIMARY KEY, -- unique globally
    name TEXT NOT NULL, -- unique per database
    dimension INTEGER,
    database_id TEXT NOT NULL REFERENCES databases(id) ON DELETE CASCADE, config_json_str TEXT, schema_str TEXT,
    UNIQUE (name, database_id)
);
INSERT INTO "collections" VALUES('3c5add09-d3dc-4924-92b9-e0814d7277ac','conf_ip',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"ip","embedding_function":{"type":"legacy"},"hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}},"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"ip","embedding_function":{"type":"legacy"},"source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}}}}');
INSERT INTO "collections" VALUES('2a83550c-4d42-41c7-9217-6754a3e2cfd6','schema_ip',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"ip","embedding_function":{"type":"known","name":"default","config":{}},"hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}},"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"ip","embedding_function":{"type":"known","name":"default","config":{}},"source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}}}}');
INSERT INTO "collections" VALUES('52026c96-583b-4d00-90ba-2ec96259a3a5','modified',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"cosine","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"cosine","source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}},"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}}}}');
INSERT INTO "collections" VALUES('3f9da12a-8eee-46da-a02e-9877508a32e5','conflict',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"ip","embedding_function":{"type":"legacy"},"hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"ip","embedding_function":{"type":"legacy"},"source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}},"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}}}}');
INSERT INTO "collections" VALUES('226a1045-0409-4b16-b4b5-6ad2d5b8f136','nometa',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"l2","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"l2","source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}},"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}}}}');
INSERT INTO "collections" VALUES('57ceb827-d818-40c6-a579-974efe86b376','holders_ip',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"cosine","embedding_function":{"type":"known","name":"default","config":{}},"hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"ip","embedding_function":{"type":"known","name":"default","config":{}},"source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}},"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}}}}');
INSERT INTO "collections" VALUES('d1d78f82-28a8-4ca9-a9e3-676e5c05414a','holders_unset',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"cosine","embedding_function":{"type":"known","name":"default","config":{}},"hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"l2","embedding_function":{"type":"known","name":"default","config":{}},"source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}},"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}}}}');
INSERT INTO "collections" VALUES('7bbd7d1a-ed7e-4b63-9e94-51d33d937a05','conf_spann_ip',3,'00000000-0000-0000-0000-000000000000','{}','{"defaults":{"string":{"fts_index":{"enabled":false,"config":{}},"string_inverted_index":{"enabled":true,"config":{}}},"float_list":{"vector_index":{"enabled":false,"config":{"space":"ip","embedding_function":{"type":"legacy"},"hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}},"sparse_vector":{"sparse_vector_index":{"enabled":false,"config":{"embedding_function":{"type":"unknown"},"bm25":false}}},"int":{"int_inverted_index":{"enabled":true,"config":{}}},"float":{"float_inverted_index":{"enabled":true,"config":{}}},"bool":{"bool_inverted_index":{"enabled":true,"config":{}}}},"keys":{"#embedding":{"float_list":{"vector_index":{"enabled":true,"config":{"space":"ip","embedding_function":{"type":"legacy"},"source_key":"#document","hnsw":{"ef_construction":100,"max_neighbors":16,"ef_search":100,"num_threads":24,"batch_size":100,"sync_threshold":1000,"resize_factor":1.2}}}}},"#document":{"string":{"fts_index":{"enabled":true,"config":{}},"string_inverted_index":{"enabled":false,"config":{}}}}}}');
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
INSERT INTO "embedding_metadata" VALUES(1,'chroma:document','space text 20',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(2,'chroma:document','space text 21',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(3,'chroma:document','space text 22',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(4,'chroma:document','space text 30',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(5,'chroma:document','space text 31',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(6,'chroma:document','space text 32',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(7,'chroma:document','space text 40',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(8,'chroma:document','space text 41',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(9,'chroma:document','space text 42',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(10,'chroma:document','space text 50',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(11,'chroma:document','space text 51',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(12,'chroma:document','space text 52',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(13,'chroma:document','space text 60',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(14,'chroma:document','space text 61',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(15,'chroma:document','space text 62',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(16,'chroma:document','space text 70',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(17,'chroma:document','space text 71',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(18,'chroma:document','space text 72',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(19,'chroma:document','space text 80',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(20,'chroma:document','space text 81',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(21,'chroma:document','space text 82',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(22,'chroma:document','space text 90',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(23,'chroma:document','space text 91',NULL,NULL,NULL);
INSERT INTO "embedding_metadata" VALUES(24,'chroma:document','space text 92',NULL,NULL,NULL);
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    segment_id TEXT NOT NULL,
    embedding_id TEXT NOT NULL,
    seq_id BLOB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (segment_id, embedding_id)
);
INSERT INTO "embeddings" VALUES(1,'692d97f2-a016-4444-8019-ba4e26ff7e73','item-20',1,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(2,'692d97f2-a016-4444-8019-ba4e26ff7e73','item-21',2,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(3,'692d97f2-a016-4444-8019-ba4e26ff7e73','item-22',3,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(4,'b4cc2eab-65b3-415f-873e-8af7888486a9','item-30',4,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(5,'b4cc2eab-65b3-415f-873e-8af7888486a9','item-31',5,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(6,'b4cc2eab-65b3-415f-873e-8af7888486a9','item-32',6,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(7,'867e415a-e817-4b25-ab64-9fee0d3c6424','item-40',7,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(8,'867e415a-e817-4b25-ab64-9fee0d3c6424','item-41',8,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(9,'867e415a-e817-4b25-ab64-9fee0d3c6424','item-42',9,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(10,'f0ae5ee4-d6d5-4c33-9107-192d230349f2','item-50',10,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(11,'f0ae5ee4-d6d5-4c33-9107-192d230349f2','item-51',11,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(12,'f0ae5ee4-d6d5-4c33-9107-192d230349f2','item-52',12,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(13,'c27333a1-f684-4b33-adef-dcfcdede44b2','item-60',13,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(14,'c27333a1-f684-4b33-adef-dcfcdede44b2','item-61',14,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(15,'c27333a1-f684-4b33-adef-dcfcdede44b2','item-62',15,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(16,'f58ff2cc-5c67-4bc6-ab3a-a700fb502a57','item-70',16,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(17,'f58ff2cc-5c67-4bc6-ab3a-a700fb502a57','item-71',17,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(18,'f58ff2cc-5c67-4bc6-ab3a-a700fb502a57','item-72',18,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(19,'b76a26cb-0b26-4529-a43c-44ef2bc01c0f','item-80',19,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(20,'b76a26cb-0b26-4529-a43c-44ef2bc01c0f','item-81',20,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(21,'b76a26cb-0b26-4529-a43c-44ef2bc01c0f','item-82',21,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(22,'8dcc865a-2c43-4162-bfdf-bf50999b4ce2','item-90',22,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(23,'8dcc865a-2c43-4162-bfdf-bf50999b4ce2','item-91',23,'2026-10-25 04:03:38');
INSERT INTO "embeddings" VALUES(24,'8dcc865a-2c43-4162-bfdf-bf50999b4ce2','item-92',24,'2026-10-25 04:03:38');
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
INSERT INTO "embeddings_queue" VALUES(1,'2026-10-25 04:03:38',0,'persistent://default/default/3c5add09-d3dc-4924-92b9-e0814d7277ac','item-20',X'000000BF000000BE0000803E','FLOAT32','{"chroma:document":"space text 20"}');
INSERT INTO "embeddings_queue" VALUES(2,'2026-10-25 04:03:38',0,'persistent://default/default/3c5add09-d3dc-4924-92b9-e0814d7277ac','item-21',X'0000C03E0000403F000080BF','FLOAT32','{"chroma:document":"space text 21"}');
INSERT INTO "embeddings_queue" VALUES(3,'2026-10-25 04:03:38',0,'persistent://default/default/3c5add09-d3dc-4924-92b9-e0814d7277ac','item-22',X'000060BF000000BF000000BE','FLOAT32','{"chroma:document":"space text 22"}');
INSERT INTO "embeddings_queue" VALUES(4,'2026-10-25 04:03:38',0,'persistent://default/default/2a83550c-4d42-41c7-9217-6754a3e2cfd6','item-30',X'000080BE0000003E0000003F','FLOAT32','{"chroma:document":"space text 30"}');
INSERT INTO "embeddings_queue" VALUES(5,'2026-10-25 04:03:38',0,'persistent://default/default/2a83550c-4d42-41c7-9217-6754a3e2cfd6','item-31',X'0000203F0000803F000040BF','FLOAT32','{"chroma:document":"space text 31"}');
INSERT INTO "embeddings_queue" VALUES(6,'2026-10-25 04:03:38',0,'persistent://default/default/2a83550c-4d42-41c7-9217-6754a3e2cfd6','item-32',X'000020BF000080BE0000003E','FLOAT32','{"chroma:document":"space text 32"}');
INSERT INTO "embeddings_queue" VALUES(7,'2026-10-25 04:03:38',0,'persistent://default/default/52026c96-583b-4d00-90ba-2ec96259a3a5','item-40',X'000000000000C03E0000403F','FLOAT32','{"chroma:document":"space text 40"}');
INSERT INTO "embeddings_queue" VALUES(8,'2026-10-25 04:03:38',0,'persistent://default/default/52026c96-583b-4d00-90ba-2ec96259a3a5','item-41',X'0000603F000060BF000000BF','FLOAT32','{"chroma:document":"space text 41"}');
INSERT INTO "embeddings_queue" VALUES(9,'2026-10-25 04:03:38',0,'persistent://default/default/52026c96-583b-4d00-90ba-2ec96259a3a5','item-42',X'0000C0BE000000000000C03E','FLOAT32','{"chroma:document":"space text 42"}');
INSERT INTO "embeddings_queue" VALUES(10,'2026-10-25 04:03:38',0,'persistent://default/default/3f9da12a-8eee-46da-a02e-9877508a32e5','item-50',X'0000803E0000203F0000803F','FLOAT32','{"chroma:document":"space text 50"}');
INSERT INTO "embeddings_queue" VALUES(11,'2026-10-25 04:03:38',0,'persistent://default/default/3f9da12a-8eee-46da-a02e-9877508a32e5','item-51',X'000080BF000020BF000080BE','FLOAT32','{"chroma:document":"space text 51"}');
INSERT INTO "embeddings_queue" VALUES(12,'2026-10-25 04:03:38',0,'persistent://default/default/3f9da12a-8eee-46da-a02e-9877508a32e5','item-52',X'000000BE0000803E0000203F','FLOAT32','{"chroma:document":"space text 52"}');
INSERT INTO "embeddings_queue" VALUES(13,'2026-10-25 04:03:38',0,'persistent://default/default/226a1045-0409-4b16-b4b5-6ad2d5b8f136','item-60',X'0000003F0000603F000060BF','FLOAT32','{"chroma:document":"space text 60"}');
INSERT INTO "embeddings_queue" VALUES(14,'2026-10-25 04:03:38',0,'persistent://default/default/226a1045-0409-4b16-b4b5-6ad2d5b8f136','item-61',X'000040BF0000C0BE00000000','FLOAT32','{"chroma:document":"space text 61"}');
INSERT INTO "embeddings_queue" VALUES(15,'2026-10-25 04:03:38',0,'persistent://default/default/226a1045-0409-4b16-b4b5-6ad2d5b8f136','item-62',X'0000003E0000003F0000603F','FLOAT32','{"chroma:document":"space text 62"}');
INSERT INTO "embeddings_queue" VALUES(16,'2026-10-25 04:03:38',0,'persistent://default/default/57ceb827-d818-40c6-a579-974efe86b376','item-70',X'0000403F000080BF000020BF','FLOAT32','{"chroma:document":"space text 70"}');
INSERT INTO "embeddings_queue" VALUES(17,'2026-10-25 04:03:38',0,'persistent://default/default/57ceb827-d818-40c6-a579-974efe86b376','item-71',X'000000BF000000BE0000803E','FLOAT32','{"chroma:document":"space text 71"}');
INSERT INTO "embeddings_queue" VALUES(18,'2026-10-25 04:03:38',0,'persistent://default/default/57ceb827-d818-40c6-a579-974efe86b376','item-72',X'0000C03E0000403F000080BF','FLOAT32','{"chroma:document":"space text 72"}');
INSERT INTO "embeddings_queue" VALUES(19,'2026-10-25 04:03:38',0,'persistent://default/default/d1d78f82-28a8-4ca9-a9e3-676e5c05414a','item-80',X'0000803F000040BF0000C0BE','FLOAT32','{"chroma:document":"space text 80"}');
INSERT INTO "embeddings_queue" VALUES(20,'2026-10-25 04:03:38',0,'persistent://default/default/d1d78f82-28a8-4ca9-a9e3-676e5c05414a','item-81',X'000080BE0000003E0000003F','FLOAT32','{"chroma:document":"space text 81"}');
INSERT INTO "embeddings_queue" VALUES(21,'2026-10-25 04:03:38',0,'persistent://default/default/d1d78f82-28a8-4ca9-a9e3-676e5c05414a','item-82',X'0000203F0000803F000040BF','FLOAT32','{"chroma:document":"space text 82"}');
INSERT INTO "embeddings_queue" VALUES(22,'2026-10-25 04:03:38',0,'persistent://default/default/7bbd7d1a-ed7e-4b63-9e94-51d33d937a05','item-90',X'000060BF000000BF000000BE','FLOAT32','{"chroma:document":"space text 90"}');
INSERT INTO "embeddings_queue" VALUES(23,'2026-10-25 04:03:38',0,'persistent://default/default/7bbd7d1a-ed7e-4b63-9e94-51d33d937a05','item-91',X'000000000000C03E0000403F','FLOAT32','{"chroma:document":"space text 91"}');
INSERT INTO "embeddings_queue" VALUES(24,'2026-10-25 04:03:38',0,'persistent://default/default/7bbd7d1a-ed7e-4b63-9e94-51d33d937a05','item-92',X'0000603F000060BF000000BF','FLOAT32','{"chroma:document":"space text 92"}');
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
INSERT INTO "max_seq_id" VALUES('692d97f2-a016-4444-8019-ba4e26ff7e73',3);
INSERT INTO "max_seq_id" VALUES('b4cc2eab-65b3-415f-873e-8af7888486a9',6);
INSERT INTO "max_seq_id" VALUES('867e415a-e817-4b25-ab64-9fee0d3c6424',9);
INSERT INTO "max_seq_id" VALUES('f0ae5ee4-d6d5-4c33-9107-192d230349f2',12);
INSERT INTO "max_seq_id" VALUES('c27333a1-f684-4b33-adef-dcfcdede44b2',15);
INSERT INTO "max_seq_id" VALUES('f58ff2cc-5c67-4bc6-ab3a-a700fb502a57',18);
INSERT INTO "max_seq_id" VALUES('b76a26cb-0b26-4529-a43c-44ef2bc01c0f',21);
INSERT INTO "max_seq_id" VALUES('8dcc865a-2c43-4162-bfdf-bf50999b4ce2',24);
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
CREATE TABLE "segments" (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    collection TEXT REFERENCES collection(id) NOT NULL
);
INSERT INTO "segments" VALUES('631c9818-788c-4c7b-b062-5a1065afe5cd','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','3c5add09-d3dc-4924-92b9-e0814d7277ac');
INSERT INTO "segments" VALUES('692d97f2-a016-4444-8019-ba4e26ff7e73','urn:chroma:segment/metadata/sqlite','METADATA','3c5add09-d3dc-4924-92b9-e0814d7277ac');
INSERT INTO "segments" VALUES('3a0b6d85-74b8-454f-8bab-d1f2000d855c','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','2a83550c-4d42-41c7-9217-6754a3e2cfd6');
INSERT INTO "segments" VALUES('b4cc2eab-65b3-415f-873e-8af7888486a9','urn:chroma:segment/metadata/sqlite','METADATA','2a83550c-4d42-41c7-9217-6754a3e2cfd6');
INSERT INTO "segments" VALUES('faa851e7-b0c4-4902-a496-12f57c09be1d','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','52026c96-583b-4d00-90ba-2ec96259a3a5');
INSERT INTO "segments" VALUES('867e415a-e817-4b25-ab64-9fee0d3c6424','urn:chroma:segment/metadata/sqlite','METADATA','52026c96-583b-4d00-90ba-2ec96259a3a5');
INSERT INTO "segments" VALUES('e3bcf357-4357-4734-b141-ccae04ecfa08','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','3f9da12a-8eee-46da-a02e-9877508a32e5');
INSERT INTO "segments" VALUES('f0ae5ee4-d6d5-4c33-9107-192d230349f2','urn:chroma:segment/metadata/sqlite','METADATA','3f9da12a-8eee-46da-a02e-9877508a32e5');
INSERT INTO "segments" VALUES('28e47a19-afbe-4d19-905c-da4363d6df48','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','226a1045-0409-4b16-b4b5-6ad2d5b8f136');
INSERT INTO "segments" VALUES('c27333a1-f684-4b33-adef-dcfcdede44b2','urn:chroma:segment/metadata/sqlite','METADATA','226a1045-0409-4b16-b4b5-6ad2d5b8f136');
INSERT INTO "segments" VALUES('dca07998-20a5-4291-8f3d-5fe56ea9f9ff','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','57ceb827-d818-40c6-a579-974efe86b376');
INSERT INTO "segments" VALUES('f58ff2cc-5c67-4bc6-ab3a-a700fb502a57','urn:chroma:segment/metadata/sqlite','METADATA','57ceb827-d818-40c6-a579-974efe86b376');
INSERT INTO "segments" VALUES('37983290-7f32-4c36-a441-c406d222b7db','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','d1d78f82-28a8-4ca9-a9e3-676e5c05414a');
INSERT INTO "segments" VALUES('b76a26cb-0b26-4529-a43c-44ef2bc01c0f','urn:chroma:segment/metadata/sqlite','METADATA','d1d78f82-28a8-4ca9-a9e3-676e5c05414a');
INSERT INTO "segments" VALUES('70729d11-acf1-41c2-9541-cab557560a70','urn:chroma:segment/vector/hnsw-local-persisted','VECTOR','7bbd7d1a-ed7e-4b63-9e94-51d33d937a05');
INSERT INTO "segments" VALUES('8dcc865a-2c43-4162-bfdf-bf50999b4ce2','urn:chroma:segment/metadata/sqlite','METADATA','7bbd7d1a-ed7e-4b63-9e94-51d33d937a05');
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
