-- The second-factor store as block 176 wrote it, for TF14 (tests/test_two_factor_account_contracts.py).
-- Made on the 176 tree (labo177's base, 2026-10-23) by its own code: the rows under the literal 'admin' through
-- the 176 routes as the settings panels called them (TOTP setup and verify, recovery codes, one app password),
-- the rows of account 5f0c2a8e-... through the 176 manager, one security key each by the 176 manager's own INSERT.
-- No secret: the seed is a stand-in encryption of a fixed test seed, codes and app passwords are keyed hashes.
BEGIN TRANSACTION;
CREATE TABLE app_passwords (
                password_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_used REAL DEFAULT 0,
                revoked INTEGER DEFAULT 0
            );
INSERT INTO "app_passwords" VALUES('8A8s30eVt3c','admin','cli','v2:6349ef9492cc96e1f4d0df85b5decff26fe62326329b16f8e83cda19046446ce',1700000000.0,0.0,0);
INSERT INTO "app_passwords" VALUES('6IkZ94-lS3M','5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','keeper-cli','v2:e5db259b02f67a3e94a4a1460da31b72b53e40a6f636af49fc8f51791f2571c2',1700000000.0,0.0,0);
CREATE TABLE recovery_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                used_at REAL DEFAULT 0,
                created_at REAL NOT NULL
            );
INSERT INTO "recovery_codes" VALUES(1,'admin','v2:71329150b71320102bd9e4f4a8893cb7b005dfd47ae71c25e2f8db7d9e6e26d4',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(2,'admin','v2:a68d2d9443c9d0becf5ce9309334c57e29bec2ea25db98b5c73d4867fbcc45d5',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(3,'admin','v2:a8bd1a03641d055cf1813fe42ce94ee2d395bfe8c263a465fa5a883798b5be0a',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(4,'admin','v2:b68e83b0b7854b104ffd9749cf6ea39a113b4b8c008525142cfef34f10377da3',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(5,'admin','v2:de077dc4cf3f6120ab8583f23dc1e0f8de2803fde48ef414fe6797ce3edf7fd3',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(6,'admin','v2:290e099fb5c282ae35ee966f1179db9c9b28a15b9ce5ddf37ace222ce41439ac',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(7,'admin','v2:8f6e682347b562e9b03274b1ada1aaba682fb0b80fe3f011dd60d86aeba75b53',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(8,'admin','v2:0f788fea15f7768a8e644f982c4ad11e7f2472e7cfec8bb5c59aae31ef7b7e26',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(9,'admin','v2:e342a57fb12f936c1e18ae504e1fc631abe75541356bd099127632182a02be10',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(10,'admin','v2:ce02d12bcfcf63b0dba64d58233ebbb8e0fc985cd8d691177ee7483886cac91f',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(11,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:ddc7bb55d49c7dd4fc96497bdc5011a0b4c950e64c0b7ed0b7ea15c8b39c5c99',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(12,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:e6722fde26a33dcfc951f7a6cf0d2df6dee287a43edcc8675e019adbecd13fcb',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(13,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:61f20b04f5a7d5846d474edfd53d3e473a499327cb460f5e419bcb1c2a809308',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(14,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:bba8ea5c555bdbaf9995f2d854224a3808ced488267be792577c3b124a5e30c8',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(15,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:5df559a1aa24665efd69e00579b1425d3e663efa05e34ce44ab188aab16e58d5',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(16,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:41f9989f6ca1842c7831626cca716ac6cfd4635cb34aeb45995822ab59023b9a',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(17,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:875c7f94383593b1ed8a3600503c789a2eb3b70a62d676ecb11812607ceaa8ca',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(18,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:70965df2dc95fc1caea87845f0d76f5de713fce34438bb3d24bf72076dec4035',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(19,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:f566ebea0323a1c0084a513a483f7dff1da744e9b213789100a62621c43aca2a',0,0.0,1700000000.0);
INSERT INTO "recovery_codes" VALUES(20,'5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','v2:d141bd913473c82327741a2ce29d2e2cec72ddb86bb5fe5a9e737f472c01a1d4',0,0.0,1700000000.0);
CREATE TABLE totp_config (
                user_id TEXT PRIMARY KEY,
                secret_encrypted TEXT NOT NULL,
                issuer TEXT DEFAULT 'Opti-Oignon',
                algorithm TEXT DEFAULT 'SHA1',
                digits INTEGER DEFAULT 6,
                period INTEGER DEFAULT 30,
                verified INTEGER DEFAULT 0,
                last_step INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
INSERT INTO "totp_config" VALUES('admin','ENC:VEVTVFNFRURCQVNFMzJBQg==','Opti-Oignon','SHA1',6,30,1,0,1700000000.0);
INSERT INTO "totp_config" VALUES('5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','ENC:VEVTVFNFRURCQVNFMzJBQg==','Opti-Oignon','SHA1',6,30,1,0,1700000000.0);
CREATE TABLE webauthn_credentials (
                credential_id TEXT PRIMARY KEY,
                public_key TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT 'Security Key',
                created_at REAL NOT NULL,
                last_used REAL DEFAULT 0,
                sign_count INTEGER DEFAULT 0
            );
INSERT INTO "webauthn_credentials" VALUES('cred-admin-1','cHVibGljLWtleS0cred-admin-1','admin','Security Key',1.792764084979874849e+09,0.0,0);
INSERT INTO "webauthn_credentials" VALUES('cred-keeper-1','cHVibGljLWtleS0cred-keeper-1','5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90','Security Key',1.792764084979886055e+09,0.0,0);
CREATE INDEX idx_webauthn_user
                ON webauthn_credentials(user_id);
CREATE INDEX idx_recovery_user
                ON recovery_codes(user_id);
CREATE INDEX idx_app_pw_user
                ON app_passwords(user_id);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('recovery_codes',20);
COMMIT;
