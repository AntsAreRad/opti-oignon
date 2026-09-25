//! The componion's deterministic engine, the twin of `opti_oignon/allium`.
//!
//! The being is a pure function of its facts. This crate never reads a clock,
//! never touches a file or the network, and holds no float: time and facts
//! are arguments, every quantity is an integer in fixed point, and every
//! random draw is addressed by content. Its one entry point is a byte
//! protocol, `call(&[u8]) -> Vec<u8>`, answered byte for byte as the Python
//! reference answers it; the native core wraps it for Python, and the same
//! bytes will serve the phone.
#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(clippy::arithmetic_side_effects)]

extern crate alloc;

pub mod civil;
pub mod fx;
pub mod journal;
pub mod lawdata;
pub mod laws;
pub mod ocj;
pub mod organs;
pub mod protocol;
pub mod rng;
pub mod world;

pub use protocol::{call, engine_info, ENGINE_VERSION};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stepped_decay_and_a_table_decay_differ_as_the_law_file_says() {
        let mut work = fx::Work::default();
        assert_eq!(fx::decay_iter(7, 2, 1, &mut work), 6);
        assert_eq!(fx::decay_lazy(7, 2, 1, 256, &mut work), 5);
        assert_eq!(work.alarm, 0);
    }

    #[test]
    fn a_canonical_document_is_its_own_re_emission() {
        let doc = br#"{"a":[1,-2,true,false,null,"x\"y\\z"],"b":{}}"#;
        let value = ocj::parse(doc, false).expect("canonical");
        assert_eq!(ocj::emit(&value).expect("emits"), doc.to_vec());
    }

    #[test]
    fn an_object_built_out_of_key_order_is_refused_on_emission_and_obj_sorts() {
        use alloc::string::String;
        use alloc::vec;
        let member = |key: &str| (String::from(key), ocj::Value::Int(1));
        let unsorted = ocj::Value::Obj(vec![member("b"), member("a")]);
        let refusal = ocj::emit(&unsorted).expect_err("keys out of order");
        assert_eq!((refusal.code, refusal.detail.as_str()), ("engine_panic", "object order"));
        let repeated = ocj::Value::Obj(vec![member("a"), member("a")]);
        assert!(ocj::emit(&repeated).is_err(), "a repeated key is not ascending");
        let sorted = ocj::obj(vec![member("b"), member("a")]);
        assert_eq!(ocj::emit(&sorted).expect("sorted"), br#"{"a":1,"b":1}"#.to_vec());
    }

    #[test]
    fn every_listed_op_has_its_own_arm_and_an_unlisted_one_is_unknown() {
        let answer = call(br#"{"op":"genome_mutate","v":1}"#);
        assert_eq!(answer, br#"{"detail":"op","refused":"unknown_op"}"#.to_vec());
        let answer = call(br#"{"domain":"d","index":0,"kind":"unit","n":1,"op":"rng","seed":"0000000000000000000000000000000000000000000000000000000000000000","v":1}"#);
        assert!(answer.starts_with(br#"{"out":["#), "rng answers through its own arm");
    }

    /// The fixture law as a sound life, parsed from the embedded file.
    fn fixture() -> (ocj::Value, alloc::string::String) {
        lawdata::law_file("fixture").expect("the fixture law is carried")
    }

    /// The fixture law with the member at `path` replaced (or removed when `value` is `None`).
    fn edited(law: &ocj::Value, path: &[&str], value: Option<ocj::Value>) -> ocj::Value {
        use alloc::string::String;
        let ocj::Value::Obj(members) = law else { return law.clone() };
        let (head, rest) = path.split_first().expect("a path");
        let mut out = alloc::vec::Vec::new();
        for (key, item) in members {
            if key != head {
                out.push((key.clone(), item.clone()));
            } else if rest.is_empty() {
                if let Some(value) = value.clone() {
                    out.push((String::from(*head), value));
                }
            } else {
                out.push((key.clone(), edited(item, rest, value.clone())));
            }
        }
        ocj::Value::Obj(out)
    }

    #[test]
    fn both_carried_laws_are_sound_lives_with_the_organs_the_code_names() {
        for name in ["fixture", "v0_1"] {
            let life = lawdata::life(name).expect("a carried law is a sound life");
            let organs: alloc::vec::Vec<&str> = life.organs.iter().map(|organ| organ.name()).collect();
            assert_eq!(organs, ["chem", "clock", "soil", "stage"]);
            assert!(life.is_quiescent(lawdata::Organ::Chem) && life.is_quiescent(lawdata::Organ::Clock));
            assert!(!life.is_quiescent(lawdata::Organ::Soil) && !life.is_quiescent(lawdata::Organ::Stage));
            assert_eq!(life.limits.get("tz").copied(), life.limits.get("clock").copied());
        }
    }

    #[test]
    fn a_defective_life_section_is_refused_by_the_name_the_reference_gives() {
        let (law, digest) = fixture();
        let detail = |law: ocj::Value| match lawdata::life_of("fixture", digest.clone(), law) {
            Ok(_) => alloc::string::String::from("ok"),
            Err(refusal) => refusal.detail,
        };
        assert_eq!(detail(law.clone()), "ok");
        assert_eq!(detail(edited(&law, &["code"], Some(ocj::s("seed_2")))), "code");
        assert_eq!(detail(edited(&law, &["world"], None)), "life law");
        assert_eq!(detail(edited(&law, &["world", "daylength", "ramp"], Some(ocj::Value::Int(121)))), "life law");
        assert_eq!(detail(edited(&law, &["journal", "budgets", "act"], None)), "journal");
        assert_eq!(detail(edited(&law, &["constants", "stage", "theta_wet"], Some(ocj::Value::Int(65537)))), "life law");
    }

    #[test]
    fn a_fresh_life_answers_and_a_change_of_law_is_refused_as_a_migration() {
        use alloc::format;
        let (_, digest) = fixture();
        let (_, other) = lawdata::law_file("v0_1").expect("the full law is carried");
        let being = "00112233445566778899aabbccddeeff";
        let genesis = format!(
            concat!(
                r#"{{"being":"{b}","body":{{"band":"long","birth":{{"tz":0,"wall":1760000000}},"derive":1,"#,
                r#""hemisphere":"north","laws":{{"name":"fixture","params":{{"evap_awake":4096,"evap_dormant":1024,"#,
                r#""rain_gain":65536,"sun_max":65536}},"provisional":true,"sha256":"{d}","v":0}},"owner":"{b}","#,
                r#""rhythm_consent":false,"seed":"{s}","soil":"encrypted","weather":"garden"}},"kind":"genesis","#,
                r#""laws":0,"origin":"0011223344556677","oseq":0,"t":0}}"#
            ),
            b = being,
            d = digest,
            s = "ab".repeat(32)
        );
        let request = |facts: &str| {
            format!(r#"{{"budget":9007199254740991,"facts":[{}],"genesis":{},"op":"advance","state":null,"to":3000,"v":1}}"#, facts, genesis)
        };
        let answer = call(request("").as_bytes());
        assert!(answer.starts_with(br#"{"alarm":0,"at":3000,"done":true,"env":{"#), "{:?}", core::str::from_utf8(&answer));
        let evolve = format!(
            concat!(
                r#"{{"being":"{b}","body":{{"effective_from":2440,"from":{{"name":"fixture","sha256":"{d}"}},"#,
                r#""params":{{"evap_awake":4096,"evap_dormant":1024,"rain_gain":65536,"sun_max":65536}},"#,
                r#""to":{{"name":"v0_1","sha256":"{o}","v":0}}}},"kind":"evolve","laws":0,"origin":"0011223344556677","oseq":1,"t":2000}}"#
            ),
            b = being,
            d = digest,
            o = other
        );
        assert_eq!(call(request(&evolve).as_bytes()), br#"{"detail":"migration","refused":"unknown_law"}"#.to_vec());
    }

    #[test]
    fn a_float_is_refused_by_name() {
        let answer = call(br#"{"doc":1.5,"op":"echo","v":1}"#);
        assert_eq!(answer, br#"{"detail":"number at 7","refused":"float"}"#.to_vec());
    }
}
