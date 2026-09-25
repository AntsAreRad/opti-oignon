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

pub mod fx;
pub mod journal;
pub mod laws;
pub mod ocj;
pub mod organs;
pub mod protocol;
pub mod rng;

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

    #[test]
    fn a_float_is_refused_by_name() {
        let answer = call(br#"{"doc":1.5,"op":"echo","v":1}"#);
        assert_eq!(answer, br#"{"detail":"number at 7","refused":"float"}"#.to_vec());
    }
}
