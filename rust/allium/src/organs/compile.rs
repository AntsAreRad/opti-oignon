//! Compilation: a genome to flat tables, derived and never stored; the twin
//! of `opti_oignon/allium/ref/organs/compile.py`.
//!
//! Steps, in order:
//!
//! 1. decode (any refusal propagates);
//! 2. promoters: for each non-CIS record, the run of CIS records just before it;
//! 3. one row per non-CIS law locus present on its pair, in ascending locus
//!    id: dose 1 keeps the allele as it is; dose 2 combines by the locus's
//!    mode -- DOM_MAX keeps the allele with the larger (strength, body bytes)
//!    key, ADD floors the mean of every field that is not structural,
//!    LOAD_REC emits a load row only when both alleles carry the load;
//! 4. edges: for each gene, the union of the CIS loci in its promoters on the
//!    homologs where it is present, in ascending CIS locus id, each combined
//!    by its own mode as in step 3;
//! 5. species decay: 0 for input and conserved species, else the largest
//!    compiled `deg` of the TF rows writing it, else the law's default;
//! 6. unit conversion: Q4.12 fields times 16, Q3.13 fields and angles
//!    times 8; the reserved kinds pass through raw.
//!
//! Every intermediate is an i64 and every operation on one is checked; a
//! failed check refuses `engine_panic "genome compile"`. The output columns
//! are chassis bulk strings, whose range check is the single guard on output
//! values, and every object is built with `ocj::obj`.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::genome::{self as g, Record, View, BODY};
use crate::ocj::{self, obj, Refused, Value};

pub const TABLES_SCHEMA: i64 = 1;

/// The kinds whose rows pass through raw, in the order of the `reserved` table.
pub const RESERVED: [&str; 7] = ["brain", "lex", "plast", "prc", "shape", "temp", "vern"];

type Columns = &'static [(&'static str, &'static str)];

/// Each gene table and its columns, with their bulk types. A column is the
/// kind's field of that name (`km` reads `Km`), or the gene row for `gene`.
const TABLES: [(&str, Columns); 8] = [
    ("tf", &[("bias", "i32"), ("deg", "i32"), ("gene", "u16"), ("out", "u8"), ("prod", "i32"), ("rate", "i32")]),
    (
        "enz",
        &[
            ("gene", "u16"),
            ("kcat", "i32"),
            ("km", "i32"),
            ("p1", "u8"),
            ("p2", "u8"),
            ("s1", "u8"),
            ("s2", "u8"),
            ("yield", "i32"),
        ],
    ),
    ("rec", &[("channel", "u8"), ("gain", "i32"), ("gene", "u16"), ("species", "u8"), ("threshold", "i32")]),
    (
        "morph",
        &[
            ("angle", "i32"),
            ("gene", "u16"),
            ("guard", "u8"),
            ("pred", "u8"),
            ("rate", "i32"),
            ("succ", "u8"),
            ("threshold", "i32"),
        ],
    ),
    ("imm", &[("class", "u8"), ("cost", "i32"), ("gene", "u16"), ("strength", "i32")]),
    ("pig", &[("allele", "u8"), ("class", "u8"), ("gene", "u16")]),
    ("te", &[("activity", "i32"), ("gene", "u16"), ("target_bias", "u8")]),
    ("loads", &[("effect", "i32"), ("gene", "u16"), ("target", "u8")]),
];

const EDGE_COLUMNS: [(&str, &str); 6] = [("cis", "u16"), ("k", "i32"), ("mode", "u8"), ("n", "u8"), ("src", "u8"), ("w", "i32")];

const GENE_COLUMNS: [(&str, &str); 5] = [("dose", "u8"), ("edge_start", "u32"), ("kind", "u8"), ("locus", "u16"), ("stages", "u16")];

fn compile_panic() -> Refused {
    g::engine_panic("genome compile")
}

/// One allele as compile reads it: its decoded values and its raw body.
type Allele<'a> = (&'a [i64], &'a [u8; BODY]);

/// One compiled gene row.
struct Gene {
    ident: i64,
    kind: u8,
    stages: i64,
    dose: usize,
    values: Vec<i64>,
    edges: Vec<(i64, Vec<i64>)>,
    loaded: Option<Vec<i64>>,
}

/// One locus's values from its one or two alleles.
fn combine(kind_code: u8, mode: i64, alleles: &[Allele<'_>]) -> Result<Vec<i64>, Refused> {
    match alleles {
        [(values, _)] => Ok(values.to_vec()),
        [(va, ba), (vb, bb)] => {
            if mode == g::DOM_MAX {
                let key_a = (g::strength(kind_code, va, ba).ok_or_else(compile_panic)?, **ba);
                let key_b = (g::strength(kind_code, vb, bb).ok_or_else(compile_panic)?, **bb);
                return Ok(if key_a >= key_b { va.to_vec() } else { vb.to_vec() });
            }
            let kind = g::kind_by_code(kind_code).ok_or_else(compile_panic)?;
            let mut out = Vec::with_capacity(kind.fields.len());
            for ((field, a), b) in kind.fields.iter().zip(va.iter()).zip(vb.iter()) {
                if field.marks & g::S != 0 {
                    if a != b {
                        return Err(compile_panic());
                    }
                    out.push(*a);
                } else {
                    let mean = a.checked_add(*b).and_then(|sum| sum.checked_div_euclid(2)).ok_or_else(compile_panic)?;
                    out.push(mean);
                }
            }
            Ok(out)
        }
        _ => Err(compile_panic()),
    }
}

/// A row's values in Q16.16: each field times its scale; reserved kinds raw.
fn scaled(kind_code: u8, values: &[i64]) -> Result<Vec<i64>, Refused> {
    let kind = g::kind_by_code(kind_code).ok_or_else(compile_panic)?;
    let mut out = Vec::with_capacity(values.len());
    for (field, value) in kind.fields.iter().zip(values.iter()) {
        if kind.reserved {
            out.push(*value);
        } else {
            out.push(value.checked_mul(field.scale).ok_or_else(compile_panic)?);
        }
    }
    Ok(out)
}

/// The value of the field named `name` in a row aligned with its kind's fields.
fn field_value(kind_code: u8, row: &[i64], name: &str) -> Result<i64, Refused> {
    let kind = g::kind_by_code(kind_code).ok_or_else(compile_panic)?;
    let index = kind.fields.iter().position(|field| field.name == name).ok_or_else(compile_panic)?;
    row.get(index).copied().ok_or_else(compile_panic)
}

fn row_index(n: usize) -> Result<i64, Refused> {
    i64::try_from(n).map_err(|_| compile_panic())
}

fn push(columns: &mut BTreeMap<&'static str, Vec<i64>>, name: &str, value: i64) -> Result<(), Refused> {
    columns.get_mut(name).ok_or_else(compile_panic)?.push(value);
    Ok(())
}

fn column(kind: &str, values: &[i64]) -> Result<Value, Refused> {
    let items: Vec<Value> = values.iter().map(|value| Value::Int(*value)).collect();
    Ok(Value::Str(ocj::pack_bulk(kind, &items)?))
}

fn packed(spec: &[(&'static str, &'static str)], columns: &BTreeMap<&'static str, Vec<i64>>) -> Result<Value, Refused> {
    let mut members = Vec::with_capacity(spec.len());
    for (name, kind) in spec {
        let values = columns.get(name).ok_or_else(compile_panic)?;
        members.push((String::from(*name), column(kind, values)?));
    }
    Ok(obj(members))
}

fn empty(spec: &[(&'static str, &'static str)]) -> BTreeMap<&'static str, Vec<i64>> {
    spec.iter().map(|(name, _)| (*name, Vec::new())).collect()
}

/// Where each locus of one chromosome sits, and the promoter before each record.
type Placement = (BTreeMap<i64, usize>, Vec<Vec<usize>>);

/// `(tables, work)` for canonical genome bytes under a validated law.
pub fn compile_genome(data: &[u8], lawview: &View, law_digest: &str) -> Result<(Value, u64), Refused> {
    let chromosomes: Vec<Vec<Record>> = g::decode(data, lawview)?;
    // Where each locus sits, per chromosome, and the promoter before each record.
    let mut placed: Vec<Placement> = Vec::with_capacity(chromosomes.len());
    for chrom in &chromosomes {
        let mut at: BTreeMap<i64, usize> = BTreeMap::new();
        let mut promoter: Vec<Vec<usize>> = Vec::with_capacity(chrom.len());
        let mut run: Vec<usize> = Vec::new();
        for (index, record) in chrom.iter().enumerate() {
            if record.kind == g::KIND_CIS {
                run.push(index);
                promoter.push(Vec::new());
            } else {
                promoter.push(core::mem::take(&mut run));
            }
            at.insert(record.locus, index);
        }
        placed.push((at, promoter));
    }
    let mut genes: Vec<Gene> = Vec::new();
    let mut edges_total: u64 = 0;
    for ident in &lawview.order {
        let locus = lawview.loci.get(ident).ok_or_else(compile_panic)?;
        if locus.kind == g::KIND_CIS {
            continue;
        }
        let home = usize::try_from(*ident >> 8).ok().and_then(|pair| pair.checked_mul(g::PLOIDY)).ok_or_else(compile_panic)?;
        let mut alleles: Vec<Allele<'_>> = Vec::with_capacity(g::PLOIDY);
        let mut occurrences: BTreeMap<i64, Vec<Allele<'_>>> = BTreeMap::new();
        for h in 0..g::PLOIDY {
            let c = home.checked_add(h).ok_or_else(compile_panic)?;
            let (at, promoter) = placed.get(c).ok_or_else(compile_panic)?;
            let Some(index) = at.get(ident) else {
                continue;
            };
            let chrom = chromosomes.get(c).ok_or_else(compile_panic)?;
            let record = chrom.get(*index).ok_or_else(compile_panic)?;
            alleles.push((record.fields.as_slice(), &record.body));
            for cis_index in promoter.get(*index).ok_or_else(compile_panic)? {
                let cis_record = chrom.get(*cis_index).ok_or_else(compile_panic)?;
                occurrences.entry(cis_record.locus).or_default().push((cis_record.fields.as_slice(), &cis_record.body));
            }
        }
        if alleles.is_empty() {
            continue;
        }
        let values = combine(locus.kind, locus.mode(), &alleles)?;
        let both_loaded = alleles.len() == 2 && alleles.iter().all(|(fields, _)| fields.get(1).is_some_and(|effect| *effect > 0));
        let loaded = if locus.kind == g::KIND_LOAD && both_loaded { Some(values.clone()) } else { None };
        let mut edges: Vec<(i64, Vec<i64>)> = Vec::with_capacity(occurrences.len());
        for (cis_id, found) in &occurrences {
            let cis_locus = lawview.loci.get(cis_id).ok_or_else(compile_panic)?;
            edges.push((*cis_id, combine(g::KIND_CIS, cis_locus.mode(), found)?));
        }
        let count = u64::try_from(edges.len()).map_err(|_| compile_panic())?;
        edges_total = edges_total.checked_add(count).ok_or_else(compile_panic)?;
        genes.push(Gene { ident: *ident, kind: locus.kind, stages: locus.stages, dose: alleles.len(), values, edges, loaded });
    }
    let tables = emit_tables(&genes, lawview, data, law_digest)?;
    let mut records: u64 = 0;
    for chrom in &chromosomes {
        let count = u64::try_from(chrom.len()).map_err(|_| compile_panic())?;
        records = records.checked_add(count).ok_or_else(compile_panic)?;
    }
    let work = g::blocks(data.len())
        .and_then(|blocks| blocks.checked_add(records))
        .and_then(|sum| sum.checked_add(edges_total))
        .ok_or_else(compile_panic)?;
    Ok((tables, work))
}

fn emit_tables(genes: &[Gene], lawview: &View, data: &[u8], law_digest: &str) -> Result<Value, Refused> {
    let mut columns: BTreeMap<&'static str, BTreeMap<&'static str, Vec<i64>>> =
        TABLES.iter().map(|(table, spec)| (*table, empty(spec))).collect();
    let mut reserved: BTreeMap<&'static str, (Vec<i64>, Vec<i64>)> =
        RESERVED.iter().map(|name| (*name, (Vec::new(), Vec::new()))).collect();
    let mut gene_cols = empty(&GENE_COLUMNS);
    push(&mut gene_cols, "edge_start", 0)?;
    let mut edge_cols = empty(&EDGE_COLUMNS);
    let mut tf_deg: BTreeMap<i64, i64> = BTreeMap::new();
    for (row, gene) in genes.iter().enumerate() {
        let row = row_index(row)?;
        let kind = g::kind_by_code(gene.kind).ok_or_else(compile_panic)?;
        push(&mut gene_cols, "dose", row_index(gene.dose)?)?;
        push(&mut gene_cols, "kind", i64::from(gene.kind))?;
        push(&mut gene_cols, "locus", gene.ident)?;
        push(&mut gene_cols, "stages", gene.stages)?;
        for (cis_id, edge) in &gene.edges {
            let edge = scaled(g::KIND_CIS, edge)?;
            push(&mut edge_cols, "cis", *cis_id)?;
            push(&mut edge_cols, "k", field_value(g::KIND_CIS, &edge, "K")?)?;
            push(&mut edge_cols, "mode", field_value(g::KIND_CIS, &edge, "mode")?)?;
            push(&mut edge_cols, "n", field_value(g::KIND_CIS, &edge, "n")?)?;
            push(&mut edge_cols, "src", field_value(g::KIND_CIS, &edge, "src")?)?;
            push(&mut edge_cols, "w", field_value(g::KIND_CIS, &edge, "w")?)?;
        }
        let emitted = edge_cols.get("cis").map(Vec::len).ok_or_else(compile_panic)?;
        push(&mut gene_cols, "edge_start", row_index(emitted)?)?;
        if kind.reserved {
            let (fields, rows) = reserved.get_mut(kind.name).ok_or_else(compile_panic)?;
            rows.push(row);
            fields.extend_from_slice(&gene.values);
            continue;
        }
        if kind.name == "load" {
            if let Some(loaded) = &gene.loaded {
                let loads = columns.get_mut("loads").ok_or_else(compile_panic)?;
                push(loads, "effect", field_value(gene.kind, loaded, "effect")?)?;
                push(loads, "gene", row)?;
                push(loads, "target", field_value(gene.kind, loaded, "target")?)?;
            }
            continue;
        }
        let (table, spec) = TABLES.iter().find(|(table, _)| *table == kind.name).ok_or_else(compile_panic)?;
        let values = scaled(gene.kind, &gene.values)?;
        let target = columns.get_mut(table).ok_or_else(compile_panic)?;
        for (name, _) in spec.iter() {
            if *name == "gene" {
                push(target, "gene", row)?;
            } else {
                let field = if *name == "km" { "Km" } else { *name };
                push(target, name, field_value(gene.kind, &values, field)?)?;
            }
        }
        if kind.name == "tf" {
            let out = field_value(gene.kind, &values, "out")?;
            let deg = field_value(gene.kind, &values, "deg")?;
            let slot = tf_deg.entry(out).or_insert(0);
            *slot = (*slot).max(deg);
        }
    }
    let mut species_deg: Vec<i64> = Vec::with_capacity(g::SPECIES);
    for code in 0..g::SPECIES {
        let class = *lawview.classes.get(code).ok_or_else(compile_panic)?;
        let key = row_index(code)?;
        if class == "input" || class == "conserved" {
            species_deg.push(0);
        } else if let Some(deg) = tf_deg.get(&key) {
            species_deg.push(*deg);
        } else {
            species_deg.push(*lawview.deg.get(code).ok_or_else(compile_panic)?);
        }
    }
    let mut reserved_members = Vec::with_capacity(RESERVED.len());
    for name in RESERVED {
        let (fields, rows) = reserved.get(name).ok_or_else(compile_panic)?;
        reserved_members.push((
            String::from(name),
            obj(vec![(String::from("fields"), column("i32", fields)?), (String::from("gene"), column("u16", rows)?)]),
        ));
    }
    let mut members = vec![
        (String::from("edges"), packed(&EDGE_COLUMNS, &edge_cols)?),
        (String::from("genes"), packed(&GENE_COLUMNS, &gene_cols)?),
        (String::from("genome"), Value::Str(g::sha256_hex(data))),
        (String::from("law"), Value::Str(String::from(law_digest))),
        (String::from("reserved"), obj(reserved_members)),
        (String::from("schema"), Value::Int(TABLES_SCHEMA)),
        (String::from("species"), obj(vec![(String::from("deg"), column("i32", &species_deg)?)])),
    ];
    for (table, spec) in TABLES.iter() {
        let values = columns.get(table).ok_or_else(compile_panic)?;
        members.push((String::from(*table), packed(spec, values)?));
    }
    Ok(obj(members))
}
