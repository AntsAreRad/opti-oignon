"""Compilation: a genome to flat tables, derived and never stored.

Steps, in order:

1. decode (any refusal propagates);
2. promoters: for each non-CIS record, the run of CIS records just before it;
3. one row per non-CIS law locus present on its pair, in ascending locus id:
   dose 1 keeps the allele as it is; dose 2 combines by the locus's mode --
   DOM_MAX keeps the allele with the larger ``(strength, body bytes)`` key,
   ADD floors the mean of every field that is not structural, LOAD_REC emits
   a load row only when both alleles carry the load;
4. edges: for each gene, the union of the CIS loci in its promoters on the
   homologs where it is present, in ascending CIS locus id, each combined by
   its own mode as in step 3;
5. species decay: 0 for input and conserved species, else the largest
   compiled ``deg`` of the TF rows writing it, else the law's default;
6. unit conversion: Q4.12 fields times 16, Q3.13 fields and angles times 8;
   the reserved kinds pass through raw.

Every intermediate is an integer; the output columns are chassis bulk
strings, whose range check is the single guard on output values.
"""

from ... import wire
from . import genome as g
from .genome import F_MARKS, F_NAME, F_SCALE, K_FIELDS, K_NAME, K_RESERVED

checkpoint_before_apply = True

TABLES_SCHEMA = 1

_COLUMNS = {
    "tf": (("bias", "i32"), ("deg", "i32"), ("gene", "u16"), ("out", "u8"), ("prod", "i32"),
           ("rate", "i32")),
    "enz": (("gene", "u16"), ("kcat", "i32"), ("km", "i32"), ("p1", "u8"), ("p2", "u8"),
            ("s1", "u8"), ("s2", "u8"), ("yield", "i32")),
    "rec": (("channel", "u8"), ("gain", "i32"), ("gene", "u16"), ("species", "u8"),
            ("threshold", "i32")),
    "morph": (("angle", "i32"), ("gene", "u16"), ("guard", "u8"), ("pred", "u8"), ("rate", "i32"),
              ("succ", "u8"), ("threshold", "i32")),
    "imm": (("class", "u8"), ("cost", "i32"), ("gene", "u16"), ("strength", "i32")),
    "pig": (("allele", "u8"), ("class", "u8"), ("gene", "u16")),
    "te": (("activity", "i32"), ("gene", "u16"), ("target_bias", "u8")),
    "loads": (("effect", "i32"), ("gene", "u16"), ("target", "u8")),
}
_FIELD_OF_COLUMN = {"km": "Km"}
_TABLE_OF_KIND = {"tf": "tf", "enz": "enz", "rec": "rec", "morph": "morph", "imm": "imm",
                  "pig": "pig", "te": "te"}
RESERVED = ("brain", "lex", "plast", "prc", "shape", "temp", "vern")


def _combine(kind_code, mode, alleles):
    """One locus's values from its one or two alleles ``(values, body)``."""
    if len(alleles) == 1:
        return list(alleles[0][0])
    (va, ba), (vb, bb) = alleles
    if mode == g.DOM_MAX:
        key_a = (g.strength(kind_code, va, ba), bytes(ba))
        key_b = (g.strength(kind_code, vb, bb), bytes(bb))
        return list(va) if key_a >= key_b else list(vb)
    fields = g.KIND_BY_CODE[kind_code][K_FIELDS]
    out = []
    for field, a, b in zip(fields, va, vb):
        if field[F_MARKS] & g.S:
            if a != b:
                raise wire.Refused("engine_panic", "genome compile")
            out.append(a)
        else:
            out.append((a + b) // 2)
    return out


def _scaled(kind_code, values):
    kind = g.KIND_BY_CODE[kind_code]
    if kind[K_RESERVED]:
        return {field[F_NAME]: value for field, value in zip(kind[K_FIELDS], values)}
    return {field[F_NAME]: value * field[F_SCALE] for field, value in zip(kind[K_FIELDS], values)}


def compile_genome(data, law, lawview, law_digest):
    """``(tables, work)`` for canonical genome bytes under a validated law."""
    chromosomes = g._decode(data, lawview)
    # Where each locus sits, per chromosome, and the promoter before each record.
    placed = []
    for chrom in chromosomes:
        at = {}
        promoter = []
        run = []
        for index, (record, body) in enumerate(chrom):
            ident = record["locus"]
            if record["kind"] == g.KIND_CIS:
                run.append(index)
                promoter.append(())
            else:
                promoter.append(tuple(run))
                run = []
            at[ident] = index
        placed.append((at, promoter))
    genes = []  # (ident, kind_code, dose, values, edges)
    edges_total = 0
    for ident in lawview.order:
        locus = lawview.loci[ident]
        if locus.kind == g.KIND_CIS:
            continue
        pair = ident >> 8
        alleles = []
        cis_occurrences = {}
        for h in range(g.PLOIDY):
            c = pair * g.PLOIDY + h
            at, promoter = placed[c]
            if ident not in at:
                continue
            index = at[ident]
            record, body = chromosomes[c][index]
            alleles.append((record["fields"], body))
            for cis_index in promoter[index]:
                cis_record, cis_body = chromosomes[c][cis_index]
                cis_occurrences.setdefault(cis_record["locus"], []).append(
                    (cis_record["fields"], cis_body))
        if not alleles:
            continue
        values = _combine(locus.kind, locus.mode, alleles)
        loaded = None
        if locus.kind == g.KIND_LOAD and len(alleles) == 2 \
                and alleles[0][0][1] > 0 and alleles[1][0][1] > 0:
            loaded = values
        edges = []
        for cis_id in sorted(cis_occurrences):
            cis_locus = lawview.loci[cis_id]
            edge = _combine(g.KIND_CIS, cis_locus.mode, cis_occurrences[cis_id])
            edges.append((cis_id, edge))
        edges_total += len(edges)
        genes.append((ident, locus, len(alleles), values, edges, loaded))
    tables = _emit_tables(genes, lawview, data, law_digest)
    records = 0
    for chrom in chromosomes:
        records += len(chrom)
    work = (len(data) + 63) // 64 + records + edges_total
    return tables, work


def _column(kind, values):
    return wire.pack_bulk(kind, values)


def _emit_tables(genes, lawview, data, law_digest):
    columns = {table: {name: [] for name, _ in cols} for table, cols in _COLUMNS.items()}
    reserved = {name: {"fields": [], "gene": []} for name in RESERVED}
    gene_cols = {"dose": [], "edge_start": [0], "kind": [], "locus": [], "stages": []}
    edge_cols = {"cis": [], "k": [], "mode": [], "n": [], "src": [], "w": []}
    tf_deg = {}
    for row, (ident, locus, dose, values, edges, loaded) in enumerate(genes):
        kind = g.KIND_BY_CODE[locus.kind]
        name = kind[K_NAME]
        gene_cols["dose"].append(dose)
        gene_cols["kind"].append(locus.kind)
        gene_cols["locus"].append(ident)
        gene_cols["stages"].append(locus.stages)
        for cis_id, edge in edges:
            scaled = _scaled(g.KIND_CIS, edge)
            edge_cols["cis"].append(cis_id)
            edge_cols["k"].append(scaled["K"])
            edge_cols["mode"].append(scaled["mode"])
            edge_cols["n"].append(scaled["n"])
            edge_cols["src"].append(scaled["src"])
            edge_cols["w"].append(scaled["w"])
        gene_cols["edge_start"].append(len(edge_cols["cis"]))
        if kind[K_RESERVED]:
            reserved[name]["gene"].append(row)
            reserved[name]["fields"].extend(values)
            continue
        if name == "load":
            if loaded is not None:
                columns["loads"]["effect"].append(loaded[1])
                columns["loads"]["gene"].append(row)
                columns["loads"]["target"].append(loaded[0])
            continue
        table = _TABLE_OF_KIND[name]
        scaled = _scaled(locus.kind, values)
        for column, _ in _COLUMNS[table]:
            if column == "gene":
                columns[table]["gene"].append(row)
            else:
                columns[table][column].append(scaled[_FIELD_OF_COLUMN.get(column, column)])
        if name == "tf":
            out = scaled["out"]
            tf_deg[out] = max(tf_deg.get(out, 0), scaled["deg"])
    species_deg = []
    for code in range(g.SPECIES):
        if lawview.classes[code] in ("input", "conserved"):
            species_deg.append(0)
        elif code in tf_deg:
            species_deg.append(tf_deg[code])
        else:
            species_deg.append(lawview.deg[code])
    tables = {
        "edges": {name: _column(kind, edge_cols[name]) for name, kind in
                  (("cis", "u16"), ("k", "i32"), ("mode", "u8"), ("n", "u8"), ("src", "u8"),
                   ("w", "i32"))},
        "genes": {name: _column(kind, gene_cols[name]) for name, kind in
                  (("dose", "u8"), ("edge_start", "u32"), ("kind", "u8"), ("locus", "u16"),
                   ("stages", "u16"))},
        "genome": g.sha256(data),
        "law": law_digest,
        "reserved": {name: {"fields": _column("i32", reserved[name]["fields"]),
                            "gene": _column("u16", reserved[name]["gene"])} for name in RESERVED},
        "schema": TABLES_SCHEMA,
        "species": {"deg": _column("i32", species_deg)},
    }
    for table, cols in _COLUMNS.items():
        tables[table] = {name: _column(kind, columns[table][name]) for name, kind in cols}
    return tables


def tables_digest(tables):
    return g.sha256(wire.emit(tables))
