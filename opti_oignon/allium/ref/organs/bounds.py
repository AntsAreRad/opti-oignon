"""The static bound analysis a law records about its genome.

``compute(law)`` returns ``{"genome_bounds": ..., "stock_bounds": ...}``,
computed from the law's kind boxes, its loci and the fixed-point constants.
A law file records the result, and a contract recomputes it: a box changed
without re-recording the analysis is caught, and so is a bound that leaves
its ceiling. This is a law-authoring check, not an engine operation, and it
exists in Python only.

Every ``premise`` and ``proof`` is printable ASCII, compared byte for byte.

``genome_bounds`` covers the factors a genome can supply: promoter terms,
activation sums, gains, Hill constants and their powers, Michaelis-Menten
intermediates. ``stock_bounds`` is a skeleton: for each conserved species,
the flow a genome's enzymes could move in one step. The terms the genome
does not supply (provision, costs, litter, rain, evaporation, the stock
ceiling) are owed to the physiology.
"""

from ... import fx

checkpoint_before_apply = True

I32_MAX = (1 << 31) - 1
OCJ_MAX = (1 << 53) - 1
STOCK_OWED = ("the physiology: provision, brain_cost, defense_cost, litter, rain, evaporation "
              "and the stock ceiling")
STOCK_PREMISE = "a reaction moves at most mul(kcat, ONE) of each substrate or product per step"


def _pow_raw(c, n):
    x = c
    for _ in range(n - 1):
        x = (x * c) >> 16
    return x


def _kind_box(genome, kind_name, field_name):
    for entry in genome["kinds"]:
        if entry["name"] == kind_name:
            for field in entry["fields"]:
                if field["name"] == field_name:
                    return field["lo"], field["hi"]
    raise KeyError(f"{kind_name}.{field_name}")


def _max_abs(box):
    return max(abs(box[0]), abs(box[1]))


def _entry(ceiling, max_raw, premise, proof):
    return {"ceiling": ceiling, "max_raw": max_raw, "premise": premise, "proof": proof}


def within(ceiling, max_raw, max_records):
    """True when ``max_raw`` meets its ceiling."""
    if ceiling == "i32":
        return max_raw <= I32_MAX
    if ceiling == "i64":
        return max_raw <= OCJ_MAX
    if ceiling == "ge1":
        return max_raw >= 1
    if ceiling == "le_cmax":
        return max_raw <= fx.CMAX
    if ceiling == "count":
        return max_raw <= max_records
    return False


def compute(law):
    genome = law["genome"]
    cmax = fx.CMAX
    one = fx.ONE
    max_promoter = genome["max_promoter"]
    per_pair = [0] * genome["pairs"]
    for entry in genome["loci"]:
        if entry["kind"] == "cis":
            per_pair[entry["id"] >> 8] += 1
    cis_on_pair_max = max(per_pair)
    edges = min(2 * max_promoter, cis_on_pair_max)
    w = _max_abs(_kind_box(genome, "cis", "w"))
    b = _max_abs(_kind_box(genome, "tf", "bias"))
    gain = _max_abs(_kind_box(genome, "rec", "gain"))
    k_lo_raw = min(_kind_box(genome, "cis", "K")[0], _kind_box(genome, "enz", "Km")[0])
    k_hi_raw = max(_kind_box(genome, "cis", "K")[1], _kind_box(genome, "enz", "Km")[1])
    n_max = _kind_box(genome, "cis", "n")[1]
    k_min = k_lo_raw * 8
    k_max = k_hi_raw * 8
    k_floor = _pow_raw(k_min, n_max)
    pow_max = _pow_raw(cmax, n_max)
    den_k = _pow_raw(k_max, n_max)
    before_last = _pow_raw(cmax, n_max - 1)
    hill_intermediate = max(before_last * cmax, pow_max * one)
    activation = b * 16 + edges * w * 16
    bounds = {
        "edges_per_gene_max": _entry(
            "count", edges, "none",
            f"min(2*max_promoter, cis_on_pair_max) = min(2*{max_promoter}, {cis_on_pair_max}) = {edges}"),
        "promoter_term_max": _entry("i32", w * 16, "act in [0, ONE]", f"W*16 = {w}*16 = {w * 16}"),
        "activation_max": _entry(
            "i32", activation, "act in [0, ONE]",
            f"B*16 + E*W*16 = {b}*16 + {edges}*{w}*16 = {activation}"),
        "gain_max": _entry("i32", gain * 16, "none", f"G*16 = {gain}*16 = {gain * 16}"),
        "k_min": _entry("ge1", k_min, "none", f"min(K lo, Km lo)*8 = {k_lo_raw}*8 = {k_min}"),
        "hill_k_floor": _entry(
            "ge1", k_floor, "none", f"pow_raw(k_min, n_max) = pow_raw({k_min}, {n_max}) = {k_floor}"),
        "k_max": _entry("le_cmax", k_max, "none", f"max(K hi, Km hi)*8 = {k_hi_raw}*8 = {k_max}"),
        "hill_pow_max": _entry(
            "i32", pow_max, "c in [0, CMAX]",
            f"pow_raw(CMAX, {n_max}) = pow_raw({cmax}, {n_max}) = {pow_max}"),
        "hill_den_max": _entry(
            "i32", den_k + pow_max, "c in [0, CMAX]",
            f"pow_raw(k_max, {n_max}) + hill_pow_max = {den_k} + {pow_max} = {den_k + pow_max}"),
        "hill_intermediate_max": _entry(
            "i64", hill_intermediate, "c in [0, CMAX]",
            f"max({before_last}*{cmax}, {pow_max}*{one}) = {hill_intermediate}"),
        "mm_intermediate_max": _entry(
            "i64", cmax * one, "c in [0, CMAX]", f"CMAX*{one} = {cmax}*{one} = {cmax * one}"),
    }
    stocks = {}
    enz_loci = []
    for entry in genome["loci"]:
        if entry["kind"] == "enz":
            fixed = {box["field"]: box["lo"] for box in entry["box"] if box["lo"] == box["hi"]}
            enz_loci.append(fixed)
    for species in genome["species"]:
        if species["class"] != "conserved":
            continue
        code = species["code"]
        writers = 0
        readers = 0
        for fixed in enz_loci:
            if fixed.get("p1") == code or fixed.get("p2") == code:
                writers += 1
            if fixed.get("s1") == code or fixed.get("s2") == code:
                readers += 1
        flow = (writers + readers) * 65535
        stocks[species["name"]] = {
            "enz_readers": readers,
            "enz_writers": writers,
            "flow_max_raw": flow,
            "owed": STOCK_OWED,
            "premise": STOCK_PREMISE,
            "proof": f"({writers}+{readers})*65535 = {flow}",
        }
    return {"genome_bounds": bounds, "stock_bounds": stocks}


def defects(law):
    """Every recorded bound that differs from the recomputed one or leaves its ceiling."""
    out = []
    computed = compute(law)
    for section in ("genome_bounds", "stock_bounds"):
        if law.get(section) != computed[section]:
            out.append(f"{section}: recorded analysis differs from the recomputed one")
    max_records = law["genome"]["max_records"]
    for name, entry in computed["genome_bounds"].items():
        if not within(entry["ceiling"], entry["max_raw"], max_records):
            out.append(f"genome_bounds: {name} leaves its ceiling")
    return out
