//! The engine without backtracking, read side: `linear_engine` (Python, the reference) compiles a pattern into
//! Pike programs and hands them here as data; this module reads them over a text exactly as the reference does.
//!
//! Every step mirrors the reference: the classes of characters, the anchors, the lookarounds' tables (a lookahead by
//! the backward pass of its body, a lookbehind by a forward pass), the backward pass with its moves kept up to a bound
//! and its windows read again, and the Pike machine pruned by it, groups and empty rounds included. The work is
//! counted in the reference's own units, so that the contracts that hold the reference to linear work hold this
//! read to the same numbers. What the reference reads and this module does not (groups in a positive lookaround,
//! more than 64 sets of characters or tests) is refused at compilation, by name, and the reference reads it.

use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyStringData};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

/// The version of the data a program is handed in; a program of another version is refused.
pub const FORMAT: u32 = 1;

const CHAR: u8 = 0;
const SPLIT: u8 = 1;
const JMP: u8 = 2;
const MATCH: u8 = 3;
const ASSERT: u8 = 4;
const ORACLE: u8 = 5;
const SAVE: u8 = 6;
const MARK: u8 = 7;
const CHECK: u8 = 8;
const GUARD: u8 = 9;
const FAIL: i64 = -1;
const BIG: i64 = 1 << 30;

// The anchors, numbered as the Python side numbers them.
const AT_BEGINNING: u8 = 0;
const AT_BEGINNING_STRING: u8 = 1;
const AT_END: u8 = 2;
const AT_END_STRING: u8 = 3;
const AT_BOUNDARY: u8 = 4;
const AT_NON_BOUNDARY: u8 = 5;

struct Program {
    kind: Vec<u8>,
    a: Vec<i64>,
    b: Vec<i64>,
    c: Vec<i64>,
    depth: Vec<i64>,
    chars: Vec<(usize, u32)>,
    ends: Vec<usize>,
    back: Vec<Vec<(usize, i64)>>,
    anchored: bool,
    oracle_tests: Vec<(u32, usize, bool)>,
    width: usize,
    // where the keys of the Pike machine's `seen` start for each instruction: a key is (pc, j), j <= depth[pc]
    key_at: Vec<usize>,
    keys: usize,
}

enum Test {
    At { kind: u8, multiline: bool, ascii: bool },
    Look,
}

struct Look {
    behind: bool,
    scan: usize,
}

/// (members, ends a line, a word character in Unicode, in ASCII): what a class of characters answers.
type Info = (u64, bool, bool, bool);

struct Cache {
    moves: HashMap<(u32, i64, u64, u8), u32>,
    states: Vec<Arc<[u8]>>,
    numbers: HashMap<Arc<[u8]>, u32>,
    // bumped at every clear: a state's number taken in this generation still names it (states are only let go
    // all at once), so a pass that carries the number need not hash the state again
    generation: u64,
}

/// A state of the backward pass and, when known in the cache's current generation, its number there.
#[derive(Clone)]
struct Held {
    state: Arc<[u8]>,
    number: Option<(u32, u64)>,
}

impl Cache {
    fn new() -> Self {
        Cache { moves: HashMap::new(), states: Vec::new(), numbers: HashMap::new(), generation: 0 }
    }

    fn number(&mut self, state: &Arc<[u8]>) -> u32 {
        if let Some(found) = self.numbers.get(state) {
            return *found;
        }
        let found = self.states.len() as u32;
        self.numbers.insert(state.clone(), found);
        self.states.push(state.clone());
        found
    }

    fn clear(&mut self) {
        self.moves.clear();
        self.states.clear();
        self.numbers.clear();
        self.generation += 1;
    }

    /// The number of a held state: the one it carries if taken in this generation, else `number` (which keeps it).
    fn number_of(&mut self, held: &Held) -> u32 {
        match held.number {
            Some((number, generation)) if generation == self.generation => number,
            _ => self.number(&held.state),
        }
    }
}

/// What the reads of one pattern share and fill: the classes of characters, the anchors by context, the moves.
struct Kept {
    classes: HashMap<u32, u16>,
    signatures: HashMap<Info, u32>,
    infos: Vec<Info>,
    contexts: HashMap<(i64, i64, bool, bool, bool), u64>,
    caches: Vec<Cache>, // one per program, then the main program's cache of final-only passes
}

struct Shared {
    sets: Vec<Vec<(u32, u32)>>,
    word: Vec<(u32, u32)>,
    word_ascii: Vec<(u32, u32)>,
    tests: Vec<Test>,
    asserts: Vec<usize>,
    looks: Vec<Look>,
    programs: Vec<Program>,
    main: usize,
    ncaps: usize,
    boundary_reads_empty: bool,
    // the code points re's search tries a start at, when it compiles a set of them (None: every start)
    tried: Option<Vec<(u32, u32)>>,
}

/// A buffer of `len` copies of `value`, or MemoryError: a buffer as long as the text is reserved so that a failure
/// is the caller's error, as it is for the reference, never an abort of the process.
fn buffer<T: Clone>(len: usize, value: T) -> PyResult<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err(format!("linear read: no memory for a buffer of {len} positions")))?;
    out.resize(len, value);
    Ok(out)
}

fn within(ranges: &[(u32, u32)], point: u32) -> bool {
    let at = ranges.partition_point(|&(first, _)| first <= point);
    at > 0 && ranges[at - 1].1 >= point
}

fn holds(state: &[u8], pc: usize) -> bool {
    state[pc >> 3] >> (pc & 7) & 1 == 1
}

fn anchor(shared: &Shared, kind: u8, multiline: bool, ascii: bool, prev: Option<&Info>, cur: Option<&Info>,
          first: bool, last: bool, penultimate: bool) -> bool {
    match kind {
        AT_BEGINNING => first || (multiline && prev.map_or(false, |p| p.1)),
        AT_BEGINNING_STRING => first,
        AT_END => {
            if multiline {
                last || cur.map_or(false, |c| c.1)
            } else {
                last || (penultimate && cur.map_or(false, |c| c.1))
            }
        }
        AT_END_STRING => last,
        _ => {
            if first && last && !shared.boundary_reads_empty {
                return false;
            }
            let word = |info: &Info| if ascii { info.3 } else { info.2 };
            let before = prev.map_or(false, word);
            let after = cur.map_or(false, word);
            (before != after) == (kind == AT_BOUNDARY)
        }
    }
}

/// One read of one text up to `end`: the class of each character, the lookarounds' tables, the backward passes.
struct Read {
    // the keys the Pike machine takes, kept between its calls (stamped: reused without clearing)
    seen: Seen,
    // whether re's search tries a start at each position (a bit each), when it compiles a set of starts
    tried_at: Vec<u64>,
    end: usize,
    most_moves: usize,
    window: usize,
    cls: Vec<u16>,
    oracles: Vec<Vec<u8>>,
    alive: [Option<Alive>; 2],
    work: u64,
}

struct Alive {
    start: usize,
    end: usize,
    final_only: bool,
    marks: HashMap<usize, Arc<[u8]>>,
    window: Option<usize>,
    sets: Vec<Arc<[u8]>>,
    base: usize,
    size: usize,
    starts: Vec<u8>,
}

struct Reader<'a> {
    shared: &'a Shared,
    kept: &'a mut Kept,
    read: &'a mut Read,
}

impl<'a> Reader<'a> {
    fn cls(&self, at: i64) -> i64 {
        if at < 0 || at as usize >= self.read.end {
            -1
        } else {
            self.read.cls[at as usize] as i64
        }
    }

    fn members(&self, cls: i64) -> Option<u64> {
        if cls < 0 { None } else { Some(self.kept.infos[cls as usize].0) }
    }

    fn anchors(&mut self, before: i64, here: i64, first: bool, last: bool, penultimate: bool) -> u64 {
        let key = (before, here, first, last, penultimate);
        if let Some(found) = self.kept.contexts.get(&key) {
            return *found;
        }
        let shared = self.shared;
        let prev = if before < 0 { None } else { Some(self.kept.infos[before as usize]) };
        let cur = if here < 0 { None } else { Some(self.kept.infos[here as usize]) };
        let mut found = 0u64;
        for &test_id in &shared.asserts {
            if let Test::At { kind, multiline, ascii } = shared.tests[test_id] {
                if anchor(shared, kind, multiline, ascii, prev.as_ref(), cur.as_ref(), first, last, penultimate) {
                    found |= 1 << test_id;
                }
            }
        }
        if self.kept.contexts.len() >= self.read.most_moves {
            self.kept.contexts.clear();
        }
        self.kept.contexts.insert(key, found);
        found
    }

    fn bits(&mut self, prog: usize, at: usize, here: i64) -> u64 {
        let program = &self.shared.programs[prog];
        let mut bits = 0u64;
        if program.anchored {
            let end = self.read.end;
            let before = if at > 0 { self.cls(at as i64 - 1) } else { -1 };
            bits = self.anchors(before, here, at == 0, at == end, at + 1 == end);
        }
        for &(test_id, index, positive) in &program.oracle_tests {
            if (self.read.oracles[index][at] == 1) == positive {
                bits |= 1 << test_id;
            }
        }
        bits
    }

    fn tested(&self, prog: usize) -> bool {
        let program = &self.shared.programs[prog];
        program.anchored || !program.oracle_tests.is_empty()
    }

    /// The instructions at a position from which a match can be completed (the reference's `_completes`).
    fn completes(&self, prog: usize, following: &[u8], members: Option<u64>, bits: u64, may_end: bool) -> Arc<[u8]> {
        let program = &self.shared.programs[prog];
        let mut stack: Vec<usize> = Vec::new();
        if let Some(members) = members {
            for &(pc, set_id) in &program.chars {
                let after = pc + 1;
                if members >> set_id & 1 == 1 && holds(following, after) {
                    stack.push(pc);
                }
            }
        }
        if may_end {
            stack.extend(program.ends.iter().copied());
        }
        let mut seen = vec![0u8; program.width];
        while let Some(pc) = stack.pop() {
            if holds(&seen, pc) {
                continue;
            }
            seen[pc >> 3] |= 1 << (pc & 7);
            for &(before, test_id) in &program.back[pc] {
                if test_id < 0 || bits >> test_id & 1 == 1 {
                    stack.push(before);
                }
            }
        }
        Arc::from(seen.into_boxed_slice())
    }

    fn cache_of(&self, prog: usize, final_only: bool) -> usize {
        if final_only { self.shared.programs.len() } else { prog }
    }

    /// The backward move of `_Alive.move`, its cache and its work as the reference keeps them.
    fn move_back(&mut self, prog: usize, final_only: bool, end: usize, following: &Held, at: usize) -> Held {
        let cls = self.cls(at as i64);
        let bits = if self.tested(prog) { self.bits(prog, at, cls) } else { 0 };
        let may_end = !final_only || at == end;
        let ci = self.cache_of(prog, final_only);
        let number = self.kept.caches[ci].number_of(following);
        let key = (number, cls, bits, may_end as u8);
        let generation = self.kept.caches[ci].generation;
        if let Some(&found) = self.kept.caches[ci].moves.get(&key) {
            return Held { state: self.kept.caches[ci].states[found as usize].clone(), number: Some((found, generation)) };
        }
        let state = self.completes(prog, &following.state, self.members(cls), bits, may_end);
        let program = &self.shared.programs[prog];
        self.read.work += (program.kind.len() + program.chars.len()) as u64;
        let cache = &mut self.kept.caches[ci];
        if cache.moves.len() >= self.read.most_moves {
            cache.clear();
            return Held { state, number: None };
        }
        let number = cache.number(&state);
        cache.moves.insert(key, number);
        Held { state, number: Some((number, cache.generation)) }
    }

    fn alive(&mut self, prog: usize, start: usize, end: usize, final_only: bool) -> PyResult<Alive> {
        let size = self.read.window;
        let empty: Arc<[u8]> = Arc::from(vec![0u8; self.shared.programs[prog].width].into_boxed_slice());
        // one more than the positions: a lookahead's table takes the place past the end without a new buffer
        let mut starts = buffer(end - start + 2, 0u8)?;
        starts.truncate(end - start + 1);
        let mut marks = HashMap::new();
        let mut held = Held { state: empty, number: None };
        self.read.work += (end - start + 1) as u64;
        let mut at = end as i64;
        while at >= start as i64 {
            let place = at as usize;
            held = self.move_back(prog, final_only, end, &held, place);
            if held.state[0] & 1 == 1 {
                starts[place - start] = 1;
            }
            if place == end || (place - start) % size == 0 {
                marks.insert(place, held.state.clone());
            }
            at -= 1;
        }
        Ok(Alive { start, end, final_only, marks, window: None, sets: Vec::new(), base: 0, size, starts })
    }

    /// The set of the backward pass at a position, its window read again when the machine reaches it.
    fn alive_at(&mut self, prog: usize, which: usize, at: usize) -> Arc<[u8]> {
        let (start, end, size, final_only, current) = {
            let alive = self.read.alive[which].as_ref().unwrap();
            (alive.start, alive.end, alive.size, alive.final_only, alive.window)
        };
        let window = (at - start) / size;
        if current != Some(window) {
            let base = start + window * size;
            let top = (base + size).min(end);
            let mark = self.read.alive[which].as_ref().unwrap().marks[&top].clone();
            let mut sets: Vec<Arc<[u8]>> = vec![mark.clone(); top - base + 1];
            let mut held = Held { state: mark, number: None };
            self.read.work += (top - base) as u64;
            let mut place = top as i64 - 1;
            while place >= base as i64 {
                held = self.move_back(prog, final_only, end, &held, place as usize);
                sets[place as usize - base] = held.state.clone();
                place -= 1;
            }
            let alive = self.read.alive[which].as_mut().unwrap();
            alive.window = Some(window);
            alive.sets = sets;
            alive.base = base;
        }
        let alive = self.read.alive[which].as_ref().unwrap();
        alive.sets[at - alive.base].clone()
    }

    /// The forward set of a lookbehind's pass (the reference's `_forward_set`).
    fn forward_set(&self, prog: usize, state: &[u8], members: Option<u64>, bits: u64) -> (Arc<[u8]>, bool) {
        let program = &self.shared.programs[prog];
        let mut stack: Vec<usize> = vec![0];
        for pc in 0..program.kind.len() {
            if holds(state, pc) {
                stack.push(pc);
            }
        }
        let mut seen = vec![0u8; program.width];
        let mut live = vec![0u8; program.width];
        let mut matched = false;
        while let Some(pc) = stack.pop() {
            if holds(&seen, pc) {
                continue;
            }
            seen[pc >> 3] |= 1 << (pc & 7);
            match program.kind[pc] {
                JMP => stack.push(program.a[pc] as usize),
                SPLIT => {
                    stack.push(program.a[pc] as usize);
                    stack.push(program.b[pc] as usize);
                }
                SAVE | MARK | GUARD => stack.push(pc + 1),
                CHECK => {
                    for target in [program.b[pc], program.c[pc]] {
                        if target != FAIL {
                            stack.push(target as usize);
                        }
                    }
                }
                ASSERT | ORACLE => {
                    if bits >> program.a[pc] & 1 == 1 {
                        stack.push(pc + 1);
                    }
                }
                MATCH => matched = true,
                _ => {
                    if let Some(members) = members {
                        if members >> program.a[pc] & 1 == 1 {
                            let after = pc + 1;
                            live[after >> 3] |= 1 << (after & 7);
                        }
                    }
                }
            }
        }
        (Arc::from(live.into_boxed_slice()), matched)
    }

    fn oracle(&mut self, index: usize) -> PyResult<Vec<u8>> {
        let look = &self.shared.looks[index];
        let prog = look.scan;
        let end = self.read.end;
        if !look.behind {
            let alive = self.alive(prog, 0, end, false)?;
            let mut table = alive.starts;
            table.push(0);
            return Ok(table);
        }
        let mut table = buffer(end + 2, 0u8)?;
        let mut state: Arc<[u8]> = Arc::from(vec![0u8; self.shared.programs[prog].width].into_boxed_slice());
        self.read.work += (end + 1) as u64;
        for at in 0..=end {
            let cls = self.cls(at as i64);
            let bits = if self.tested(prog) { self.bits(prog, at, cls) } else { 0 };
            let number = self.kept.caches[prog].number(&state);
            let key = (number, cls, bits, 2u8);
            let matched;
            if let Some(&found) = self.kept.caches[prog].moves.get(&key) {
                let pair = self.kept.caches[prog].states[found as usize].clone();
                // a kept forward move is the pair (state, matched): matched rides in a last byte
                matched = pair[pair.len() - 1] == 1;
                state = Arc::from(pair[..pair.len() - 1].to_vec().into_boxed_slice());
            } else {
                let (next, ended) = self.forward_set(prog, &state, self.members(cls), bits);
                let program = &self.shared.programs[prog];
                self.read.work += (program.kind.len() + program.chars.len()) as u64;
                let cache = &mut self.kept.caches[prog];
                if cache.moves.len() >= self.read.most_moves {
                    cache.clear();
                } else {
                    let mut pair = next.to_vec();
                    pair.push(ended as u8);
                    let pair: Arc<[u8]> = Arc::from(pair.into_boxed_slice());
                    let number = cache.number(&pair);
                    cache.moves.insert(key, number);
                }
                state = next;
                matched = ended;
            }
            table[at] = matched as u8;
        }
        Ok(table)
    }

    /// The threads reached from (pc, j) at `at` by steps that read nothing (the reference's `_close`).
    #[allow(clippy::too_many_arguments)]
    fn close(&self, prog: usize, pc: i64, j: i64, caps: Rc<Vec<i64>>, at: usize, end: usize, bits: u64,
             seen: &mut Seen, out: &mut Vec<(usize, Rc<Vec<i64>>)>, stack: &mut Vec<(i64, i64, Rc<Vec<i64>>)>) {
        let program = &self.shared.programs[prog];
        stack.clear();
        stack.push((pc, j, caps));
        while let Some((pc, j, caps)) = stack.pop() {
            if pc == FAIL {
                continue;
            }
            let p = pc as usize;
            let kind = program.kind[p];
            let jj = if kind == CHAR || kind == MATCH { 0 } else { j.min(program.depth[p]) };
            if !seen.take(program.key_at[p] + jj as usize) {
                continue;
            }
            match kind {
                JMP => stack.push((program.a[p], j, caps)),
                SPLIT => {
                    stack.push((program.b[p], j, caps.clone()));
                    stack.push((program.a[p], j, caps));
                }
                SAVE => {
                    let slot = program.a[p] as usize;
                    let mut changed = (*caps).clone();
                    changed[slot] = at as i64;
                    if slot & 1 == 1 && slot > 1 {
                        let last = changed.len() - 1;
                        changed[last] = (slot / 2) as i64;
                    }
                    stack.push((pc + 1, j, Rc::new(changed)));
                }
                ASSERT | ORACLE => {
                    if bits >> program.a[p] & 1 == 1 {
                        stack.push((pc + 1, j, caps));
                    }
                }
                MARK => stack.push((pc + 1, j.min(program.a[p]), caps)),
                GUARD => {
                    if at <= end {
                        stack.push((pc + 1, j, caps));
                    }
                }
                CHECK => stack.push((if program.a[p] < j { program.b[p] } else { program.c[p] }, j, caps)),
                _ => out.push((p, caps)),
            }
        }
    }

    /// The caps of the leftmost-first match from `start`, or None (the reference's `_pike`).
    fn pike(&mut self, prog: usize, start: usize, final_only: bool, refuse: i64, which: Option<usize>) -> Option<Vec<i64>> {
        let end = self.read.end;
        let finish = end;
        let mut caps0 = vec![-1i64; self.shared.ncaps + 1];
        caps0[self.shared.ncaps] = -1;
        let caps0 = Rc::new(caps0);
        let mut at = start;
        let mut here = self.cls(at as i64);
        let tested = self.tested(prog);
        let bits = if tested { self.bits(prog, at, here) } else { 0 };
        let keys = self.shared.programs[prog].keys;
        let mut seen = std::mem::replace(&mut self.read.seen, Seen::new(0));
        if seen.stamp.len() < keys {
            seen = Seen::new(keys);
        }
        seen.reset();
        let mut threads = Vec::new();
        let mut stack = Vec::new();
        self.close(prog, 0, BIG, caps0, at, end, bits, &mut seen, &mut threads, &mut stack);
        let mut matched = None;
        let mut work: u64 = 0;
        let mut following = Vec::new();
        while !threads.is_empty() {
            work += threads.len() as u64;
            following.clear();
            seen.reset();
            let mut live: Option<Arc<[u8]>> = None;
            let mut members = 0u64;
            let mut nxt = at;
            let mut there = -1;
            let mut nbits = 0u64;
            if at < finish {
                members = self.members(here).unwrap_or(0);
                nxt = at + 1;
                there = self.cls(nxt as i64);
                nbits = if tested { self.bits(prog, nxt, there) } else { 0 };
                if let Some(which) = which {
                    live = Some(self.alive_at(prog, which, nxt));
                }
            }
            for (pc, caps) in threads.iter() {
                let program = &self.shared.programs[prog];
                if program.kind[*pc] == MATCH {
                    if final_only && at != end {
                        continue;
                    }
                    if refuse >= 0 && at as i64 == refuse {
                        continue;
                    }
                    matched = Some((**caps).clone());
                    break;
                }
                if at < finish && members >> program.a[*pc] & 1 == 1
                    && live.as_ref().map_or(true, |l| holds(l, *pc + 1)) {
                    self.close(prog, *pc as i64 + 1, BIG, caps.clone(), nxt, end, nbits, &mut seen, &mut following,
                               &mut stack);
                }
            }
            work += seen.count as u64;
            if at >= finish {
                break;
            }
            at = nxt;
            here = there;
            std::mem::swap(&mut threads, &mut following);
        }
        self.read.work += work;
        self.read.seen = seen;
        matched
    }
}

/// The keys of the Pike machine taken at one position: stamped, so that a reset costs nothing.
struct Seen {
    stamp: Vec<u32>,
    now: u32,
    count: usize,
}

impl Seen {
    fn new(keys: usize) -> Self {
        Seen { stamp: vec![0; keys], now: 1, count: 0 }
    }

    fn take(&mut self, key: usize) -> bool {
        if self.stamp[key] == self.now {
            return false;
        }
        self.stamp[key] = self.now;
        self.count += 1;
        true
    }

    fn reset(&mut self) {
        if self.now == u32::MAX {
            // every stamp is let go before the count wraps: a key taken long ago is never read as taken now
            self.stamp.iter_mut().for_each(|stamp| *stamp = 0);
            self.now = 0;
        }
        self.now += 1;
        self.count = 0;
    }
}

// -- what Python sees ----------------------------------------------------------------------------------------------

/// Run a read with the GIL released: other Python threads run while a long text is read.
fn released<T: Send, F: FnOnce() -> T + Send>(py: Python<'_>, read: F) -> T {
    py.allow_threads(read)
}

type Code = Vec<(u8, i64, i64, i64)>;
type PlanProgram = (Code, Vec<i64>, Vec<(u32, usize, bool)>, bool);
type Plan = (
    u32,
    Vec<Vec<(u32, u32)>>,
    Vec<(u32, u32)>,
    Vec<(u32, u32)>,
    Vec<(u8, i64, bool, bool)>,
    Vec<(bool, usize)>,
    Vec<PlanProgram>,
    (usize, usize, bool),
    Option<Vec<(u32, u32)>>,
);

fn program_from(plan: PlanProgram, tests: usize, sets: usize, ncaps: usize) -> PyResult<Program> {
    let (code, depth, oracle_tests, anchored) = plan;
    let n = code.len();
    if depth.len() != n || n == 0 {
        return Err(PyValueError::new_err("linear program: code and depths differ in length"));
    }
    let bad = |what: &str| PyValueError::new_err(format!("linear program: {what}"));
    let target = |t: i64| t == FAIL || (t >= 0 && (t as usize) < n);
    let mut kind = Vec::with_capacity(n);
    let (mut a, mut b, mut c) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut chars = Vec::new();
    let mut ends = Vec::new();
    let mut back: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n];
    for (pc, &(k, x, y, z)) in code.iter().enumerate() {
        let next_ok = pc + 1 < n;
        match k {
            CHAR => {
                if x < 0 || x as usize >= sets || x >= 64 || !next_ok {
                    return Err(bad("a step on a set it does not hold"));
                }
                chars.push((pc, x as u32));
            }
            SPLIT => {
                if !target(x) || !target(y) || x == FAIL || y == FAIL {
                    return Err(bad("a split out of the program"));
                }
                back[x as usize].push((pc, -1));
                back[y as usize].push((pc, -1));
            }
            JMP => {
                if !target(x) || x == FAIL {
                    return Err(bad("a jump out of the program"));
                }
                back[x as usize].push((pc, -1));
            }
            MATCH => ends.push(pc),
            ASSERT | ORACLE => {
                if x < 0 || x as usize >= tests || x >= 64 || !next_ok {
                    return Err(bad("a test it does not hold"));
                }
                if k == ORACLE && y != FAIL {
                    return Err(bad("groups in a lookaround (read by the reference)"));
                }
                back[pc + 1].push((pc, x));
            }
            SAVE | MARK | GUARD => {
                if !next_ok || (k == SAVE && (x < 0 || x as usize >= ncaps)) || (k == MARK && (x < 0 || x as usize > n)) {
                    return Err(bad("a mark out of the program"));
                }
                back[pc + 1].push((pc, -1));
            }
            CHECK => {
                if !target(y) || !target(z) || x < 0 || x as usize > n {
                    return Err(bad("a check out of the program"));
                }
                for t in [y, z] {
                    if t != FAIL {
                        back[t as usize].push((pc, -1));
                    }
                }
            }
            _ => return Err(bad("an instruction it does not know")),
        }
        kind.push(k);
        a.push(x);
        b.push(y);
        c.push(z);
    }
    let mut key_at = Vec::with_capacity(n);
    let mut keys = 0usize;
    for &d in &depth {
        if d < 0 || d as usize > n {
            return Err(bad("a depth out of the program"));
        }
        key_at.push(keys);
        keys += d as usize + 1;
    }
    for &(t, _, _) in &oracle_tests {
        if t as usize >= tests || t >= 64 {
            return Err(bad("a lookaround test it does not hold"));
        }
    }
    Ok(Program { kind, a, b, c, depth, chars, ends, back, anchored, oracle_tests, width: n / 8 + 1, key_at, keys })
}

/// A pattern's programs, as the reference compiled them, and what its reads keep between texts.
#[pyclass(module = "oo_core")]
pub struct LinearProgram {
    shared: Arc<Shared>,
    kept: Arc<Mutex<Kept>>,
}

#[pymethods]
impl LinearProgram {
    #[new]
    fn new(plan: Plan) -> PyResult<Self> {
        let (format, sets, word, word_ascii, tests, looks, programs, header, tried) = plan;
        if format != FORMAT {
            return Err(PyValueError::new_err(format!(
                "linear program of format {format}: this core reads format {FORMAT}"
            )));
        }
        let (main, ncaps, boundary_reads_empty) = header;
        if ncaps < 2 || ncaps > 1 << 16 {
            return Err(PyValueError::new_err("linear program: a count of groups out of shape"));
        }
        let sorted = |ranges: &[(u32, u32)]| {
            ranges.iter().all(|&(first, last)| first <= last)
                && ranges.windows(2).all(|pair| pair[0].1 < pair[1].0)
        };
        if !sets.iter().all(|ranges| sorted(ranges)) || !sorted(&word) || !sorted(&word_ascii)
            || !tried.as_deref().map_or(true, sorted) {
            return Err(PyValueError::new_err("linear program: a set of characters out of order"));
        }
        if sets.len() > 64 || tests.len() > 64 {
            return Err(PyValueError::new_err("linear program: more than 64 sets or tests (read by the reference)"));
        }
        let mut asserts = Vec::new();
        let mut parsed_tests = Vec::new();
        for (i, &(tag, kind, multiline, ascii)) in tests.iter().enumerate() {
            if tag == 0 {
                if kind < 0 || kind > AT_NON_BOUNDARY as i64 {
                    return Err(PyValueError::new_err("linear program: an anchor it does not know"));
                }
                asserts.push(i);
                parsed_tests.push(Test::At { kind: kind as u8, multiline, ascii });
            } else {
                parsed_tests.push(Test::Look);
            }
        }
        let count = programs.len();
        let mut built = Vec::with_capacity(count);
        for plan in programs {
            built.push(program_from(plan, tests.len(), sets.len(), ncaps)?);
        }
        if main >= count {
            return Err(PyValueError::new_err("linear program: no main program"));
        }
        let mut parsed_looks = Vec::new();
        for (index, &(behind, scan)) in looks.iter().enumerate() {
            if scan >= count || scan == main {
                return Err(PyValueError::new_err("linear program: a lookaround without its own program"));
            }
            // a lookaround's body reads only the tables of the lookarounds inside it, made before its own
            if built[scan].oracle_tests.iter().any(|&(_, inner, _)| inner >= index) {
                return Err(PyValueError::new_err("linear program: a lookaround that reads a table not yet made"));
            }
            parsed_looks.push(Look { behind, scan });
        }
        for program in &built {
            for &(_, index, _) in &program.oracle_tests {
                if index >= parsed_looks.len() {
                    return Err(PyValueError::new_err("linear program: a test of a lookaround it does not hold"));
                }
            }
        }
        let shared = Shared {
            sets,
            word,
            word_ascii,
            tests: parsed_tests,
            asserts,
            looks: parsed_looks,
            programs: built,
            main,
            ncaps,
            boundary_reads_empty,
            tried,
        };
        let caches = (0..=count).map(|_| Cache::new()).collect();
        let kept = Kept {
            classes: HashMap::new(),
            signatures: HashMap::new(),
            infos: Vec::new(),
            contexts: HashMap::new(),
            caches,
        };
        Ok(LinearProgram { shared: Arc::new(shared), kept: Arc::new(Mutex::new(kept)) })
    }

    /// One text up to `end`, read for every call on it: classes, lookarounds' tables (the work spent in `work`).
    #[pyo3(signature = (text, end, most_moves, window, most_classes))]
    fn read(&self, py: Python<'_>, text: &Bound<'_, PyString>, end: usize, most_moves: usize, window: usize,
            most_classes: usize) -> PyResult<LinearRead> {
        if window == 0 {
            return Err(PyValueError::new_err("linear read: a window of zero"));
        }
        let shared = self.shared.clone();
        let kept = self.kept.clone();
        // The code points as Python holds them (one, two or four bytes each), read in place: a lone surrogate, which
        // no UTF-8 conversion takes, is the code point it is, as for re.
        // SAFETY: a ready str of CPython 3.12 or later (the only versions the engine reads). The caller holds a
        // reference to it for the whole call, and a str is never changed: its units stay where they are, and may be
        // read with the GIL released.
        let units = unsafe { text.data()? };
        let length = match &units {
            PyStringData::Ucs1(u) => u.len(),
            PyStringData::Ucs2(u) => u.len(),
            PyStringData::Ucs4(u) => u.len(),
        };
        if end > length {
            return Err(PyValueError::new_err("linear read: end past the text"));
        }
        released(py, move || {
            let mut read = Read { seen: Seen::new(0), end, most_moves, window, cls: Vec::new(),
                                  tried_at: Vec::new(), oracles: Vec::new(), alive: [None, None], work: 0 };
            let mut guard = kept.lock().map_err(|_| PyValueError::new_err("linear program: a read panicked"))?;
            {
                // The classes of the text, all made when the read begins and kept with it; the classes the pattern
                // keeps for the next read are let go once they number most_classes or more.
                let tried = shared.tried.as_ref();
                if tried.is_some() {
                    read.tried_at = buffer(end / 64 + 1, 0u64)?;
                }
                let mut cls: Vec<u16> = Vec::new();
                cls.try_reserve_exact(end).map_err(|_| {
                    PyMemoryError::new_err(format!("linear read: no memory for a buffer of {end} positions"))
                })?;
                let mut tried_at = std::mem::take(&mut read.tried_at);
                let mut work = 0u64;
                let mut classify = |at: usize, point: u32| -> PyResult<u16> {
                    if let Some(tried) = tried {
                        if within(tried, point) {
                            tried_at[at >> 6] |= 1 << (at & 63);
                        }
                    }
                    if let Some(&found) = guard.classes.get(&point) {
                        return Ok(found);
                    }
                    work += shared.sets.len() as u64 + 2;
                    let mut members = 0u64;
                    for (set_id, ranges) in shared.sets.iter().enumerate() {
                        if within(ranges, point) {
                            members |= 1 << set_id;
                        }
                    }
                    let info = (members, point == 10, within(&shared.word, point), within(&shared.word_ascii, point));
                    let number = match guard.signatures.get(&info) {
                        Some(&n) => n,
                        None => {
                            let n = guard.infos.len() as u32;
                            if n >= u16::MAX as u32 {
                                return Err(PyValueError::new_err("linear read: more than 65535 classes of characters"));
                            }
                            guard.signatures.insert(info, n);
                            guard.infos.push(info);
                            n
                        }
                    } as u16;
                    guard.classes.insert(point, number);
                    Ok(number)
                };
                match &units {
                    PyStringData::Ucs1(u) => {
                        for (at, &unit) in u.iter().take(end).enumerate() {
                            cls.push(classify(at, unit as u32)?);
                        }
                    }
                    PyStringData::Ucs2(u) => {
                        for (at, &unit) in u.iter().take(end).enumerate() {
                            cls.push(classify(at, unit as u32)?);
                        }
                    }
                    PyStringData::Ucs4(u) => {
                        for (at, &unit) in u.iter().take(end).enumerate() {
                            cls.push(classify(at, unit)?);
                        }
                    }
                }
                read.cls = cls;
                read.tried_at = tried_at;
                read.work += work;
                if guard.classes.len() >= most_classes {
                    guard.classes.clear();
                }
            }
            for index in 0..shared.looks.len() {
                let table = {
                    let mut reader = Reader { shared: &shared, kept: &mut guard, read: &mut read };
                    reader.oracle(index)?
                };
                read.oracles.push(table);
            }
            drop(guard);
            Ok(LinearRead { shared, kept, read: Mutex::new(read) })
        })
    }

    #[getter]
    fn sets(&self) -> usize {
        self.shared.sets.len()
    }
}

/// The reads of one text: its classes and tables, and the backward passes of the main program once made.
#[pyclass(module = "oo_core")]
pub struct LinearRead {
    shared: Arc<Shared>,
    kept: Arc<Mutex<Kept>>,
    read: Mutex<Read>,
}

impl LinearRead {
    fn with<T>(&self, f: impl FnOnce(&mut Reader) -> T) -> PyResult<T> {
        let mut kept = self.kept.lock().map_err(|_| PyValueError::new_err("linear program: a read panicked"))?;
        let mut read = self.read.lock().map_err(|_| PyValueError::new_err("linear read: a read panicked"))?;
        let mut reader = Reader { shared: &self.shared, kept: &mut kept, read: &mut read };
        Ok(f(&mut reader))
    }
}

#[pymethods]
impl LinearRead {
    #[getter]
    fn work(&self) -> PyResult<u64> {
        Ok(self.read.lock().map_err(|_| PyValueError::new_err("linear read: a read panicked"))?.work)
    }

    /// Make the main program's backward pass over [0, end] (all ends, or the end only), once.
    fn pass_(&self, py: Python<'_>, final_only: bool) -> PyResult<()> {
        released(py, || {
            self.with(|reader| {
                let which = final_only as usize;
                if reader.read.alive[which].is_none() {
                    let main = reader.shared.main;
                    let end = reader.read.end;
                    let alive = reader.alive(main, 0, end, final_only)?;
                    reader.read.alive[which] = Some(alive);
                }
                Ok(())
            })?
        })
    }

    /// Whether a match of the pass starts at `at`.
    fn starts(&self, final_only: bool, at: usize) -> PyResult<bool> {
        let read = self.read.lock().map_err(|_| PyValueError::new_err("linear read: a read panicked"))?;
        let alive = read.alive[final_only as usize].as_ref().ok_or_else(|| PyValueError::new_err("no pass made"))?;
        Ok(at >= alive.start && at <= alive.end && alive.starts[at - alive.start] == 1)
    }

    /// The next matches of a search from `at` (the reference's `_step`, in turn): at most `most` of them (all when
    /// 0), read with the GIL released, as (caps of each, where the search stands, the empty match it must not repeat
    /// there, whether it is done). re's search tries a start only where the read marked one (`tried_at`).
    #[pyo3(signature = (at, refuse, most))]
    fn steps(&self, py: Python<'_>, at: usize, refuse: Option<usize>, most: usize)
             -> PyResult<(Vec<Vec<Option<i64>>>, usize, Option<usize>, bool)> {
        released(py, || {
            self.with(|reader| {
                let end = reader.read.end;
                if reader.read.alive[0].is_none() {
                    return Err(PyValueError::new_err("linear read: no pass made"));
                }
                let main = reader.shared.main;
                let (mut at, mut refuse) = (at, refuse.map_or(-1i64, |r| r as i64));
                let mut found = Vec::new();
                let begin_from = |reader: &Reader, from: usize| -> Option<usize> {
                    let alive = reader.read.alive[0].as_ref().unwrap();
                    let next = |from: usize| -> Option<usize> {
                        if from > alive.end {
                            return None;
                        }
                        let k = from.saturating_sub(alive.start);
                        alive.starts[k..].iter().position(|&s| s == 1).map(|j| alive.start + k + j)
                    };
                    let mut begin = next(from);
                    if reader.shared.tried.is_some() {
                        let tried_at = &reader.read.tried_at;
                        while let Some(b) = begin {
                            if b < alive.end && tried_at[b >> 6] >> (b & 63) & 1 == 0 {
                                begin = next(b + 1);
                            } else {
                                break;
                            }
                        }
                    }
                    begin
                };
                loop {
                    if most > 0 && found.len() >= most {
                        return Ok((found, at, if refuse < 0 { None } else { Some(refuse as usize) }, false));
                    }
                    if at > end {
                        return Ok((found, at, None, true));
                    }
                    let mut begin = match begin_from(reader, at) {
                        Some(b) => b,
                        None => return Ok((found, at, None, true)),
                    };
                    let mut caps = None;
                    if begin as i64 == refuse {
                        caps = reader.pike(main, begin, false, refuse, Some(0));
                        if caps.is_none() {
                            begin = match begin_from(reader, begin + 1) {
                                Some(b) => b,
                                None => return Ok((found, at, None, true)),
                            };
                        }
                    }
                    let caps = match caps {
                        Some(caps) => caps,
                        None => reader.pike(main, begin, false, -1, Some(0)).ok_or_else(|| {
                            PyValueError::new_err("linear read: a start the backward pass found holds no match")
                        })?,
                    };
                    let (start, stop) = (caps[0], caps[1]);
                    at = stop as usize;
                    refuse = if stop == start { stop } else { -1 };
                    found.push(caps.into_iter().map(|v| if v < 0 { None } else { Some(v) }).collect());
                }
            })?
        })
    }

    /// The caps of the leftmost-first match from `start` (groups' ends, then the last group closed), or None.
    #[pyo3(signature = (start, final_only, refuse=None))]
    fn pike(&self, py: Python<'_>, start: usize, final_only: bool, refuse: Option<usize>) -> PyResult<Option<Vec<Option<i64>>>> {
        let found = released(py, || {
            self.with(|reader| {
                let main = reader.shared.main;
                let which = final_only as usize;
                if reader.read.alive[which].is_none() || start > reader.read.end {
                    return Err(PyValueError::new_err("linear read: no pass made, or a start past the end"));
                }
                Ok(reader.pike(main, start, final_only, refuse.map_or(-1, |r| r as i64), Some(which)))
            })
        })??;
        Ok(found.map(|caps| caps.into_iter().map(|v| if v < 0 { None } else { Some(v) }).collect()))
    }
}
