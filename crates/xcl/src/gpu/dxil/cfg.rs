//! Basic blocks of a decoded RDNA2 program, their edges and a reverse
//! post-order (the emission order: every block after its dominators).

use std::collections::BTreeSet;

use crate::gpu::rdna2::{self, Decoded};

/// How a block ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    /// falls through / jumps to the block (index)
    Jump(usize),
    /// conditional branch: (instruction index of the branch, taken, not taken)
    Cond(usize, usize, usize),
    /// s_endpgm or the end of the code
    End,
}

pub struct CfgBlock {
    /// byte offset of the block's first instruction
    pub pc: u64,
    /// instruction indices (the terminating branch excluded for `Cond`/`Jump` by branch)
    pub insts: Vec<usize>,
    pub term: Term,
    pub preds: Vec<usize>,
}

pub struct Cfg {
    pub insts: Vec<Decoded>,
    pub blocks: Vec<CfgBlock>,
    /// reachable blocks in reverse post-order (blocks[rpo[0]] is the entry)
    pub rpo: Vec<usize>,
    /// targets of back edges (loop headers)
    pub headers: BTreeSet<usize>,
}

fn is_end(name: &str) -> bool {
    matches!(name, "S_ENDPGM" | "S_ENDPGM_SAVED" | "S_ENDPGM_ORDERED_PS_DONE")
}

impl Cfg {
    pub fn build(code: &[u8]) -> Result<Cfg, String> {
        Self::build_with(code, None)
    }

    /// Geometry mode: the ES's `s_setpc_b64` continues at `gs_start` (the GS code) when
    /// that is past the start, else ends the program (`recomp::Mode::Geometry`).
    pub fn build_geometry(code: &[u8], gs_start: Option<u64>) -> Result<Cfg, String> {
        Self::build_with(code, Some(gs_start.filter(|&s| s > 0)))
    }

    fn build_with(code: &[u8], setpc: Option<Option<u64>>) -> Result<Cfg, String> {
        let mut insts: Vec<Decoded> = Vec::new();
        let mut off = 0usize;
        while off + 4 <= code.len() {
            let d = rdna2::decode(&code[off..], off as u64).map_err(|e| format!("+{off:#x}: cannot decode ({e:?})"))?;
            off += d.len;
            insts.push(d);
        }
        rdna2::resolve_far_jumps(&mut insts);
        let index_of = |pc: u64| insts.binary_search_by_key(&pc, |d| d.pc).ok();
        let mut is_leader = vec![false; insts.len()];
        if !is_leader.is_empty() {
            is_leader[0] = true;
        }
        let is_setpc = |d: &Decoded| setpc.is_some() && d.inst.name == "S_SETPC_B64" && d.far_target.is_none();
        let setpc_target = match setpc {
            Some(Some(t)) => Some(index_of(t).ok_or_else(|| format!("GS start {t:#x} is not the start of an instruction"))?),
            _ => None,
        };
        if let Some(t) = setpc_target {
            is_leader[t] = true;
        }
        for (i, d) in insts.iter().enumerate() {
            if is_setpc(d) {
                if i + 1 < is_leader.len() {
                    is_leader[i + 1] = true;
                }
                continue;
            }
            if let Some(t) = rdna2::branch_target(d) {
                let ti = index_of(t).ok_or_else(|| format!("+{:#x}: branch to {t:#x}, which is not the start of an instruction", d.pc))?;
                is_leader[ti] = true;
                if i + 1 < is_leader.len() {
                    is_leader[i + 1] = true;
                }
            } else if d.inst.flags & rdna2::F_TERMINATOR != 0 || is_end(d.inst.name) {
                if i + 1 < is_leader.len() {
                    is_leader[i + 1] = true;
                }
            }
        }
        let starts: Vec<usize> = (0..insts.len()).filter(|&i| is_leader[i]).collect();
        let mut block_of = vec![usize::MAX; insts.len()];
        for (b, &i) in starts.iter().enumerate() {
            block_of[i] = b;
        }
        let mut blocks = Vec::with_capacity(starts.len());
        for (b, &s) in starts.iter().enumerate() {
            let e = starts.get(b + 1).copied().unwrap_or(insts.len());
            let mut list = Vec::with_capacity(e - s);
            let mut term = if e < insts.len() { Term::Jump(b + 1) } else { Term::End };
            for i in s..e {
                let d = &insts[i];
                if is_end(d.inst.name) {
                    term = Term::End;
                    break;
                }
                if is_setpc(d) {
                    term = match setpc_target {
                        Some(t) => Term::Jump(block_of[t]),
                        None => Term::End,
                    };
                    break;
                }
                if let Some(t) = rdna2::branch_target(d) {
                    let tb = block_of[index_of(t).unwrap()];
                    let name = d.inst.name;
                    term = if name == "S_BRANCH" || d.far_target.is_some() {
                        Term::Jump(tb)
                    } else {
                        let next = if e < insts.len() { b + 1 } else { usize::MAX };
                        if next == usize::MAX {
                            return Err(format!("+{:#x}: conditional branch at the end of the code", d.pc));
                        }
                        Term::Cond(i, tb, next)
                    };
                    break;
                }
                list.push(i);
            }
            blocks.push(CfgBlock { pc: insts[s].pc, insts: list, term, preds: Vec::new() });
        }
        // a conditional branch whose both edges meet is a jump
        for b in &mut blocks {
            if let Term::Cond(_, t, f) = b.term {
                if t == f {
                    b.term = Term::Jump(t);
                }
            }
        }
        // DXIL requires reducible control flow (AMD's compiler faults on an irreducible
        // CFG; the console compiler emits loops entered in the middle): split nodes until
        // every retreating edge is a back edge to a dominator
        let (post, headers) = make_reducible(&mut blocks)?;
        single_latch(&mut blocks, &post, &headers);
        let (post, headers) = order(&blocks);
        for &b in &post {
            for t in succ_of(&blocks[b]) {
                if !blocks[t].preds.contains(&b) {
                    blocks[t].preds.push(b);
                }
            }
        }
        Ok(Cfg { insts, blocks, rpo: post, headers })
    }
}

/// Makes the CFG reducible by splitting: for an irreducible edge's innermost multi-entry
/// cycle region, the region minus its first entry (in RPO) is copied for the edges entering
/// it elsewhere. The reverse post-order and loop headers of the result; gives up past four
/// times the original block count.
fn make_reducible(blocks: &mut Vec<CfgBlock>) -> Result<(Vec<usize>, BTreeSet<usize>), String> {
    let (post, headers, _) = make_reducible_inner(blocks)?;
    Ok((post, headers))
}

fn make_reducible_inner(blocks: &mut Vec<CfgBlock>) -> Result<(Vec<usize>, BTreeSet<usize>, usize), String> {
    let limit = 4 * blocks.len() + 256;
    let mut dom_passes = 0usize;
    loop {
        let (post, headers) = order(blocks);
        if headers.is_empty() {
            return Ok((post, headers, dom_passes));
        }
        dom_passes += 1;
        let Some((_, t)) = irreducible_edge(blocks, &post) else { return Ok((post, headers, dom_passes)) };
        let n = blocks.len();
        let mut pos = vec![usize::MAX; n];
        for (i, &b) in post.iter().enumerate() {
            pos[b] = i;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &post {
            for s in succ_of(&blocks[b]) {
                preds[s].push(b);
            }
        }
        // nodes reachable from `from` within `inside`, along successors or predecessors
        let reach = |from: usize, inside: &[bool], fwd: bool| {
            let mut seen = vec![false; n];
            let mut stack = vec![from];
            seen[from] = true;
            while let Some(x) = stack.pop() {
                let succs = succ_of(&blocks[x]);
                let next: &[usize] = if fwd { succs.as_slice() } else { &preds[x] };
                for &y in next {
                    if inside[y] && !seen[y] {
                        seen[y] = true;
                        stack.push(y);
                    }
                }
            }
            seen
        };
        // shrink to the innermost cycle region around `t` with more than one entry
        let mut inside: Vec<bool> = (0..n).map(|b| pos[b] != usize::MAX).collect();
        let (region, entries) = loop {
            let (f, r) = (reach(t, &inside, true), reach(t, &inside, false));
            let region: Vec<bool> = (0..n).map(|b| f[b] && r[b]).collect();
            let entries: Vec<usize> = (0..n).filter(|&b| region[b] && (b == post[0] || preds[b].iter().any(|&p| !region[p]))).collect();
            if entries.len() != 1 {
                break (region, entries);
            }
            inside = region;
            inside[entries[0]] = false;
            if !inside[t] {
                return Err("irreducible control flow (no multi-entry region)".into());
            }
        };
        if blocks.len() >= limit || entries.is_empty() {
            return Err(format!("irreducible control flow (still after splitting to {} blocks)", blocks.len()));
        }
        let h = *entries.iter().min_by_key(|&&e| pos[e]).unwrap();
        let mut copy = vec![usize::MAX; n];
        for b in (0..n).filter(|&b| region[b] && b != h) {
            copy[b] = blocks.len();
            blocks.push(CfgBlock { pc: blocks[b].pc, insts: blocks[b].insts.clone(), term: blocks[b].term.clone(), preds: Vec::new() });
        }
        let redirect = |term: &mut Term, to: &dyn Fn(usize) -> usize| match term {
            Term::Jump(x) => *x = to(*x),
            Term::Cond(_, x, y) => {
                *x = to(*x);
                *y = to(*y)
            }
            Term::End => {}
        };
        // the copies stay among themselves except for `h` and the region's exits
        for b in (0..n).filter(|&b| copy[b] != usize::MAX) {
            redirect(&mut blocks[copy[b]].term, &|x| if copy[x] != usize::MAX { copy[x] } else { x });
        }
        // edges from outside the region into its other entries go to the copies
        for p in (0..n).filter(|&p| !region[p] && pos[p] != usize::MAX) {
            redirect(&mut blocks[p].term, &|x| if region[x] && x != h { copy[x] } else { x });
        }
    }
}

#[derive(Clone, Copy)]
struct Succs {
    buf: [usize; 2],
    len: usize,
}

impl Succs {
    fn as_slice(&self) -> &[usize] {
        &self.buf[..self.len]
    }
}

impl IntoIterator for Succs {
    type Item = usize;
    type IntoIter = std::iter::Take<std::array::IntoIter<usize, 2>>;
    fn into_iter(self) -> Self::IntoIter {
        self.buf.into_iter().take(self.len)
    }
}

/// One back edge per loop head and per block: a head with several, or one whose back edge
/// leaves a block that has a second, gets a new block that jumps to it and takes them.
/// vkd3d-proton's dxil-spirv assumes this where it handles wave shuffles and overflows its
/// stack otherwise (#1805: 77 of the title's 2,682 compute programs).
fn single_latch(blocks: &mut Vec<CfgBlock>, rpo: &[usize], headers: &BTreeSet<usize>) {
    let mut pos = vec![usize::MAX; blocks.len()];
    for (i, &b) in rpo.iter().enumerate() {
        pos[b] = i;
    }
    let back_edges = |b: &CfgBlock, p: usize| succ_of(b).into_iter().filter(|&t| pos[t] <= pos[p]).count();
    let twice: Vec<bool> = (0..blocks.len()).map(|p| pos[p] != usize::MAX && back_edges(&blocks[p], p) > 1).collect();
    for &h in headers {
        let back: Vec<usize> = rpo.iter().copied().filter(|&p| pos[p] >= pos[h] && succ_of(&blocks[p]).as_slice().contains(&h)).collect();
        if back.len() < 2 && !back.iter().any(|&p| twice[p]) {
            continue;
        }
        let latch = blocks.len();
        blocks.push(CfgBlock { pc: blocks[h].pc, insts: Vec::new(), term: Term::Jump(h), preds: Vec::new() });
        for p in back {
            match &mut blocks[p].term {
                Term::Jump(x) => *x = latch,
                Term::Cond(_, x, y) => {
                    if *x == h {
                        *x = latch
                    }
                    if *y == h {
                        *y = latch
                    }
                }
                Term::End => {}
            }
        }
    }
}

fn succ_of(b: &CfgBlock) -> Succs {
    match b.term {
        Term::Jump(t) => Succs { buf: [t, 0], len: 1 },
        Term::Cond(_, t, f) => Succs { buf: [t, f], len: 2 },
        Term::End => Succs { buf: [0, 0], len: 0 },
    }
}

/// Reachable blocks in reverse post-order from the entry (iterative DFS) and the targets of
/// retreating edges.
fn order(blocks: &[CfgBlock]) -> (Vec<usize>, BTreeSet<usize>) {
    let n = blocks.len();
    let mut state = vec![0u8; n]; // 0 new, 1 on stack, 2 done
    let mut post = Vec::with_capacity(n);
    let mut headers = BTreeSet::new();
    if n > 0 {
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(n.min(64).max(1));
        stack.push((0, 0));
        state[0] = 1;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            let succs = succ_of(&blocks[b]);
            let s = succs.as_slice();
            if *k < s.len() {
                let t = s[*k];
                *k += 1;
                match state[t] {
                    0 => {
                        state[t] = 1;
                        stack.push((t, 0));
                    }
                    1 => {
                        headers.insert(t);
                    }
                    _ => {}
                }
            } else {
                state[b] = 2;
                post.push(b);
                stack.pop();
            }
        }
    }
    post.reverse();
    (post, headers)
}

/// A retreating edge (a, t) of the reverse post-order `rpo` whose target does not dominate
/// its source (the CFG is irreducible), if any.
fn irreducible_edge(blocks: &[CfgBlock], rpo: &[usize]) -> Option<(usize, usize)> {
    let mut pos = vec![usize::MAX; blocks.len()];
    for (i, &b) in rpo.iter().enumerate() {
        pos[b] = i;
    }
    let mut preds = vec![Vec::new(); blocks.len()];
    for &b in rpo {
        for t in succ_of(&blocks[b]) {
            preds[t].push(b);
        }
    }
    // Cooper–Harvey–Kennedy dominators over the RPO
    let mut idom = vec![usize::MAX; blocks.len()];
    let entry = *rpo.first()?;
    idom[entry] = entry;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo[1..] {
            let mut nd = usize::MAX;
            for &p in preds[b].iter().filter(|&&p| idom[p] != usize::MAX) {
                nd = if nd == usize::MAX {
                    p
                } else {
                    let (mut x, mut y) = (p, nd);
                    while x != y {
                        while pos[x] > pos[y] {
                            x = idom[x];
                        }
                        while pos[y] > pos[x] {
                            y = idom[y];
                        }
                    }
                    x
                };
            }
            if idom[b] != nd {
                idom[b] = nd;
                changed = true;
            }
        }
    }
    let dominates = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        if b == entry {
            return false;
        }
        b = idom[b];
    };
    rpo.iter().find_map(|&a| succ_of(&blocks[a]).into_iter().find(|&t| pos[t] <= pos[a] && !dominates(t, a)).map(|t| (a, t)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(term: Term) -> CfgBlock {
        CfgBlock { pc: 0, insts: Vec::new(), term, preds: Vec::new() }
    }

    /// The console's loop entered in the middle (`62b584e061c05fc5`'s shape): 0 → {1 | 3},
    /// 1 → 2, cycle 2 → 3 → {2 | 4}; split into a single-entry loop.
    #[test]
    fn loop_entered_in_the_middle_is_split() {
        let mut b = vec![blk(Term::Cond(0, 3, 1)), blk(Term::Jump(2)), blk(Term::Jump(3)), blk(Term::Cond(0, 2, 4)), blk(Term::End)];
        let (rpo, headers) = order(&b);
        assert!(irreducible_edge(&b, &rpo).is_some());
        let (rpo, headers2) = make_reducible(&mut b).unwrap();
        assert!(irreducible_edge(&b, &rpo).is_none());
        assert_eq!(b.len(), 6);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers2.len(), 1);
    }

    #[test]
    fn acyclic_cfg_skips_dominator_pass_and_matches_rpo_and_preds() {
        let mut code = Vec::new();
        code.extend(rdna2::encode("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 2)], None).unwrap());
        code.extend(rdna2::encode("S_NOP", "ENC_SOPP", &[("SIMM16", 0)], None).unwrap());
        code.extend(rdna2::encode("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None).unwrap());
        code.extend(rdna2::encode("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 0)], None).unwrap());
        code.extend(rdna2::encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        code.extend(rdna2::encode("S_NOP", "ENC_SOPP", &[("SIMM16", 0)], None).unwrap());

        let mut cfg = Cfg::build(&code).unwrap();
        assert_eq!(cfg.blocks.len(), 5);
        assert_eq!(cfg.blocks[2].term, Term::Jump(3));
        assert!(cfg.headers.is_empty());
        assert_eq!(cfg.rpo, vec![0, 1, 2, 3]);
        assert_eq!(cfg.blocks[0].preds, Vec::<usize>::new());
        assert_eq!(cfg.blocks[1].preds, vec![0]);
        assert_eq!(cfg.blocks[2].preds, vec![0]);
        assert_eq!(cfg.blocks[3].preds, vec![1, 2]);
        assert_eq!(cfg.blocks[4].preds, Vec::<usize>::new());
        assert!(irreducible_edge(&cfg.blocks, &cfg.rpo).is_none());

        let (rpo, headers, dom_passes) = make_reducible_inner(&mut cfg.blocks).unwrap();
        assert_eq!(dom_passes, 0, "acyclic CFG must skip dominator pass");
        assert!(headers.is_empty());
        assert_eq!(rpo, cfg.rpo);
    }

    #[test]
    fn succ_of_is_copy_and_covers_all_terminators() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Succs>();
        for (term, want) in [
            (Term::End, &[][..]),
            (Term::Jump(5), &[5][..]),
            (Term::Cond(9, 2, 7), &[2, 7][..]),
        ] {
            let s = succ_of(&blk(term));
            assert_eq!(s.as_slice(), want);
            assert_eq!(s.into_iter().collect::<Vec<_>>(), want);
        }
    }

    #[test]
    fn empty_reducible_loop_irreducible_and_geometry_cfgs_match() {
        let empty = Cfg::build(&[]).unwrap();
        assert!(empty.insts.is_empty() && empty.blocks.is_empty() && empty.rpo.is_empty() && empty.headers.is_empty());

        let mut red = vec![blk(Term::Jump(1)), blk(Term::Cond(0, 1, 2)), blk(Term::End)];
        let (rpo, headers, dom_passes) = make_reducible_inner(&mut red).unwrap();
        assert_eq!((dom_passes, rpo, headers), (1, vec![0, 1, 2], BTreeSet::from([1])));

        let mut irred = vec![blk(Term::Cond(0, 3, 1)), blk(Term::Jump(2)), blk(Term::Jump(3)), blk(Term::Cond(0, 2, 4)), blk(Term::End)];
        let (rpo, headers, dom_passes) = make_reducible_inner(&mut irred).unwrap();
        assert!(dom_passes >= 2 && headers.len() == 1 && irreducible_edge(&irred, &rpo).is_none());

        let mut gs = Vec::new();
        gs.extend(rdna2::encode("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 0)], None).unwrap());
        gs.extend(rdna2::encode("S_NOP", "ENC_SOPP", &[("SIMM16", 0)], None).unwrap());
        gs.extend(rdna2::encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let with_gs = Cfg::build_geometry(&gs, Some(8)).unwrap();
        assert_eq!(with_gs.blocks[0].term, Term::Jump(2));
        assert_eq!((with_gs.blocks[2].pc, with_gs.rpo), (8, vec![0, 2]));
        for start in [None, Some(0)] {
            let ended = Cfg::build_geometry(&gs, start).unwrap();
            assert_eq!((&ended.blocks[0].term, &ended.rpo), (&Term::End, &vec![0]));
        }
        assert!(Cfg::build_geometry(&gs, Some(2)).err().unwrap().contains("GS start"));

        let bad_br = rdna2::encode("S_BRANCH", "ENC_SOPP", &[("SIMM16", 8)], None).unwrap();
        assert!(Cfg::build(&bad_br).err().unwrap().contains("not the start of an instruction"));
        let bad_cbr = rdna2::encode("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", (-1i16) as u16 as u64)], None).unwrap();
        assert!(Cfg::build(&bad_cbr).err().unwrap().contains("conditional branch at the end of the code"));
    }

    /// A loop with two ways back to its head (a `continue` and the loop's end): 0 → 1,
    /// 1 → {2 | 4}, 2 → {1 | 3}, 3 → 1. Both go through one latch block afterwards.
    #[test]
    fn two_back_edges_share_a_latch() {
        let mut b = vec![blk(Term::Jump(1)), blk(Term::Cond(0, 2, 4)), blk(Term::Cond(0, 1, 3)), blk(Term::Jump(1)), blk(Term::End)];
        let (rpo, headers) = order(&b);
        single_latch(&mut b, &rpo, &headers);
        let (rpo, headers) = order(&b);
        assert_eq!(b.len(), 6);
        assert_eq!(b[5].term, Term::Jump(1));
        assert_eq!(headers, BTreeSet::from([1]));
        let back: Vec<usize> = rpo.iter().copied().filter(|&p| p != 0 && succ_of(&b[p]).as_slice().contains(&1)).collect();
        assert_eq!(back, [5]);
        // a loop with one back edge is left as it is
        let mut one = vec![blk(Term::Jump(1)), blk(Term::Cond(0, 1, 2)), blk(Term::End)];
        let (rpo, headers) = order(&one);
        single_latch(&mut one, &rpo, &headers);
        assert_eq!(one.len(), 3);
    }

    /// A block that goes back to the inner loop's head or to the outer one's: 0 → 1 → 2,
    /// 2 → {3 | 4}, 3 → {2 | 1}. Each head gets a latch, so no block has two back edges.
    #[test]
    fn a_block_keeps_one_back_edge() {
        let mut b = vec![blk(Term::Jump(1)), blk(Term::Jump(2)), blk(Term::Cond(0, 3, 4)), blk(Term::Cond(0, 2, 1)), blk(Term::End)];
        let (rpo, headers) = order(&b);
        single_latch(&mut b, &rpo, &headers);
        let (rpo, headers) = order(&b);
        assert_eq!(b.len(), 7);
        assert_eq!(headers, BTreeSet::from([1, 2]));
        let mut pos = vec![0; b.len()];
        for (i, &x) in rpo.iter().enumerate() {
            pos[x] = i;
        }
        assert!(rpo.iter().all(|&p| succ_of(&b[p]).into_iter().filter(|&t| pos[t] <= pos[p]).count() <= 1));
    }
}
