//! Cheap cleanup of the entry function before it is written: unreachable
//! blocks emptied, trivial phis replaced by their single value, common
//! subexpressions merged (pure instructions along the dominator tree, wave
//! intrinsics within a block), dead instructions removed.
//!
//! The translator emits each guest instruction's per-lane expansion as is
//! (repeated lane-bit / ballot / extract sequences, operands of discarded
//! results); this pass removes the redundancy without changing what any
//! remaining instruction computes.

use std::collections::HashMap;

use super::{BinOp, Block, CastOp, FnAttrs, Module, Op, Pred, Ty, Value};

impl Op {
    /// Calls `f` on every value operand.
    pub(super) fn for_each_operand(&self, mut f: impl FnMut(Value)) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                f(*a);
                f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                f(*a);
                f(*b);
                f(*c)
            }
            Op::Call(_, args) => args.iter().copied().for_each(f),
            Op::Phi(inc) => inc.iter().for_each(|p| f(p.0)),
            Op::Gep(p, idx) => {
                f(*p);
                idx.iter().copied().for_each(f)
            }
            Op::Switch(v, _, cases) => {
                f(*v);
                cases.iter().for_each(|c| f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }

    /// Rewrites every value operand through `f`.
    pub(super) fn map_operands(&mut self, mut f: impl FnMut(Value) -> Value) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                *a = f(*a);
                *b = f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => *v = f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                *a = f(*a);
                *b = f(*b);
                *c = f(*c)
            }
            Op::Call(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
            Op::Phi(inc) => inc.iter_mut().for_each(|p| p.0 = f(p.0)),
            Op::Gep(p, idx) => {
                *p = f(*p);
                idx.iter_mut().for_each(|v| *v = f(*v))
            }
            Op::Switch(v, _, cases) => {
                *v = f(*v);
                cases.iter_mut().for_each(|c| c.0 = f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Tests can compare emitted code with the guarded-select fold disabled.
    pub(crate) static NO_SELECT_FOLD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// How an instruction may be merged with an identical one.
#[derive(PartialEq)]
enum Cse {
    No,
    /// pure: any dominating copy will do
    Dom,
    /// a wave intrinsic: same result only under the same active lanes (same block)
    Block,
}

impl Module {
    fn wave_call(&self, f: u32) -> bool {
        let n = &self.funcs[f as usize].name;
        n.starts_with("dx.op.wave") || n.starts_with("dx.op.quad")
    }

    fn is_const_global_ptr(&self, v: Value) -> bool {
        match v {
            Value::Global(g) => self.globals[g as usize].constant,
            Value::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(base, _) => self.is_const_global_ptr(*base),
                _ => false,
            },
            _ => false,
        }
    }

    fn cse_kind(&self, op: &Op) -> Cse {
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) => Cse::Dom,
            Op::Load(p, _) if self.is_const_global_ptr(*p) => Cse::Dom,
            Op::Phi(_) if cfg!(test) && std::env::var_os("XCL_SAMPLE_PHI_CSE").is_some() => Cse::Block,
            Op::Call(f, _) if self.funcs[*f as usize].attrs == FnAttrs::ReadNone => Cse::Dom,
            Op::Call(f, _) if self.wave_call(*f) && !self.funcs[*f as usize].name.starts_with("dx.op.waveIsFirstLane") => Cse::Block,
            _ => Cse::No,
        }
    }

    /// Conservative unsigned `[lo, hi]` value range of an `i32` value, up to `depth` steps deep.
    fn unsigned_range_i32(&self, v: Value, depth: u8) -> (u32, u32) {
        if let Some(c) = self.const_u64(v) {
            let k = c as u32;
            return (k, k);
        }
        if depth == 0 {
            return (0, u32::MAX);
        }
        let Value::Inst(j) = v else { return (0, u32::MAX) };
        let d = depth - 1;
        match &self.insts[j as usize].op {
            Op::Bin(BinOp::And, x, y, 0) => match self.const_u64(*x).or_else(|| self.const_u64(*y)) {
                Some(m) => (0, m as u32),
                None => (0, u32::MAX),
            },
            Op::Bin(BinOp::LShr, _, sh, 0) => match self.const_u64(*sh) {
                Some(s) if s < 32 => (0, u32::MAX >> s),
                _ => (0, u32::MAX),
            },
            Op::Cast(CastOp::ZExt, inner, _) => match self.int_width(self.value_ty(*inner)) {
                Some(iw) if iw < 32 => (0, (1u32 << iw) - 1),
                _ => (0, u32::MAX),
            },
            Op::Select(_, t, f) => {
                let (lt, ht) = self.unsigned_range_i32(*t, d);
                let (lf, hf) = self.unsigned_range_i32(*f, d);
                (lt.min(lf), ht.max(hf))
            }
            Op::Call(f, args) if args.len() == 3 && self.funcs[*f as usize].name == "dx.op.binary.i32" => {
                match self.const_u64(args[0]) {
                    Some(39) => {
                        let (la, ha) = self.unsigned_range_i32(args[1], d);
                        let (lb, hb) = self.unsigned_range_i32(args[2], d);
                        (la.max(lb), ha.max(hb))
                    }
                    Some(40) => {
                        let (la, ha) = self.unsigned_range_i32(args[1], d);
                        let (lb, hb) = self.unsigned_range_i32(args[2], d);
                        (la.min(lb), ha.min(hb))
                    }
                    _ => (0, u32::MAX),
                }
            }
            _ => (0, u32::MAX),
        }
    }

    /// Conservative signed `[lo, hi]` value range of an `i32` value, up to `depth` steps deep.
    fn signed_range_i32(&self, v: Value, depth: u8) -> (i32, i32) {
        if let Some(c) = self.const_u64(v) {
            let k = c as u32 as i32;
            return (k, k);
        }
        if depth == 0 {
            return (i32::MIN, i32::MAX);
        }
        let Value::Inst(j) = v else { return (i32::MIN, i32::MAX) };
        let d = depth - 1;
        match &self.insts[j as usize].op {
            Op::Bin(BinOp::And, x, y, 0) => match self.const_u64(*x).or_else(|| self.const_u64(*y)) {
                Some(m) if (m as u32) <= i32::MAX as u32 => (0, m as u32 as i32),
                _ => (i32::MIN, i32::MAX),
            },
            Op::Bin(BinOp::LShr, _, sh, 0) => match self.const_u64(*sh) {
                Some(s) if (1..32).contains(&s) => (0, (u32::MAX >> s) as i32),
                _ => (i32::MIN, i32::MAX),
            },
            Op::Bin(BinOp::AShr, _, sh, 0) => match self.const_u64(*sh) {
                Some(s) if (1..32).contains(&s) => (i32::MIN >> s, i32::MAX >> s),
                _ => (i32::MIN, i32::MAX),
            },
            Op::Cast(CastOp::ZExt, inner, _) => match self.int_width(self.value_ty(*inner)) {
                Some(iw) if iw < 32 => (0, ((1u32 << iw) - 1) as i32),
                _ => (i32::MIN, i32::MAX),
            },
            Op::Cast(CastOp::SExt, inner, _) => match self.int_width(self.value_ty(*inner)) {
                Some(iw) if (1..32).contains(&iw) => (-(1i32 << (iw - 1)), (1i32 << (iw - 1)) - 1),
                _ => (i32::MIN, i32::MAX),
            },
            Op::Select(_, t, f) => {
                let (lt, ht) = self.signed_range_i32(*t, d);
                let (lf, hf) = self.signed_range_i32(*f, d);
                (lt.min(lf), ht.max(hf))
            }
            Op::Call(f, args) if args.len() == 3 && self.funcs[*f as usize].name == "dx.op.binary.i32" => {
                match self.const_u64(args[0]) {
                    Some(37) => {
                        let (la, ha) = self.signed_range_i32(args[1], d);
                        let (lb, hb) = self.signed_range_i32(args[2], d);
                        (la.max(lb), ha.max(hb))
                    }
                    Some(38) => {
                        let (la, ha) = self.signed_range_i32(args[1], d);
                        let (lb, hb) = self.signed_range_i32(args[2], d);
                        (la.min(lb), ha.min(hb))
                    }
                    _ => (i32::MIN, i32::MAX),
                }
            }
            _ => (i32::MIN, i32::MAX),
        }
    }

    /// Simplifies `dx.op.binary.i32` integer min/max (`IMax=37`, `IMin=38`, `UMax=39`, `UMin=40`)
    /// via operand identity, lattice absorption, and conservative `[lo, hi]` interval bounds.
    fn simplify_int_minmax(&self, op_code: u64, a: Value, b: Value) -> Option<Value> {
        let (is_signed, is_max, dual) = match op_code {
            37 => (true, true, 38u64),
            38 => (true, false, 37u64),
            39 => (false, true, 40u64),
            40 => (false, false, 39u64),
            _ => return None,
        };
        if a == b {
            return Some(a);
        }
        // Lattice idempotence & absorption: min(min(x,y), x) = min(x,y), min(max(x,y), x) = x, etc.
        for (cand, other) in [(a, b), (b, a)] {
            if let Value::Inst(j) = cand {
                if let Op::Call(f2, a2) = &self.insts[j as usize].op {
                    if a2.len() == 3 && self.funcs[*f2 as usize].name == "dx.op.binary.i32" {
                        let inner_op = self.const_u64(a2[0]);
                        if a2[1] == other || a2[2] == other {
                            if inner_op == Some(op_code) {
                                return Some(cand);
                            }
                            if inner_op == Some(dual) {
                                return Some(other);
                            }
                        }
                    }
                }
            }
        }
        if is_signed {
            let (la, ha) = self.signed_range_i32(a, 4);
            let (lb, hb) = self.signed_range_i32(b, 4);
            if is_max {
                if la >= hb {
                    return Some(a);
                }
                if lb >= ha {
                    return Some(b);
                }
            } else {
                if ha <= lb {
                    return Some(a);
                }
                if hb <= la {
                    return Some(b);
                }
            }
        } else {
            let (la, ha) = self.unsigned_range_i32(a, 4);
            let (lb, hb) = self.unsigned_range_i32(b, 4);
            if is_max {
                if la >= hb {
                    return Some(a);
                }
                if lb >= ha {
                    return Some(b);
                }
            } else {
                if ha <= lb {
                    return Some(a);
                }
                if hb <= la {
                    return Some(b);
                }
            }
        }
        None
    }

    /// Cheap algebraic & constant simplification on already-resolved operands.
    fn simplify_op(&mut self, op: &Op, ty: Ty) -> Option<Value> {
        // Keep 64-bit instructions intact so Int64Ops / Doubles shader flags stay consistent.
        if matches!(self.types[ty.0 as usize], super::TypeDef::Int(64) | super::TypeDef::Double) {
            return None;
        }
        if let Op::Call(f, args) = op {
            if args.len() == 3 && self.funcs[*f as usize].name == "dx.op.binary.i32" {
                if let Some(op_code) = self.const_u64(args[0]) {
                    return self.simplify_int_minmax(op_code, args[1], args[2]);
                }
            }
            return None;
        }
        match *op {
            Op::Select(c, t, f) => {
                if t == f {
                    return Some(t);
                }
                match self.const_u64(c) {
                    Some(1) => Some(t),
                    Some(0) => Some(f),
                    _ => None,
                }
            }
            Op::Bin(bop, a, b, 0) => {
                let w = self.int_width(ty)?;
                // avoid dropping 64-bit ops that keep Int64Ops shader flags consistent
                if w > 32 {
                    return None;
                }
                let mask = (1u64 << w) - 1;
                let (ca, cb) = (self.const_u64(a), self.const_u64(b));
                if let (Some(x), Some(y)) = (ca, cb) {
                    let r = match bop {
                        BinOp::Add => Some(x.wrapping_add(y)),
                        BinOp::Sub => Some(x.wrapping_sub(y)),
                        BinOp::Mul => Some(x.wrapping_mul(y)),
                        BinOp::And => Some(x & y),
                        BinOp::Or => Some(x | y),
                        BinOp::Xor => Some(x ^ y),
                        BinOp::Shl if y < w as u64 => Some(x << y),
                        BinOp::LShr if y < w as u64 => Some(x >> y),
                        BinOp::AShr if y < w as u64 => Some((((x << (64 - w)) as i64) >> (64 - w + y as u32)) as u64),
                        BinOp::UDiv if y != 0 => Some(x / y),
                        BinOp::URem if y != 0 => Some(x % y),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r & mask));
                    }
                }
                match (bop, cb) {
                    (BinOp::Add | BinOp::Sub | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return Some(a),
                    (BinOp::And, Some(m)) if m == mask => return Some(a),
                    (BinOp::Or, Some(m)) if m == mask => return Some(b),
                    (BinOp::And | BinOp::Mul, Some(0)) => return Some(b),
                    (BinOp::Mul, Some(1)) => return Some(a),
                    _ => {}
                }
                match (bop, ca) {
                    (BinOp::Add | BinOp::Or | BinOp::Xor, Some(0)) => return Some(b),
                    (BinOp::And, Some(m)) if m == mask => return Some(b),
                    (BinOp::Or, Some(m)) if m == mask => return Some(a),
                    (BinOp::And | BinOp::Mul | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return Some(a),
                    (BinOp::Mul, Some(1)) => return Some(b),
                    _ => {}
                }
                if a == b {
                    match bop {
                        BinOp::And | BinOp::Or => return Some(a),
                        BinOp::Sub | BinOp::Xor => return Some(self.const_int(ty, 0)),
                        _ => {}
                    }
                }
                match bop {
                    BinOp::Xor => {
                        for (cand, other) in [(a, b), (b, a)] {
                            if let Value::Inst(j) = cand {
                                if let Op::Bin(BinOp::Xor, x, y, 0) = self.insts[j as usize].op {
                                    if other == y {
                                        return Some(x);
                                    }
                                    if other == x {
                                        return Some(y);
                                    }
                                }
                            }
                        }
                    }
                    BinOp::And => {
                        if let Some((v, m)) = cb.map(|m| (a, m)).or_else(|| ca.map(|m| (b, m))) {
                            if let Value::Inst(j) = v {
                                match self.insts[j as usize].op {
                                    Op::Bin(BinOp::LShr, _, sh, 0) => {
                                        if let Some(s) = self.const_u64(sh).filter(|&s| s < w as u64) {
                                            let known = mask >> s;
                                            if known & m == known {
                                                return Some(v);
                                            }
                                            if known & m == 0 {
                                                return Some(self.const_int(ty, 0));
                                            }
                                        }
                                    }
                                    Op::Bin(BinOp::Shl, _, sh, 0) => {
                                        if let Some(s) = self.const_u64(sh).filter(|&s| s < w as u64) {
                                            let known = (mask << s) & mask;
                                            if known & m == known {
                                                return Some(v);
                                            }
                                            if known & m == 0 {
                                                return Some(self.const_int(ty, 0));
                                            }
                                        }
                                    }
                                    Op::Bin(BinOp::And, x, y, 0) => {
                                        if let Some(c1) = self.const_u64(y).or_else(|| self.const_u64(x)) {
                                            if c1 & m == c1 {
                                                return Some(v);
                                            }
                                            if c1 & m == 0 {
                                                return Some(self.const_int(ty, 0));
                                            }
                                        }
                                    }
                                    Op::Cast(CastOp::ZExt, inner, _) if self.int_width(self.value_ty(inner)) == Some(1) => {
                                        return Some(if m & 1 != 0 { v } else { self.const_int(ty, 0) });
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    BinOp::Or => {
                        if let Some((v, m)) = cb.map(|m| (a, m)).or_else(|| ca.map(|m| (b, m))) {
                            if let Value::Inst(j) = v {
                                if let Op::Bin(BinOp::Or, x, y, 0) = self.insts[j as usize].op {
                                    if let Some(c1) = self.const_u64(y).or_else(|| self.const_u64(x)) {
                                        if c1 | m == c1 {
                                            return Some(v);
                                        }
                                        if (c1 | m) == mask {
                                            return Some(self.const_int(ty, mask));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
                None
            }
            Op::Cmp(p, a, b) => {
                let at = self.value_ty(a);
                let w = self.int_width(at)?;
                if w > 32 {
                    return None;
                }
                if let (Some(x), Some(y)) = (self.const_u64(a), self.const_u64(b)) {
                    let sx = |v: u64| ((v << (64 - w)) as i64) >> (64 - w);
                    let r = match p {
                        Pred::Eq => Some(x == y),
                        Pred::Ne => Some(x != y),
                        Pred::UGt => Some(x > y),
                        Pred::UGe => Some(x >= y),
                        Pred::ULt => Some(x < y),
                        Pred::ULe => Some(x <= y),
                        Pred::SGt => Some(sx(x) > sx(y)),
                        Pred::SGe => Some(sx(x) >= sx(y)),
                        Pred::SLt => Some(sx(x) < sx(y)),
                        Pred::SLe => Some(sx(x) <= sx(y)),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                if a == b {
                    let r = match p {
                        Pred::Eq | Pred::ULe | Pred::UGe | Pred::SLe | Pred::SGe => Some(true),
                        Pred::Ne | Pred::ULt | Pred::UGt | Pred::SLt | Pred::SGt => Some(false),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                None
            }
            Op::Cast(CastOp::BitCast, Value::Inst(j), to) => {
                if matches!(self.types[self.insts[j as usize].ty.0 as usize], super::TypeDef::Int(64) | super::TypeDef::Double) {
                    return None;
                }
                if let Op::Cast(CastOp::BitCast, inner, _) = self.insts[j as usize].op {
                    if self.value_ty(inner) == to {
                        return Some(inner);
                    }
                }
                None
            }
            Op::Cast(cop @ (CastOp::ZExt | CastOp::SExt | CastOp::Trunc), v, to) => {
                let sw = self.int_width(self.value_ty(v))?;
                let dw = self.int_width(to)?;
                if sw > 32 || dw > 32 {
                    return None;
                }
                let x = self.const_u64(v)?;
                let r = match cop {
                    CastOp::ZExt | CastOp::Trunc => x & ((1u64 << dw) - 1),
                    CastOp::SExt => ((((x << (64 - sw)) as i64) >> (64 - sw)) as u64) & ((1u64 << dw) - 1),
                    _ => unreachable!(),
                };
                Some(self.const_int(to, r))
            }
            _ => None,
        }
    }

    /// Whether an unused instruction can go. 64-bit values stay: helpers already set
    /// the Int64Ops / Doubles / DoubleExtensions flags for them.
    fn removable(&self, op: &Op, ty: Ty, wide: &[Ty]) -> bool {
        if wide.contains(&ty) {
            return false;
        }
        let mut w = false;
        op.for_each_operand(|v| w |= wide.contains(&self.value_ty(v)));
        if w {
            return false;
        }
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) | Op::Phi(_) | Op::Load(..) | Op::Alloca(..) => true,
            // only calls whose presence sets no shader flag (the flags are collected while
            // building and must match the code: no wave ops, f64 math, typed / raw loads)
            Op::Call(f, _) => {
                let d = &self.funcs[*f as usize];
                matches!(d.attrs, FnAttrs::ReadNone | FnAttrs::ReadOnly)
                    && !d.name.ends_with(".f64")
                    && ["dx.op.unary.", "dx.op.binary.", "dx.op.tertiary.", "dx.op.loadInput.", "dx.op.annotateHandle", "dx.op.createHandleFromBinding", "dx.op.cbufferLoadLegacy."]
                        .iter()
                        .any(|p| d.name.starts_with(p))
            }
            _ => false,
        }
    }

    /// Whether every non-phi operand of the reachable instructions is written before its
    /// use and has a value (what the writer checks).
    fn uses_well_formed(&self, order: &[u32]) -> bool {
        let mut at = vec![usize::MAX; self.insts.len()];
        let mut k = 0;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                at[i as usize] = k;
                k += 1;
            }
        }
        let mut ok = true;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                let op = &self.insts[i as usize].op;
                if matches!(op, Op::Phi(_)) {
                    continue;
                }
                op.for_each_operand(|v| {
                    if let Value::Inst(j) = v {
                        ok &= at[j as usize] < at[i as usize] && !matches!(self.types[self.insts[j as usize].ty.0 as usize], super::TypeDef::Void);
                    }
                });
            }
        }
        ok
    }

    /// Immediate dominators of the blocks in `order` (reverse postorder), by
    /// Cooper, Harvey and Kennedy's iteration ("A Simple, Fast Dominance Algorithm").
    pub(super) fn idoms(&self, order: &[u32]) -> Vec<Option<usize>> {
        let n = self.blocks.len();
        let mut rpo = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            rpo[b as usize] = k;
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &b in order {
            for s in self.successors(b as usize) {
                preds[s.0 as usize].push(b as usize);
            }
        }
        let mut idom: Vec<Option<usize>> = vec![None; n];
        idom[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for &b in order.iter().skip(1) {
                let b = b as usize;
                let mut new: Option<usize> = None;
                for &p in &preds[b] {
                    if idom[p].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(mut x) => {
                            let mut y = p;
                            while x != y {
                                while rpo[x] > rpo[y] {
                                    x = idom[x].unwrap();
                                }
                                while rpo[y] > rpo[x] {
                                    y = idom[y].unwrap();
                                }
                            }
                            x
                        }
                    });
                }
                if new.is_some() && idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        idom
    }

    /// Cleans up the entry function (see the module docs of `opt`). Values
    /// keep their identities; replaced ones simply lose their uses.
    ///
    /// `XCL_DXIL_NO_OPT=phi,cse,select,dce,merge` (any subset, or `all`) skips passes, for bisecting.
    pub fn optimize(&mut self) {
        let off = std::env::var("XCL_DXIL_NO_OPT").unwrap_or_default();
        let skip = |p: &str| off.split(',').any(|x| x.trim() == p || x.trim() == "all");
        let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
        // malformed IR (a use before its definition or of a void result) is left for the
        // writer to report, not cleaned away
        if !self.uses_well_formed(&order) {
            return;
        }
        // unreachable blocks: emptied (the writer drops them)
        let mut reach = vec![false; self.blocks.len()];
        for &b in &order {
            reach[b as usize] = true;
        }
        for (b, r) in reach.iter().enumerate() {
            if !r {
                self.blocks[b].clear();
            }
        }
        for &b in &order {
            for k in 0..self.blocks[b as usize].len() {
                let i = self.blocks[b as usize][k] as usize;
                if let Op::Phi(inc) = &mut self.insts[i].op {
                    inc.retain(|p| reach[p.1 .0 as usize]);
                }
            }
        }

        let mut repl: Vec<Option<Value>> = vec![None; self.insts.len()];
        let mut inst_at = vec![usize::MAX; self.insts.len()];
        let mut at = 0;
        for &b in &order {
            for &i in &self.blocks[b as usize] {
                inst_at[i as usize] = at;
                at += 1;
            }
        }

        // trivial phis (all incoming values the phi itself or one value), to a fixed point
        while !skip("phi") {
            let mut changed = false;
            for &b in &order {
                for &i in &self.blocks[b as usize] {
                    let i = i as usize;
                    if repl[i].is_some() {
                        continue;
                    }
                    let Op::Phi(inc) = &self.insts[i].op else { break };
                    let me = Value::Inst(i as u32);
                    let mut same: Option<Value> = None;
                    let mut trivial = true;
                    for &(v, _) in inc {
                        let v = resolve(&mut repl, v);
                        if v == me || Some(v) == same {
                            continue;
                        }
                        if same.is_some() {
                            trivial = false;
                            break;
                        }
                        same = Some(v);
                    }
                    // A phi may refer to a value produced on a backedge. Replacing the phi
                    // with that value would make uses near the block header precede its
                    // definition in the serialized function.
                    let available = same.is_some_and(|v| match v {
                        Value::Inst(j) => inst_at[j as usize] < inst_at[i],
                        _ => true,
                    });
                    if let (true, true, Some(v)) = (trivial, available, same) {
                        repl[i] = Some(v);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // CSE along the dominator tree (preorder walk with a scoped table)
        let idom = self.idoms(&order);
        let mut kids: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
        for &b in order.iter().skip(1) {
            if let Some(d) = idom[b as usize] {
                kids[d].push(b as usize);
            }
        }
        let mut table: HashMap<(Op, Ty), Value> = HashMap::new();
        // (block, undo log length) frames; undo log holds keys to remove / restore
        let mut undo: Vec<((Op, Ty), Option<Value>)> = Vec::new();
        enum Step {
            Enter(usize),
            Leave(usize),
        }
        let mut stack = if skip("cse") { Vec::new() } else { vec![Step::Enter(0)] };
        let mut cse_replaced = false;
        while let Some(step) = stack.pop() {
            let b = match step {
                Step::Leave(mark) => {
                    while undo.len() > mark {
                        let (k, old) = undo.pop().unwrap();
                        match old {
                            Some(v) => table.insert(k, v),
                            None => table.remove(&k),
                        };
                    }
                    continue;
                }
                Step::Enter(b) => b,
            };
            stack.push(Step::Leave(undo.len()));
            let mut local: HashMap<(Op, Ty), Value> = HashMap::new();
            for k in 0..self.blocks[b].len() {
                let i = self.blocks[b][k] as usize;
                let mut op = std::mem::replace(&mut self.insts[i].op, Op::Ret);
                op.map_operands(|v| resolve(&mut repl, v));
                if repl[i].is_none() {
                    if let Some(s) = self.simplify_op(&op, self.insts[i].ty) {
                        repl[i] = Some(s);
                        cse_replaced = true;
                    } else {
                        let kind = self.cse_kind(&op);
                        if kind != Cse::No {
                            let key = (op.clone(), self.insts[i].ty);
                            let t = if kind == Cse::Dom { &mut table } else { &mut local };
                            if let Some(&v) = t.get(&key) {
                                repl[i] = Some(v);
                                cse_replaced = true;
                            } else {
                                let old = t.insert(key.clone(), Value::Inst(i as u32));
                                if kind == Cse::Dom {
                                    undo.push((key, old));
                                }
                            }
                        } else if matches!(&op, Op::Call(f, _) if self.funcs[*f as usize].name == "dx.op.discard") {
                            // `dx.op.discard` demotes lanes mid-block, changing the active lane
                            // set for subsequent wave/quad intrinsics in the same basic block.
                            local.clear();
                        }
                    }
                }
                self.insts[i].op = op;
            }
            for &c in kids[b].iter().rev() {
                stack.push(Step::Enter(c));
            }
        }
        // second trivial-phi fixed point if CSE/simplification collapsed incoming values
        if cse_replaced && !skip("phi") {
            loop {
                let mut changed = false;
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        let i = i as usize;
                        if repl[i].is_some() {
                            continue;
                        }
                        let Op::Phi(inc) = &self.insts[i].op else { break };
                        let me = Value::Inst(i as u32);
                        let mut same: Option<Value> = None;
                        let mut trivial = true;
                        for &(v, _) in inc {
                            let v = resolve(&mut repl, v);
                            if v == me || Some(v) == same {
                                continue;
                            }
                            if same.is_some() {
                                trivial = false;
                                break;
                            }
                            same = Some(v);
                        }
                        let available = same.is_some_and(|v| match v {
                            Value::Inst(j) => inst_at[j as usize] < inst_at[i],
                            _ => true,
                        });
                        if let (true, true, Some(v)) = (trivial, available, same) {
                            repl[i] = Some(v);
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        // phis may name values defined later (back edges): resolve once more
        for &b in &order {
            for k in 0..self.blocks[b as usize].len() {
                let i = self.blocks[b as usize][k] as usize;
                self.insts[i].op.map_operands(|v| resolve(&mut repl, v));
            }
        }

        #[cfg(test)]
        let skip_select = NO_SELECT_FOLD.with(|c| c.get());
        #[cfg(not(test))]
        let skip_select = false;
        if !skip("select") && !skip_select {
            self.fold_guarded_selects(&order, &repl);
        }

        // dead instructions: live = side effects, terminators and their operands, transitively
        let (i64t, f64t) = self.existing_i64_f64();
        let wide: Vec<Ty> = i64t.into_iter().chain(f64t).collect();
        let mut live = vec![false; self.insts.len()];
        let mut work: Vec<u32> = Vec::new();
        for &b in &order {
            for &i in &self.blocks[b as usize] {
                if repl[i as usize].is_none() && (skip("dce") || !self.removable(&self.insts[i as usize].op, self.insts[i as usize].ty, &wide)) {
                    live[i as usize] = true;
                    work.push(i);
                }
            }
        }
        while let Some(i) = work.pop() {
            self.insts[i as usize].op.for_each_operand(|v| {
                if let Value::Inst(j) = v {
                    if !live[j as usize] {
                        live[j as usize] = true;
                        work.push(j);
                    }
                }
            });
        }
        for b in &mut self.blocks {
            b.retain(|&i| live[i as usize]);
        }

        if !skip("merge") && self.simplify_cfg() {
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
            inst_at.fill(usize::MAX);
            let mut at = 0;
            for &b in &order {
                for &i in &self.blocks[b as usize] {
                    inst_at[i as usize] = at;
                    at += 1;
                }
            }
            let mut repl2: Vec<Option<Value>> = vec![None; self.insts.len()];
            let mut phi_changed = false;
            while !skip("phi") {
                let mut changed = false;
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        let i = i as usize;
                        if repl2[i].is_some() {
                            continue;
                        }
                        let Op::Phi(inc) = &self.insts[i].op else { break };
                        let me = Value::Inst(i as u32);
                        let mut same: Option<Value> = None;
                        let mut trivial = true;
                        for &(v, _) in inc {
                            let v = resolve(&mut repl2, v);
                            if v == me || Some(v) == same {
                                continue;
                            }
                            if same.is_some() {
                                trivial = false;
                                break;
                            }
                            same = Some(v);
                        }
                        let available = same.is_some_and(|v| match v {
                            Value::Inst(j) => inst_at[j as usize] < inst_at[i],
                            _ => true,
                        });
                        if let (true, true, Some(v)) = (trivial, available, same) {
                            repl2[i] = Some(v);
                            changed = true;
                            phi_changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            if phi_changed {
                for &b in &order {
                    for k in 0..self.blocks[b as usize].len() {
                        let i = self.blocks[b as usize][k] as usize;
                        self.insts[i].op.map_operands(|v| resolve(&mut repl2, v));
                    }
                }
            }
            if !skip("dce") {
                live.fill(false);
                work.clear();
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        if repl2[i as usize].is_none() && !self.removable(&self.insts[i as usize].op, self.insts[i as usize].ty, &wide) {
                            live[i as usize] = true;
                            work.push(i);
                        }
                    }
                }
                while let Some(i) = work.pop() {
                    self.insts[i as usize].op.for_each_operand(|v| {
                        if let Value::Inst(j) = v {
                            if !live[j as usize] {
                                live[j as usize] = true;
                                work.push(j);
                            }
                        }
                    });
                }
                for b in &mut self.blocks {
                    b.retain(|&i| live[i as usize]);
                }
            }
            debug_assert!(self.uses_well_formed(&order));
        }
    }

    /// Whether `op` computes each lane from only that lane's operands. Operations whose
    /// evaluation observes other lanes, memory, or undefined integer division operands are
    /// deliberately excluded.
    fn lane_local(&self, op: &Op, ty: Ty) -> bool {
        match op {
            Op::Bin(BinOp::UDiv | BinOp::URem, ..) => false,
            Op::Bin(BinOp::SDiv | BinOp::SRem, ..) => self.is_float(ty),
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) => true,
            Op::Call(f, args) => {
                let d = &self.funcs[*f as usize];
                d.attrs == FnAttrs::ReadNone
                    && d.name.starts_with("dx.op.")
                    && args.first().and_then(|&a| self.const_u64(a)).is_some_and(|id| matches!(id, 6..=56 | 101 | 102 | 124..=132))
            }
            _ => false,
        }
    }

    /// Reads through `select(c, current, old)` only where the value is ultimately observed
    /// under the same `c`. This removes guest-EXEC preservation arms from chains of lane-local
    /// operations without changing inactive lanes. Phis and non-lane-local operations observe
    /// all lanes and stop the fold.
    fn fold_guarded_selects(&mut self, order: &[u32], repl: &[Option<Value>]) {
        #[derive(Clone, Copy, PartialEq)]
        enum Seen {
            Unused,
            Where(Value),
            All,
        }
        fn meet(s: &mut Seen, by: Seen) {
            *s = match (*s, by) {
                (x, Seen::Unused) => x,
                (Seen::Unused, y) => y,
                (Seen::Where(a), Seen::Where(b)) if a == b => Seen::Where(a),
                _ => Seen::All,
            }
        }

        let live = |i: u32| repl[i as usize].is_none();
        let mut seen = vec![Seen::Unused; self.insts.len()];
        for &b in order {
            for &i in &self.blocks[b as usize] {
                if let (true, Op::Phi(inc)) = (live(i), &self.insts[i as usize].op) {
                    for &(v, _) in inc {
                        if let Value::Inst(j) = v {
                            seen[j as usize] = Seen::All;
                        }
                    }
                }
            }
        }
        for &b in order.iter().rev() {
            for &i in self.blocks[b as usize].iter().rev() {
                if !live(i) {
                    continue;
                }
                let (op, ty) = (&self.insts[i as usize].op, self.insts[i as usize].ty);
                if matches!(op, Op::Phi(_)) {
                    continue;
                }
                let me = if self.lane_local(op, ty) { seen[i as usize] } else { Seen::All };
                if me == Seen::Unused {
                    continue;
                }
                let mut by = |v: Value, s: Seen| {
                    if let Value::Inst(j) = v {
                        meet(&mut seen[j as usize], s);
                    }
                };
                match *op {
                    Op::Select(c, a, b) => {
                        by(c, me);
                        by(a, Seen::Where(c));
                        by(b, me);
                    }
                    _ => op.for_each_operand(|v| by(v, me)),
                }
            }
        }
        for &b in order {
            for k in 0..self.blocks[b as usize].len() {
                let i = self.blocks[b as usize][k];
                let (op, ty) = (&self.insts[i as usize].op, self.insts[i as usize].ty);
                if !live(i) || seen[i as usize] == Seen::Unused || !self.lane_local(op, ty) {
                    continue;
                }
                let me = seen[i as usize];
                let outer_condition = if let Op::Select(c, ..) = *op { Some(c) } else { None };
                let mut op = op.clone();
                let mut pos = 0;
                op.map_operands(|mut v| {
                    let at = pos;
                    pos += 1;
                    while let Value::Inst(j) = v {
                        let Op::Select(c, t, f) = self.insts[j as usize].op else { break };
                        v = match (outer_condition, at) {
                            (Some(outer), 1) if outer == c => t,
                            (Some(outer), 2) if outer == c => f,
                            _ if me == Seen::Where(c) => t,
                            _ => break,
                        };
                    }
                    v
                });
                self.insts[i as usize].op = op;
            }
        }
    }

    /// Simplifies the CFG to a fixed point:
    /// 1. Collapses degenerate `CondBr(_, t, t)` and `Switch` cases matching the default target.
    /// 2. Bypasses empty trampoline blocks (`[Br(t)]`) whenever `t`'s phis allow it.
    /// 3. Merges single-predecessor blocks `a -> b` along unconditional branches.
    fn simplify_cfg(&mut self) -> bool {
        let mut ever_changed = false;
        let mut phi_repl: Vec<Option<Value>> = Vec::new();
        loop {
            let mut changed = false;
            let (layout, pos) = self.block_layout();
            let order: Vec<u32> = layout.into_iter().map(|b| b as u32).collect();

            // Step 0: simplify degenerate terminators (`CondBr(_, t, t)`, `Switch` cases -> default).
            for &b_u32 in &order {
                let b = b_u32 as usize;
                let Some(&term_i) = self.blocks[b].last() else { continue };
                let b_blk = Block(b_u32);
                match &mut self.insts[term_i as usize].op {
                    Op::CondBr(_, t, f) if *t == *f => {
                        let target = *t;
                        self.insts[term_i as usize].op = Op::Br(target);
                        self.drop_phi_edges(target.0 as usize, b_blk, 1);
                        changed = true;
                    }
                    Op::Switch(_, def, cases) => {
                        let def = *def;
                        let before = cases.len();
                        cases.retain(|&(_, target)| target != def);
                        let dropped = before - cases.len();
                        let became_empty = cases.is_empty();
                        if dropped > 0 {
                            self.drop_phi_edges(def.0 as usize, b_blk, dropped);
                            changed = true;
                        }
                        if became_empty {
                            self.insts[term_i as usize].op = Op::Br(def);
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }

            // Step 1: bypass empty trampoline blocks (`[Br(t)]`, `b != 0`, `t != b`).
            // `preds[x]` is maintained as a superset of live blocks whose terminator targets `x`.
            let mut preds: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
            for &b_u32 in &order {
                let b = b_u32 as usize;
                for s in self.successors(b) {
                    let s = s.0 as usize;
                    if !preds[s].contains(&b) {
                        preds[s].push(b);
                    }
                }
            }
            for &b_u32 in &order {
                let b = b_u32 as usize;
                if b == 0 || self.blocks[b].len() != 1 {
                    continue;
                }
                let Op::Br(t_blk) = self.insts[self.blocks[b][0] as usize].op else { continue };
                let t = t_blk.0 as usize;
                if t == b || self.blocks[t].is_empty() || pos[t] <= pos[b] {
                    continue;
                }
                let b_blk = Block(b_u32);
                let cand_preds = preds[b].clone();
                let target_has_phi = self.blocks[t]
                    .first()
                    .is_some_and(|&i| matches!(self.insts[i as usize].op, Op::Phi(_)));
                if target_has_phi {
                    // Bypassing a multi-edge trampoline into a phi block would duplicate phi
                    // operands for every phi in `t`; only bypass when `b` has a single incoming edge.
                    let total_edges: usize = cand_preds
                        .iter()
                        .filter(|&&p| p != b && !self.blocks[p].is_empty())
                        .map(|&p| self.count_term_edges(p, b_blk))
                        .sum();
                    if total_edges > 1 {
                        continue;
                    }
                }
                let mut all_redirected = true;
                let mut any_redirected = false;
                for p in cand_preds {
                    if p == b || self.blocks[p].is_empty() {
                        continue;
                    }
                    let edges = self.count_term_edges(p, b_blk);
                    if edges == 0 {
                        continue;
                    }
                    let p_blk = Block(p as u32);
                    if !self.phi_redirect_safe(t, b_blk, p_blk) {
                        all_redirected = false;
                        continue;
                    }
                    self.rewrite_term_target(p, b_blk, t_blk);
                    self.add_redirected_phi_edges(t, b_blk, p_blk, edges);
                    if !preds[t].contains(&p) {
                        preds[t].push(p);
                    }
                    any_redirected = true;
                    changed = true;
                }
                if any_redirected && all_redirected {
                    self.blocks[b].clear();
                    self.remove_all_phi_edges(t, b_blk);
                }
            }

            // Step 2: merge single-predecessor linear blocks (`a -> b`).
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
            let mut pred_edges = vec![0u32; self.blocks.len()];
            for &b_u32 in &order {
                for s in self.successors(b_u32 as usize) {
                    pred_edges[s.0 as usize] += 1;
                }
            }
            let mut any_phi_repl = false;
            for &a_u32 in &order {
                let a = a_u32 as usize;
                if self.blocks[a].is_empty() {
                    continue;
                }
                let a_blk = Block(a_u32);
                while let Some(&term_i) = self.blocks[a].last() {
                    let Op::Br(b_blk) = self.insts[term_i as usize].op else { break };
                    let b = b_blk.0 as usize;
                    if b == 0 || b == a || self.blocks[b].is_empty() || pred_edges[b] != 1 || self.preserve_block_boundary[b] {
                        break;
                    }
                    // Verify every leading phi in `b` has an incoming value from `a` before mutating.
                    let mut phis_ok = true;
                    for &inst in &self.blocks[b] {
                        let Op::Phi(inc) = &self.insts[inst as usize].op else { break };
                        if !inc.iter().any(|&(_, bb)| bb == a_blk) {
                            phis_ok = false;
                            break;
                        }
                    }
                    if !phis_ok {
                        break;
                    }
                    // Pop `Br(b)` from `a` and move `b`'s non-phi instructions into `a`.
                    self.blocks[a].pop();
                    let b_insts = std::mem::take(&mut self.blocks[b]);
                    for inst in b_insts {
                        if let Op::Phi(inc) = &self.insts[inst as usize].op {
                            let v = inc.iter().find(|&&(_, bb)| bb == a_blk).unwrap().0;
                            if phi_repl.is_empty() {
                                phi_repl.resize(self.insts.len(), None);
                            }
                            phi_repl[inst as usize] = Some(v);
                            any_phi_repl = true;
                        } else {
                            self.def_block[inst as usize] = a_blk;
                            self.blocks[a].push(inst);
                        }
                    }
                    // Update successor phis from `b` to `a`.
                    let mut seen_s = Vec::new();
                    for s_blk in self.successors(a) {
                        if seen_s.contains(&s_blk) {
                            continue;
                        }
                        seen_s.push(s_blk);
                        let s = s_blk.0 as usize;
                        for k in 0..self.blocks[s].len() {
                            let i = self.blocks[s][k] as usize;
                            let Op::Phi(inc) = &mut self.insts[i].op else { break };
                            for (_, bb) in inc.iter_mut() {
                                if *bb == b_blk {
                                    *bb = a_blk;
                                }
                            }
                        }
                    }
                    pred_edges[b] = 0;
                    changed = true;
                }
            }
            if any_phi_repl {
                let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
                for &b_u32 in &order {
                    let b = b_u32 as usize;
                    for k in 0..self.blocks[b].len() {
                        let i = self.blocks[b][k] as usize;
                        self.insts[i].op.map_operands(|v| resolve(&mut phi_repl, v));
                    }
                }
                phi_repl.fill(None);
            }

            if !changed {
                break;
            }
            ever_changed = true;
        }
        ever_changed
    }

    fn count_term_edges(&self, b: usize, target: Block) -> usize {
        let Some(&term_i) = self.blocks[b].last() else { return 0 };
        match &self.insts[term_i as usize].op {
            Op::Br(t) => usize::from(*t == target),
            Op::CondBr(_, t, f) => usize::from(*t == target) + usize::from(*f == target),
            Op::Switch(_, d, cases) => usize::from(*d == target) + cases.iter().filter(|&&(_, c)| c == target).count(),
            _ => 0,
        }
    }

    fn rewrite_term_target(&mut self, b: usize, from: Block, to: Block) {
        let Some(&term_i) = self.blocks[b].last() else { return };
        match &mut self.insts[term_i as usize].op {
            Op::Br(t) => {
                if *t == from {
                    *t = to;
                }
            }
            Op::CondBr(_, t, f) => {
                if *t == from {
                    *t = to;
                }
                if *f == from {
                    *f = to;
                }
            }
            Op::Switch(_, d, cases) => {
                if *d == from {
                    *d = to;
                }
                for (_, c) in cases.iter_mut() {
                    if *c == from {
                        *c = to;
                    }
                }
            }
            _ => {}
        }
    }

    fn phi_redirect_safe(&self, target: usize, via: Block, pred: Block) -> bool {
        for &i in &self.blocks[target] {
            let Op::Phi(inc) = &self.insts[i as usize].op else { break };
            let Some(&(v_via, _)) = inc.iter().find(|&&(_, bb)| bb == via) else {
                return false;
            };
            if inc.iter().any(|&(v_p, bb)| bb == pred && v_p != v_via) {
                return false;
            }
        }
        true
    }

    fn add_redirected_phi_edges(&mut self, target: usize, via: Block, pred: Block, count: usize) {
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            let v_via = inc.iter().find(|&&(_, bb)| bb == via).unwrap().0;
            for _ in 0..count {
                inc.push((v_via, pred));
            }
        }
    }

    fn remove_all_phi_edges(&mut self, target: usize, from: Block) {
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            inc.retain(|&(_, bb)| bb != from);
        }
    }

    fn drop_phi_edges(&mut self, target: usize, from: Block, count: usize) {
        if count == 0 {
            return;
        }
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            let mut rem = count;
            inc.retain(|&(_, bb)| {
                if rem > 0 && bb == from {
                    rem -= 1;
                    false
                } else {
                    true
                }
            });
        }
    }
}

/// The value `v` stands for after replacements (path-compressing the chain).
fn resolve(repl: &mut [Option<Value>], v: Value) -> Value {
    let mut r = v;
    while let Value::Inst(i) = r {
        match repl[i as usize] {
            Some(n) => r = n,
            None => break,
        }
    }
    let mut c = v;
    while let Value::Inst(i) = c {
        match repl[i as usize] {
            Some(n) if n != r => {
                repl[i as usize] = Some(r);
                c = n;
            }
            _ => break,
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_calls(m: &Module, target_fn: u32) -> usize {
        m.blocks
            .iter()
            .flat_map(|b| b.iter())
            .filter(|&&i| matches!(m.insts[i as usize].op, Op::Call(f, _) if f == target_fn))
            .count()
    }

    #[test]
    fn block_cse_invalidates_wave_and_quad_calls_across_discard() {
        let build = |op_name: &str, with_discard: bool| -> (Module, u32) {
            let mut m = Module::new();
            m.define_entry("main");
            let b = m.entry_block();
            m.set_block(b);

            let void = m.void();
            let i1 = m.i1();
            let i32t = m.i32();

            let wave_ty = m.func_ty(i32t, &[i32t, i1]);
            let wave_fn = m.declare(op_name, wave_ty, FnAttrs::NoUnwind);
            let discard_ty = m.func_ty(void, &[i32t, i1]);
            let discard_fn = m.declare("dx.op.discard", discard_ty, FnAttrs::NoUnwind);
            let sink_ty = m.func_ty(void, &[i32t, i32t]);
            let sink_fn = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

            let c_op = m.i32c(116);
            let c_true = m.i1c(true);
            let c_discard_op = m.i32c(82);

            let first = m.call(wave_fn, &[c_op, c_true]);
            if with_discard {
                m.call(discard_fn, &[c_discard_op, c_true]);
            }
            let second = m.call(wave_fn, &[c_op, c_true]);
            m.call(sink_fn, &[first, second]);
            m.ret();
            m.optimize();
            (m, wave_fn)
        };

        // Without `dx.op.discard`, identical wave/quad calls in the same block still CSE to 1.
        let (m_no_discard, f_no_discard) = build("dx.op.waveActiveBallot", false);
        assert_eq!(count_calls(&m_no_discard, f_no_discard), 1);

        // Across `dx.op.discard`, active lanes change so both calls must be preserved.
        for op_name in ["dx.op.waveActiveBallot", "dx.op.quadReadLaneAt"] {
            let (m_with_discard, f_with_discard) = build(op_name, true);
            assert_eq!(
                count_calls(&m_with_discard, f_with_discard),
                2,
                "{op_name} must not be CSE-merged across dx.op.discard"
            );
        }
    }

    fn reachable_blocks(m: &Module) -> usize {
        m.block_layout().0.len()
    }

    #[test]
    fn simplify_cfg_collapses_trampoline_chain_and_merges_linear_blocks() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let t1 = m.new_block();
        let t2 = m.new_block();
        let mid = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let c7 = m.i32c(7);
        m.br(t1);

        m.set_block(t1);
        m.br(t2);

        m.set_block(t2);
        m.br(mid);

        m.set_block(mid);
        let p = m.phi(i32t);
        m.add_incoming(p, c7, t2);
        let c3 = m.i32c(3);
        let sum = m.bin(BinOp::Add, p, c3);
        m.br(exit);

        m.set_block(exit);
        m.call(sink, &[sum]);
        m.ret();

        m.optimize();
        assert_eq!(reachable_blocks(&m), 1, "entire linear trampoline chain should collapse into entry");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        assert_eq!(count_calls(&m, sink), 1);
    }

    #[test]
    fn simplify_cfg_keeps_preserved_block_boundary() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let guarded = m.new_block();
        let exit = m.new_block();
        m.preserve_block_boundary(guarded);

        let void = m.void();
        let sink_ty = m.func_ty(void, &[]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        m.br(guarded);
        m.set_block(guarded);
        m.call(sink, &[]);
        m.br(exit);
        m.set_block(exit);
        m.ret();

        m.optimize();
        assert_eq!(reachable_blocks(&m), 2, "preserved boundary should survive linear block merging");
    }

    #[test]
    fn simplify_cfg_preserves_phi_conflict_trampoline() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let tramp = m.new_block();
        let join = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let z = m.i32c(0);
        let cond = m.cmp(Pred::Eq, lane, z);
        let v1 = m.i32c(1);
        let v2 = m.i32c(2);
        m.cond_br(cond, tramp, join);

        m.set_block(tramp);
        m.br(join);

        m.set_block(join);
        let p = m.phi(i32t);
        m.add_incoming(p, v1, entry);
        m.add_incoming(p, v2, tramp);
        m.call(sink, &[p]);
        m.ret();

        m.optimize();
        // `tramp` cannot be bypassed because `entry` already feeds a different value into `join`'s phi.
        assert_eq!(reachable_blocks(&m), 3);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }

    #[test]
    fn simplify_cfg_collapses_same_target_cond_br_and_switch() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let mid = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let z = m.i32c(0);
        let c1 = m.i32c(1);
        let cond = m.cmp(Pred::Eq, lane, z);
        let v = m.i32c(42);
        m.cond_br(cond, mid, mid);

        m.set_block(mid);
        let p = m.phi(i32t);
        m.add_incoming(p, v, entry);
        m.add_incoming(p, v, entry);
        m.switch(lane, exit, &[(z, exit), (c1, exit)]);

        m.set_block(exit);
        let p2 = m.phi(i32t);
        m.add_incoming(p2, p, mid);
        m.add_incoming(p2, p, mid);
        m.add_incoming(p2, p, mid);
        m.call(sink, &[p2]);
        m.ret();

        m.optimize();
        assert_eq!(reachable_blocks(&m), 1, "degenerate CondBr and Switch should collapse and merge");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        assert_eq!(count_calls(&m, sink), 1);
    }

    #[test]
    fn simplify_cfg_merges_loop_latch_and_rewrites_header_phi() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let header = m.new_block();
        let latch = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let z = m.i32c(0);
        let one = m.i32c(1);
        let limit = m.i32c(10);
        m.br(header);

        m.set_block(header);
        let iv = m.phi(i32t);
        m.add_incoming(iv, z, entry);
        m.br(latch);

        m.set_block(latch);
        let next = m.bin(BinOp::Add, iv, one);
        m.add_incoming(iv, next, latch);
        let cont = m.cmp(Pred::ULt, next, limit);
        m.cond_br(cont, header, exit);

        m.set_block(exit);
        m.call(sink, &[next]);
        m.ret();

        m.optimize();
        // `latch` merges into `header`, leaving `entry`, `header` (self-loop), `exit`.
        assert_eq!(reachable_blocks(&m), 3);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        // Header phi's backedge predecessor must now be `header` itself.
        let Value::Inst(iv_i) = iv else { unreachable!() };
        let Op::Phi(inc) = &m.insts[iv_i as usize].op else { panic!("expected phi") };
        assert!(inc.iter().any(|&(_, b)| b == header), "header phi backedge should point to header after latch merge");
        assert!(!inc.iter().any(|&(_, b)| b == latch), "merged latch must not remain in header phi");
    }

    #[test]
    fn trivial_loop_phi_keeps_backedge_definition_after_header_uses() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let header = m.new_block();
        let latch = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let one = m.i32c(1);
        let limit = m.i32c(10);
        m.br(header);

        m.set_block(header);
        let iv = m.phi(i32t);
        m.add_incoming(iv, iv, entry);
        m.br(latch);

        m.set_block(latch);
        let next = m.bin(BinOp::Add, iv, one);
        m.add_incoming(iv, next, latch);
        let cont = m.cmp(Pred::ULt, next, limit);
        m.cond_br(cont, header, exit);

        m.set_block(exit);
        m.call(sink, &[next]);
        m.ret();

        m.optimize();
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        let Value::Inst(iv_i) = iv else { unreachable!() };
        assert!(m.blocks.iter().any(|b| b.contains(&iv_i)), "the backedge value cannot replace the header phi");
    }

    #[test]
    fn simplify_cfg_does_not_inflate_multi_pred_phi_trampoline() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let left = m.new_block();
        let right = m.new_block();
        let other = m.new_block();
        let tramp = m.new_block();
        let join = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let c0 = m.i32c(0);
        let c1 = m.i32c(1);
        let v1 = m.i32c(11);
        let v2 = m.i32c(22);
        m.switch(lane, other, &[(c0, left), (c1, right)]);

        m.set_block(left);
        m.call(sink, &[c0]);
        m.br(tramp);

        m.set_block(right);
        m.call(sink, &[c1]);
        m.br(tramp);

        m.set_block(other);
        m.br(join);

        m.set_block(tramp);
        m.br(join);

        m.set_block(join);
        let p = m.phi(i32t);
        m.add_incoming(p, v1, tramp);
        m.add_incoming(p, v2, other);
        m.call(sink, &[p]);
        m.ret();

        m.optimize();
        // `other` (1 pred -> phi) IS bypassed into `entry -> join`, while `tramp` (2 preds -> phi) stays
        // so `join`'s phi stays at 2 entries instead of growing to 3.
        let Value::Inst(p_i) = p else { unreachable!() };
        let Op::Phi(inc) = &m.insts[p_i as usize].op else { panic!("expected phi") };
        assert_eq!(inc.len(), 2);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }

    #[test]
    fn simplify_op_folds_bitmasks_double_xor_and_all_ones_or() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        m.set_block(entry);

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        let x = m.call(src, &[]);
        let c0 = m.i32c(0);
        let c2 = m.i32c(2);
        let c24 = m.i32c(24);
        let all = m.i32c(u32::MAX);
        let mff = m.i32c(0xFF);
        let mhi = m.i32c(0xFFFF_0000);
        let mlo = m.i32c(0x0000_FFFF);
        let align4 = m.i32c(!3u32);
        let low2 = m.i32c(3);
        let k = m.i32c(0xA5A5_5A5A);

        // 1. All-ones `or` absorption + `ashr 0, x -> 0`
        let or_all = m.bin(BinOp::Or, x, all);
        let ashr_z = m.bin(BinOp::AShr, c0, x);
        // 2. Double `xor` cancellation
        let x1 = m.bin(BinOp::Xor, x, k);
        let x2 = m.bin(BinOp::Xor, x1, k);
        // 3. Redundant and disjoint masks on `lshr`
        let shr = m.bin(BinOp::LShr, x, c24);
        let shr_keep = m.bin(BinOp::And, shr, mff);
        let shr_zero = m.bin(BinOp::And, shr, mhi);
        // 4. Redundant and disjoint masks on `shl`
        let shl = m.bin(BinOp::Shl, x, c2);
        let shl_keep = m.bin(BinOp::And, shl, align4);
        let shl_zero = m.bin(BinOp::And, shl, low2);
        // 5. Nested `and` / `or` masks
        let a1 = m.bin(BinOp::And, x, mff);
        let a_keep = m.bin(BinOp::And, a1, mlo);
        let a_zero = m.bin(BinOp::And, a1, mhi);
        let o1 = m.bin(BinOp::Or, x, mlo);
        let o_keep = m.bin(BinOp::Or, o1, mff);
        let o_all = m.bin(BinOp::Or, o1, mhi);
        // 6. Mask on `zext(i1)`
        let cond = m.cmp(Pred::Eq, x, c0);
        let z = m.cast(CastOp::ZExt, cond, i32t);
        let z_keep = m.bin(BinOp::And, z, mff);
        let z_zero = m.bin(BinOp::And, z, mhi);

        for v in [or_all, ashr_z, x2, shr_keep, shr_zero, shl_keep, shl_zero, a_keep, a_zero, o_keep, o_all, z_keep, z_zero] {
            m.call(sink, &[v]);
        }
        m.ret();
        m.optimize();

        let args: Vec<Value> = m.blocks[entry.0 as usize]
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(f, a) if *f == sink => Some(a[0]),
                _ => None,
            })
            .collect();
        assert_eq!(args.len(), 13);
        assert_eq!(m.const_u64(args[0]), Some(u32::MAX as u64), "or(x, ~0) -> ~0");
        assert_eq!(m.const_u64(args[1]), Some(0), "ashr(0, x) -> 0");
        assert_eq!(args[2], x, "xor(xor(x, k), k) -> x");
        assert_eq!(args[3], shr, "and(lshr(x, 24), 0xFF) -> lshr(x, 24)");
        assert_eq!(m.const_u64(args[4]), Some(0), "and(lshr(x, 24), 0xFFFF0000) -> 0");
        assert_eq!(args[5], shl, "and(shl(x, 2), !3) -> shl(x, 2)");
        assert_eq!(m.const_u64(args[6]), Some(0), "and(shl(x, 2), 3) -> 0");
        assert_eq!(args[7], a1, "and(and(x, 0xFF), 0xFFFF) -> and(x, 0xFF)");
        assert_eq!(m.const_u64(args[8]), Some(0), "and(and(x, 0xFF), 0xFFFF0000) -> 0");
        assert_eq!(args[9], o1, "or(or(x, 0xFFFF), 0xFF) -> or(x, 0xFFFF)");
        assert_eq!(m.const_u64(args[10]), Some(u32::MAX as u64), "or(or(x, 0xFFFF), 0xFFFF0000) -> ~0");
        assert_eq!(args[11], z, "and(zext(i1), 0xFF) -> zext(i1)");
        assert_eq!(m.const_u64(args[12]), Some(0), "and(zext(i1), 0xFFFF0000) -> 0");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }
    #[test]
    fn simplify_cfg_keeps_an_empty_block_on_a_back_edge() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let outer = m.new_block();
        let inner = m.new_block();
        let body = m.new_block();
        let latch = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let z = m.i32c(0);
        let one = m.i32c(1);
        m.br(outer);

        m.set_block(outer);
        m.call(sink, &[z]);
        m.br(inner);

        m.set_block(inner);
        let go = m.cmp(Pred::Eq, lane, z);
        m.cond_br(go, body, exit);

        m.set_block(body);
        m.call(sink, &[lane]);
        let again = m.cmp(Pred::Eq, lane, one);
        m.cond_br(again, inner, latch);

        m.set_block(latch);
        m.br(outer);

        m.set_block(exit);
        m.ret();

        m.optimize();
        assert!(m.successors(body.0 as usize).contains(&latch));
    }

    #[test]
    fn guarded_selects_fold_only_for_same_condition_lane_local_uses() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        m.set_block(entry);

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        let x = m.call(src, &[]);
        let (one, three, five, sentinel) = (m.i32c(1), m.i32c(3), m.i32c(5), m.i32c(0xA5A5_5A5A));
        let exec = m.cmp(Pred::ULt, x, five);
        let old = m.bin(BinOp::Xor, x, sentinel);
        let current = m.bin(BinOp::Add, x, one);
        let saved = m.select(exec, current, old);
        let product = m.bin(BinOp::Mul, saved, three);
        let guarded = m.select(exec, product, saved);

        // A second preserved value is read by an all-lane observer and must remain intact.
        let must_stay = m.select(exec, current, x);
        let all_lane_use = m.bin(BinOp::Sub, must_stay, one);
        m.call(sink, &[guarded]);
        m.call(sink, &[all_lane_use]);
        m.ret();
        m.optimize();

        let Value::Inst(guarded_i) = guarded else { panic!() };
        let Value::Inst(product_i) = product else { panic!() };
        assert!(matches!(m.insts[guarded_i as usize].op, Op::Select(c, t, f) if c == exec && t == product && f == old));
        assert!(matches!(m.insts[product_i as usize].op, Op::Bin(BinOp::Mul, a, b, 0) if (a, b) == (current, three) || (a, b) == (three, current)));
        let body = &m.blocks[entry.0 as usize];
        let present = |v: Value| matches!(v, Value::Inst(i) if body.contains(&i));
        assert!(!present(saved), "the EXEC-preservation select should have no uses");
        assert!(present(must_stay), "an all-lane use must retain its preservation select");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }
}

#[cfg(test)]
mod int_minmax_tests {
    use super::*;
    use crate::gpu::dxil::{
        kernel::{self, KernelOptions},
        validate,
    };
    use crate::gpu::rdna2::encode;
    use crate::gpu::recomp::runner;

    #[test]
    fn simplify_int_minmax_folds_bounds_absorption_and_clamps() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let i32t = m.i32();
        let void = m.void();
        let src_ty = m.func_ty(i32t, &[]);
        let sink_ty = m.func_ty(void, &[i32t]);
        let bin_ty = m.func_ty(i32t, &[i32t, i32t, i32t]);
        let src = m.declare("src", src_ty, FnAttrs::NoUnwind);
        let sink = m.declare("sink", sink_ty, FnAttrs::NoUnwind);
        let bin_fn = m.declare("dx.op.binary.i32", bin_ty, FnAttrs::ReadNone);
        m.set_block(entry);

        let x = m.call(src, &[]);
        let y = m.call(src, &[]);
        let imax = m.i32c(37);
        let imin = m.i32c(38);
        let umax = m.i32c(39);
        let umin = m.i32c(40);
        let c0 = m.i32c(0);
        let c16 = m.i32c(16);
        let c255 = m.i32c(255);
        let c100 = m.i32c(100);
        let cneg = m.i32c((-10i32) as u32);
        let cu_max = m.i32c(u32::MAX);
        let ci_min = m.i32c(i32::MIN as u32);
        let ci_max = m.i32c(i32::MAX as u32);
        let mff = m.i32c(0xFF);

        // 1. Constant folding & identity/absorbing bounds.
        let k_umin = m.call(bin_fn, &[umin, c255, c100]);
        let k_imax = m.call(bin_fn, &[imax, cneg, c100]);
        let umin_zero = m.call(bin_fn, &[umin, x, c0]);
        let umax_zero = m.call(bin_fn, &[umax, x, c0]);
        let umin_all = m.call(bin_fn, &[umin, x, cu_max]);
        let imin_min = m.call(bin_fn, &[imin, x, ci_min]);
        let imax_min = m.call(bin_fn, &[imax, x, ci_min]);
        let imin_max = m.call(bin_fn, &[imin, x, ci_max]);

        // 2. Lattice idempotence & absorption.
        let mxy = m.call(bin_fn, &[umin, x, y]);
        let idemp = m.call(bin_fn, &[umin, mxy, x]);
        let absorb = m.call(bin_fn, &[umax, mxy, x]);

        // 3. Range-based elision on masked byte (0..=255) and ashr16 (-32768..=32767).
        let byte = m.bin(BinOp::And, x, mff);
        let byte_imax0 = m.call(bin_fn, &[imax, byte, c0]);
        let byte_umin255 = m.call(bin_fn, &[umin, byte, c255]);
        let s16 = m.bin(BinOp::AShr, x, c16);
        let c_i16_lo = m.i32c((-32768i32) as u32);
        let c_i16_hi = m.i32c(32767);
        let s16_lo = m.call(bin_fn, &[imax, s16, c_i16_lo]);
        let s16_hi = m.call(bin_fn, &[imin, s16_lo, c_i16_hi]);

        // 4. Full 4-op med3_i32(x, 0, 255) collapses to 2 ops: imin(imax(x, 0), 255).
        let p = m.call(bin_fn, &[imin, x, c0]);
        let q = m.call(bin_fn, &[imax, x, c0]);
        let r = m.call(bin_fn, &[imin, q, c255]);
        let med = m.call(bin_fn, &[imax, p, r]);

        // 5. Soundness check: signed mask with bit 31 set MUST NOT fold imax(x & 0x8000_00FF, 0).
        let m_sign = m.i32c(0x8000_00FF);
        let signed_bits = m.bin(BinOp::And, x, m_sign);
        let must_stay = m.call(bin_fn, &[imax, signed_bits, c0]);

        for v in [
            k_umin,
            k_imax,
            umin_zero,
            umax_zero,
            umin_all,
            imin_min,
            imax_min,
            imin_max,
            idemp,
            absorb,
            byte_imax0,
            byte_umin255,
            s16_hi,
            med,
            must_stay,
        ] {
            m.call(sink, &[v]);
        }
        m.ret();
        m.optimize();

        let args: Vec<Value> = m.blocks[entry.0 as usize]
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(f, a) if *f == sink => Some(a[0]),
                _ => None,
            })
            .collect();
        assert_eq!(args.len(), 15);
        assert_eq!(m.const_u64(args[0]), Some(100), "umin(255, 100) -> 100");
        assert_eq!(m.const_u64(args[1]), Some(100), "imax(-10, 100) -> 100");
        assert_eq!(m.const_u64(args[2]), Some(0), "umin(x, 0) -> 0");
        assert_eq!(args[3], x, "umax(x, 0) -> x");
        assert_eq!(args[4], x, "umin(x, u32::MAX) -> x");
        assert_eq!(m.const_u64(args[5]), Some(i32::MIN as u32 as u64), "imin(x, i32::MIN) -> i32::MIN");
        assert_eq!(args[6], x, "imax(x, i32::MIN) -> x");
        assert_eq!(args[7], x, "imin(x, i32::MAX) -> x");
        assert_eq!(args[8], mxy, "umin(umin(x, y), x) -> umin(x, y)");
        assert_eq!(args[9], x, "umax(umin(x, y), x) -> x");
        assert_eq!(args[10], byte, "imax(x & 0xFF, 0) -> x & 0xFF");
        assert_eq!(args[11], byte, "umin(x & 0xFF, 255) -> x & 0xFF");
        assert_eq!(args[12], s16, "clamp(ashr(x, 16), -32768, 32767) -> ashr(x, 16)");
        assert_eq!(args[13], r, "med3_i32(x, 0, 255) collapses to imin(imax(x, 0), 255)");
        assert_eq!(args[14], must_stay, "imax(x & 0x800000FF, 0) must not fold");
    }

    #[test]
    fn med3_via_sgpr_bounds_collapses_and_matches_gpu() {
        // Real RDNA2 pattern: upper bound loaded into an SGPR (becomes constant only after mem2reg),
        // then fed into v_med3_i32 / v_med3_u32 in both bound orders.
        let mut code = Vec::new();
        code.extend(encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 255)], Some(255)).unwrap());
        code.extend(encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 128)], None).unwrap()); // 0
        // v2 = med3_i32(v0, s3=0, s2=255) -> 2 ops (down from 4)
        code.extend(encode("V_MED3_I32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 3), ("SRC2", 2)], None).unwrap());
        // v3 = med3_i32(v0, s2=255, s3=0) -> 2 ops (reversed bounds, down from 4)
        code.extend(encode("V_MED3_I32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 256), ("SRC1", 2), ("SRC2", 3)], None).unwrap());
        // v4 = med3_u32(v0, s3=0, s2=255) -> 1 op (umin(v0, 255), down from 4)
        code.extend(encode("V_MED3_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", 256), ("SRC1", 3), ("SRC2", 2)], None).unwrap());
        // v5 = med3_u32(v0, s3=0, v1) -> 1 op (umin(v0, v1), down from 4)
        code.extend(encode("V_MED3_U32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 256), ("SRC1", 3), ("SRC2", 257)], None).unwrap());
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

        let ko = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
        let dxil = kernel::translate(&code, ko).expect("translate sgpr med3");
        validate::validate(&dxil).expect("validate sgpr med3");
        let dis = validate::disassemble(&dxil).expect("disasm sgpr med3");
        let calls = dis.matches("call i32 @dx.op.binary.i32(").count();
        assert_eq!(calls, 6, "expected 6 binary.i32 calls (2+2+1+1, down from 16), got {calls}:\n{dis}");

        if !matches!(runner::wave32_supported(), Ok(true)) {
            runner::skip_test("no wave32");
            return;
        }
        let v0s: [u32; 10] = [0, 1, 128, 255, 256, 1000, (-1i32) as u32, (-255i32) as u32, i32::MIN as u32, i32::MAX as u32];
        let v1s: [u32; 10] = [10, 0, 200, 100, 300, 50, u32::MAX, 42, 0x8000_0001, 500];
        let mut st = runner::KernelState::default();
        for l in 0..v0s.len() {
            st.vgpr[l][0] = v0s[l];
            st.vgpr[l][1] = v1s[l];
        }
        let r = runner::run(&dxil, &st, &[]).expect("GPU run sgpr med3");
        for l in 0..v0s.len() {
            let (a, b) = (v0s[l], v1s[l]);
            let want_clamp_i = (a as i32).clamp(0, 255) as u32;
            assert_eq!(r.vgpr[l][2], want_clamp_i, "med3_i32(v0, 0, 255) lane {l}");
            assert_eq!(r.vgpr[l][3], want_clamp_i, "med3_i32(v0, 255, 0) lane {l}");
            assert_eq!(r.vgpr[l][4], a.min(255), "med3_u32(v0, 0, 255) lane {l}");
            assert_eq!(r.vgpr[l][5], a.min(b), "med3_u32(v0, 0, v1) lane {l}");
        }
    }

}
