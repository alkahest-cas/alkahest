/// E-graph based simplifier using egglog.
///
/// Enabled only when the `egraph` feature is active.  Falls back to the
/// rule-based engine otherwise; see `simplify_egraph` for the stable
/// public entry point that is always available.
///
/// # Encoding strategy
///
/// Alkahest uses n-ary `Add`/`Mul`, but egglog works with fixed-arity
/// constructors.  We left-fold n-ary sums/products into binary trees for
/// submission, then *flatten* the extracted binary tree back to n-ary on
/// the way out (see `parse_egglog_term`).  Commutativity is handled at
/// construction time (children sorted by ExprId); associativity is not
/// added as a rule to avoid AC explosion — the phased schedule plus the
/// flattening round-trip is sufficient for practical inputs.
///
/// # Schedule (RW-2)
///
/// The iteration counts and node/iteration limits are taken from
/// [`EgraphConfig`], allowing callers to trade completeness for bounded
/// run time on large inputs.
///
/// # Opaque atoms
///
/// The egglog datatype below can only express integer literals, variables,
/// `Add`/`Mul`/`Pow`, five known unary functions, and an uninterpreted unary
/// `Fn`.  *Everything else* — rationals, floats, out-of-`i64` integers,
/// n-ary/nullary `Func`, `Piecewise`, `Predicate`, `Forall`, `Exists`,
/// `RootSum`, `BigO` — is replaced by a freshly generated **opaque atom**
/// (`(Var "a<k>")`) that no arithmetic rule can match, with a side table
/// mapping the atom back to the original [`ExprId`].
///
/// Two properties matter and are both guaranteed by keying the table on
/// `ExprId` (which the pool interns structurally):
///
/// * the *same* subterm always gets the *same* atom, so structural sharing
///   and cancellation still work; and
/// * *different* subterms never share an atom.
///
/// Symbols go through the same table rather than being written out under
/// their own names.  That means every `Var` string egglog ever sees was
/// generated here, so a user symbol literally named `a0` cannot collide
/// with an opaque atom, and extraction restores the original symbol node —
/// including its [`Domain`](crate::kernel::Domain) and commutativity flag —
/// instead of re-interning a lookalike.
#[cfg(feature = "egraph")]
mod backend {
    use crate::kernel::{ExprData, ExprId, ExprPool};
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // 1. Serialise ExprId → egglog expression string (binary left-fold)
    // -----------------------------------------------------------------------

    /// Prefix for generated atom names.  Never collides with user symbols
    /// because user symbol names are not emitted at all — see the module
    /// docs.
    const ATOM_PREFIX: &str = "a";

    /// The five unary functions the egglog datatype models directly.
    fn known_unary(name: &str) -> Option<&'static str> {
        match name {
            "sin" => Some("Sin"),
            "cos" => Some("Cos"),
            "exp" => Some("Exp"),
            "log" => Some("Log"),
            "sqrt" => Some("Sqrt"),
            _ => None,
        }
    }

    /// Can `name` be embedded verbatim in an egglog string literal?
    ///
    /// Restricting to `[A-Za-z0-9_]+` keeps quotes, backslashes and
    /// whitespace out of the generated program; anything else falls back to
    /// an opaque atom rather than producing a malformed term.
    fn is_plain_ident(name: &str) -> bool {
        !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    /// One node of the input, classified by how the egglog datatype
    /// represents it.
    enum Node {
        Num(i64),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        /// One of the datatype's built-in unary constructors.
        Known(&'static str, ExprId),
        /// Uninterpreted unary function: `(Fn "name" arg)`.
        Fn(String, ExprId),
        /// Not expressible in the datatype — becomes an opaque atom.
        Atom,
    }

    fn classify(pool: &ExprPool, expr: ExprId) -> Node {
        pool.with(expr, |data| match data {
            // Integers outside i64 cannot be represented by `(Num i64)`.
            // Clamping would silently change the value, so they become
            // opaque atoms instead.
            ExprData::Integer(n) => match n.0.to_i64() {
                Some(v) => Node::Num(v),
                None => Node::Atom,
            },
            ExprData::Add(args) => Node::Add(args.clone()),
            ExprData::Mul(args) => Node::Mul(args.clone()),
            ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
            ExprData::Func { name, args } if args.len() == 1 => match known_unary(name) {
                Some(ctor) => Node::Known(ctor, args[0]),
                None if is_plain_ident(name) => Node::Fn(name.clone(), args[0]),
                None => Node::Atom,
            },
            // Symbols (see module docs), rationals, floats, and every
            // structured node the datatype cannot express.
            _ => Node::Atom,
        })
    }

    /// Whether `expr` encodes to an egglog constructor with `Expr` children
    /// (as opposed to a `Num`/`Var` leaf).  Mirrors [`classify`] without
    /// cloning argument vectors.
    fn is_interior(pool: &ExprPool, expr: ExprId) -> bool {
        pool.with(expr, |data| match data {
            ExprData::Add(_) | ExprData::Mul(_) | ExprData::Pow { .. } => true,
            ExprData::Func { name, args } if args.len() == 1 => {
                known_unary(name).is_some() || is_plain_ident(name)
            }
            _ => false,
        })
    }

    /// Whether some interior node of `expr` is reachable along two different
    /// parent edges — i.e. whether writing `expr` out as a tree would repeat
    /// work.  Linear in the number of distinct nodes.
    fn has_shared_interior(pool: &ExprPool, root: ExprId) -> bool {
        let mut seen = std::collections::HashSet::new();
        seen.insert(root);
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            let children: Vec<ExprId> = match classify(pool, id) {
                Node::Add(args) | Node::Mul(args) => args,
                Node::Pow(b, e) => vec![b, e],
                Node::Known(_, a) | Node::Fn(_, a) => vec![a],
                Node::Num(_) | Node::Atom => continue,
            };
            for c in children {
                if is_interior(pool, c) {
                    if !seen.insert(c) {
                        return true;
                    }
                    stack.push(c);
                }
            }
        }
        false
    }

    /// Serialiser state: the atom table built while walking the expression.
    pub(super) struct Encoder<'p> {
        pool: &'p ExprPool,
        /// Subterm → generated atom name (dedupes repeated occurrences).
        atom_of: HashMap<ExprId, String>,
        /// Generated atom name → subterm (used on the way out).
        expr_of: HashMap<String, ExprId>,
    }

    /// An expression serialised for egglog: top-level `let` bindings (empty
    /// for a tree-shaped input) followed by the root term.
    pub(super) struct Encoded {
        pub(super) lets: String,
        pub(super) root: String,
    }

    impl<'p> Encoder<'p> {
        pub(super) fn new(pool: &'p ExprPool) -> Self {
            Encoder {
                pool,
                atom_of: HashMap::new(),
                expr_of: HashMap::new(),
            }
        }

        /// The atom-name → subterm table, for decoding extracted terms.
        pub(super) fn atoms(&self) -> &HashMap<String, ExprId> {
            &self.expr_of
        }

        /// Write the opaque atom standing for `expr`, allocating it on first
        /// use so equal subterms share one atom.
        fn write_atom(&mut self, expr: ExprId, out: &mut String) {
            let next = self.atom_of.len();
            let expr_of = &mut self.expr_of;
            let name = self.atom_of.entry(expr).or_insert_with(|| {
                let name = format!("{ATOM_PREFIX}{next}");
                expr_of.insert(name.clone(), expr);
                name
            });
            out.push_str("(Var \"");
            out.push_str(name);
            out.push_str("\")");
        }

        /// Serialise `expr`, sized by its *DAG*, not its tree.
        ///
        /// A tree-shaped input (no interior node reached twice) is written
        /// out as one term, byte-for-byte what the tree encoder always
        /// produced. When some interior node is shared, the tree can be
        /// exponentially larger than the DAG — `e_{k+1} = sin(e_k)·cos(e_k)`
        /// doubles per level — so instead *every* distinct node is bound once
        /// by a top-level `(let __tK …)` and referred to by name.
        ///
        /// Binding every node, rather than only the shared ones, is what keeps
        /// the resulting e-graph *identical* to the one the tree text builds:
        /// the lets are emitted in the tree's post-order of first occurrence,
        /// so egglog inserts the same e-nodes in the same order, allocates the
        /// same e-class ids, and so breaks extraction-cost ties the same way.
        /// Opaque atoms are numbered in the same left-to-right order too.
        /// Globals are `:unextractable` in egglog, so the bindings never show
        /// up in the extracted term.
        pub(super) fn encode(&mut self, expr: ExprId) -> Encoded {
            let mut root = String::new();
            let mut lets = String::new();
            if has_shared_interior(self.pool, expr) {
                let mut names = HashMap::new();
                self.write_dag_body(expr, &mut root, &mut names, &mut lets);
            } else {
                self.write_tree(expr, &mut root);
            }
            Encoded { lets, root }
        }

        /// Write `expr` as a single egglog term (tree-shaped input).
        ///
        /// An n-ary `Add`/`Mul` becomes a binary left-fold — `(Add (Add a b)
        /// c)` for `[a, b, c]` — written into the one buffer, so a wide sum
        /// costs linear rather than quadratic time.
        fn write_tree(&mut self, expr: ExprId, out: &mut String) {
            use std::fmt::Write as _;
            match classify(self.pool, expr) {
                Node::Num(n) => {
                    let _ = write!(out, "(Num {n})");
                }
                Node::Atom => self.write_atom(expr, out),
                // Binary left-fold; the decoder flattens this back to n-ary.
                Node::Add(args) => {
                    let (&first, rest) = args.split_first().expect(
                        "Add node must have at least one argument — ExprPool invariant violated",
                    );
                    out.push_str(&"(Add ".repeat(rest.len()));
                    self.write_tree(first, out);
                    for &id in rest {
                        out.push(' ');
                        self.write_tree(id, out);
                        out.push(')');
                    }
                }
                Node::Mul(args) => {
                    let (&first, rest) = args.split_first().expect(
                        "Mul node must have at least one argument — ExprPool invariant violated",
                    );
                    out.push_str(&"(Mul ".repeat(rest.len()));
                    self.write_tree(first, out);
                    for &id in rest {
                        out.push(' ');
                        self.write_tree(id, out);
                        out.push(')');
                    }
                }
                Node::Pow(base, exp) => {
                    out.push_str("(Pow ");
                    self.write_tree(base, out);
                    out.push(' ');
                    self.write_tree(exp, out);
                    out.push(')');
                }
                Node::Known(ctor, arg) => {
                    out.push('(');
                    out.push_str(ctor);
                    out.push(' ');
                    self.write_tree(arg, out);
                    out.push(')');
                }
                Node::Fn(name, arg) => {
                    let _ = write!(out, "(Fn \"{name}\" ");
                    self.write_tree(arg, out);
                    out.push(')');
                }
            }
        }

        /// Bind `expr` (and, first, everything below it) to a `let` global,
        /// returning its name; memoised on `ExprId`.
        fn bind_dag(
            &mut self,
            expr: ExprId,
            names: &mut HashMap<ExprId, String>,
            lets: &mut String,
        ) -> String {
            if let Some(name) = names.get(&expr) {
                return name.clone();
            }
            let mut body = String::new();
            self.write_dag_body(expr, &mut body, names, lets);
            let name = format!("__t{}", names.len());
            lets.push_str("(let ");
            lets.push_str(&name);
            lets.push(' ');
            lets.push_str(&body);
            lets.push_str(")\n");
            names.insert(expr, name.clone());
            name
        }

        /// Write the constructor for `expr` with every child replaced by the
        /// name of its binding. Children are bound left to right first, which
        /// is the tree's post-order.
        fn write_dag_body(
            &mut self,
            expr: ExprId,
            out: &mut String,
            names: &mut HashMap<ExprId, String>,
            lets: &mut String,
        ) {
            use std::fmt::Write as _;
            let fold = |ctor: &str, kids: &[String], out: &mut String| {
                let (first, rest) = kids.split_first().unwrap_or_else(|| {
                    panic!(
                        "{ctor} node must have at least one argument — ExprPool invariant violated"
                    )
                });
                for _ in rest {
                    out.push('(');
                    out.push_str(ctor);
                    out.push(' ');
                }
                out.push_str(first);
                for k in rest {
                    out.push(' ');
                    out.push_str(k);
                    out.push(')');
                }
            };
            match classify(self.pool, expr) {
                Node::Num(n) => {
                    let _ = write!(out, "(Num {n})");
                }
                Node::Atom => self.write_atom(expr, out),
                Node::Add(args) => {
                    let kids: Vec<String> = args
                        .iter()
                        .map(|&a| self.bind_dag(a, names, lets))
                        .collect();
                    fold("Add", &kids, out);
                }
                Node::Mul(args) => {
                    let kids: Vec<String> = args
                        .iter()
                        .map(|&a| self.bind_dag(a, names, lets))
                        .collect();
                    fold("Mul", &kids, out);
                }
                Node::Pow(base, exp) => {
                    let b = self.bind_dag(base, names, lets);
                    let e = self.bind_dag(exp, names, lets);
                    let _ = write!(out, "(Pow {b} {e})");
                }
                Node::Known(ctor, arg) => {
                    let a = self.bind_dag(arg, names, lets);
                    let _ = write!(out, "({ctor} {a})");
                }
                Node::Fn(name, arg) => {
                    let a = self.bind_dag(arg, names, lets);
                    let _ = write!(out, "(Fn \"{name}\" {a})");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 2. Build the complete egglog program  (RW-2: uses EgraphConfig)
    // -----------------------------------------------------------------------

    /// Count unique nodes in the expression DAG.
    ///
    /// Used to enforce `EgraphConfig::node_limit` before handing the expression
    /// to egglog, preventing OOM on pathological inputs.
    fn count_dag_nodes(expr: ExprId, pool: &ExprPool) -> usize {
        let mut visited = std::collections::HashSet::new();
        count_dag_nodes_rec(expr, pool, &mut visited);
        visited.len()
    }

    fn count_dag_nodes_rec(
        expr: ExprId,
        pool: &ExprPool,
        visited: &mut std::collections::HashSet<ExprId>,
    ) {
        if !visited.insert(expr) {
            return;
        }
        match pool.get(expr) {
            ExprData::Add(args) | ExprData::Mul(args) => {
                for &a in &args {
                    count_dag_nodes_rec(a, pool, visited);
                }
            }
            ExprData::Pow { base, exp } => {
                count_dag_nodes_rec(base, pool, visited);
                count_dag_nodes_rec(exp, pool, visited);
            }
            ExprData::Func { args, .. } => {
                for &a in &args {
                    count_dag_nodes_rec(a, pool, visited);
                }
            }
            ExprData::Piecewise { branches, default } => {
                for (cond, val) in &branches {
                    count_dag_nodes_rec(*cond, pool, visited);
                    count_dag_nodes_rec(*val, pool, visited);
                }
                count_dag_nodes_rec(default, pool, visited);
            }
            ExprData::Predicate { args, .. } => {
                for a in args {
                    count_dag_nodes_rec(a, pool, visited);
                }
            }
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                count_dag_nodes_rec(var, pool, visited);
                count_dag_nodes_rec(body, pool, visited);
            }
            ExprData::BigO(arg) => {
                count_dag_nodes_rec(arg, pool, visited);
            }
            ExprData::RootSum { poly, var, body } => {
                count_dag_nodes_rec(poly, pool, visited);
                count_dag_nodes_rec(var, pool, visited);
                count_dag_nodes_rec(body, pool, visited);
            }
            // Leaf nodes
            ExprData::Integer(_)
            | ExprData::Rational(_)
            | ExprData::Float(_)
            | ExprData::Symbol { .. } => {}
        }
    }

    /// Whether `expr` has a subterm `b^k` with a negative literal exponent `k`
    /// and a base `b` that is *provably* zero — i.e. a division by zero.
    ///
    /// The e-graph cannot be trusted with such a term. Its `shrink` ruleset
    /// contains both `(Mul ?x (Num 0)) → (Num 0)` and
    /// `(Mul ?x (Pow ?x (Num -1))) → (Num 1)`, and on `0 · 0⁻¹` *both* fire —
    /// so `(Num 0)` and `(Num 1)` are unioned into one e-class and the
    /// extractor picks whichever is cheaper. That is not a wrong rewrite that
    /// could be patched away: an e-graph has no way to carry the side
    /// condition `?x ≠ 0` that makes cancellation valid, and the union it
    /// performs is `0 = 1`, which poisons every other e-class in the run.
    ///
    /// So the whole call is handed to the rule engine instead, which now
    /// leaves the undefined product alone. This mirrors the existing
    /// non-commutative bail-out immediately above it.
    ///
    /// The zero test is the three-valued
    /// [`crate::matrix::zero_test::zero_status`], so `(x - x)⁻¹` is caught as
    /// well as the literal `0⁻¹`; only a *proven* zero bails out, leaving the
    /// documented `b · b⁻¹ → 1` convention intact for every base that is not.
    /// Its cost is irrelevant here — this runs once per `simplify_egraph`
    /// call, against building and saturating an entire egglog program — and
    /// literal and symbol bases are settled without calling it at all.
    fn has_provably_zero_denominator(expr: ExprId, pool: &ExprPool) -> bool {
        let mut visited = std::collections::HashSet::new();
        count_dag_nodes_rec(expr, pool, &mut visited);
        visited
            .into_iter()
            .any(|node| is_zero_denominator(node, pool))
    }

    fn is_zero_denominator(node: ExprId, pool: &ExprPool) -> bool {
        let Some((base, exp)) = pool.with(node, |d| match d {
            ExprData::Pow { base, exp } => Some((*base, *exp)),
            _ => None,
        }) else {
            return false;
        };
        let negative_exponent = pool.with(exp, |d| match d {
            ExprData::Integer(n) => n.0 < 0,
            ExprData::Rational(r) => r.0 < 0,
            ExprData::Float(f) => f.inner < 0.0,
            _ => false,
        });
        if !negative_exponent {
            return false;
        }
        // Settle the easy bases without the general zero test.
        enum Quick {
            Zero,
            NonZero,
            Ask,
        }
        let quick = pool.with(base, |d| match d {
            ExprData::Integer(n) => {
                if n.0 == 0 {
                    Quick::Zero
                } else {
                    Quick::NonZero
                }
            }
            ExprData::Rational(_) | ExprData::Symbol { .. } => Quick::NonZero,
            ExprData::Float(f) => {
                if f.inner == 0.0 {
                    Quick::Zero
                } else {
                    Quick::NonZero
                }
            }
            _ => Quick::Ask,
        });
        match quick {
            Quick::Zero => true,
            Quick::NonZero => false,
            Quick::Ask => crate::matrix::zero_test::zero_status(pool, base).is_proven_zero(),
        }
    }

    /// Build the egglog program for `enc` as `(prelude, schedule)`.
    ///
    /// The *prelude* — options, datatype, rulesets — depends on `config`
    /// alone, never on the expression, so it is parsed once and cached (see
    /// [`prelude_egraph`]). The *schedule* carries the expression's bindings,
    /// the phased `run`s and the final `extract`. Their concatenation is the
    /// whole program this module has always run.
    fn egglog_program(
        lets: &str,
        expr_str: &str,
        config: &super::EgraphConfig,
    ) -> (String, String) {
        // node_limit is enforced as a pre-saturation DAG-size check in
        // simplify_egraph_impl; egglog 0.4 does not expose a per-run node cap.
        let node_limit_line = String::new();
        let iter_limit_line = config
            .iter_limit
            .map(|n| format!("(set-option iteration_limit {n})\n"))
            .unwrap_or_default();

        let si = config.shrink_iters;
        let ei = config.explore_iters;
        let ci = config.const_fold_iters;

        // Conditionally include trig / log-exp rules based on config flags.
        let trig_rs = if config.disjoint_schedule {
            "explore-trig"
        } else {
            "explore"
        };
        let _log_rs = if config.disjoint_schedule {
            "explore-log"
        } else {
            "explore"
        };
        let trig_rules = if config.include_trig_rules {
            // Both Mul form (sin(x)*sin(x)) and Pow form (sin(x)^2) are matched
            // so the identity fires regardless of how the square is represented.
            format!(
                "(rewrite (Add (Mul (Sin ?x) (Sin ?x)) (Mul (Cos ?x) (Cos ?x))) (Num 1) :ruleset {trig_rs})\n\
                 (rewrite (Add (Mul (Cos ?x) (Cos ?x)) (Mul (Sin ?x) (Sin ?x))) (Num 1) :ruleset {trig_rs})\n\
                 (rewrite (Add (Pow (Sin ?x) (Num 2)) (Pow (Cos ?x) (Num 2))) (Num 1) :ruleset {trig_rs})\n\
                 (rewrite (Add (Pow (Cos ?x) (Num 2)) (Pow (Sin ?x) (Num 2))) (Num 1) :ruleset {trig_rs})",
                trig_rs = trig_rs,
            )
        } else {
            String::new()
        };

        let log_exp_rules = if config.include_log_exp_rules {
            // Intentionally empty: egglog cannot check symbol domains, and
            // `log(exp(z))→z` / `exp(log(z))→z` are unsound over ℂ without
            // that check. Use `simplify_log_exp` (real-gated) or Assumptions.
            String::new()
        } else {
            String::new()
        };

        let (rules_block, schedule) = if config.disjoint_schedule {
            let shrink = format!(
                r#"
; ── match-disjoint shrink groups (distinct LHS root symbols) ────────────────
(ruleset shrink-add)
(rewrite (Add ?x (Num 0)) ?x :ruleset shrink-add)
(rewrite (Add (Num 0) ?x) ?x :ruleset shrink-add)
(rewrite (Add ?x (Mul (Num -1) ?x)) (Num 0) :ruleset shrink-add)
(rewrite (Add (Mul (Num -1) ?x) ?x) (Num 0) :ruleset shrink-add)

(ruleset shrink-mul)
(rewrite (Mul ?x (Num 1)) ?x :ruleset shrink-mul)
(rewrite (Mul (Num 1) ?x) ?x :ruleset shrink-mul)
(rewrite (Mul ?x (Num 0)) (Num 0) :ruleset shrink-mul)
(rewrite (Mul (Num 0) ?x) (Num 0) :ruleset shrink-mul)
(rewrite (Mul ?x (Pow ?x (Num -1))) (Num 1) :ruleset shrink-mul)
(rewrite (Mul (Pow ?x (Num -1)) ?x) (Num 1) :ruleset shrink-mul)

(ruleset shrink-pow)
(rewrite (Pow ?x (Num 1)) ?x :ruleset shrink-pow)
(rewrite (Pow ?x (Num 0)) (Num 1) :ruleset shrink-pow)

(ruleset explore-trig)
{trig_rules}

(ruleset explore-log)
{log_exp_rules}

(ruleset explore-mul)
(rewrite (Mul (Num -1) (Mul (Num -1) ?x)) ?x :ruleset explore-mul)
"#,
                trig_rules = trig_rules,
                log_exp_rules = log_exp_rules,
            );
            let explore_runs = {
                let mut s = String::new();
                if config.include_trig_rules {
                    s.push_str(&format!("(run explore-trig {ei})\n"));
                }
                if config.include_log_exp_rules {
                    s.push_str(&format!("(run explore-log {ei})\n"));
                }
                s.push_str(&format!("(run explore-mul {ei})\n"));
                s
            };
            let schedule = format!(
                r#"{lets}(let __expr {expr})
(run shrink-add {si})
(run shrink-mul {si})
(run shrink-pow {si})
(run const-fold {ci})
{explore_runs}(run shrink-add {si})
(run shrink-mul {si})
(run shrink-pow {si})
(run const-fold {ci})
(extract __expr)
"#,
                explore_runs = explore_runs,
                lets = lets,
                expr = expr_str,
                si = si,
                ci = ci,
            );
            (shrink, schedule)
        } else {
            let shrink = format!(
                r#"
; ── shrink ruleset: identity / absorption / cancellation ─────────────────────
(ruleset shrink)
(rewrite (Add ?x (Num 0)) ?x :ruleset shrink)
(rewrite (Add (Num 0) ?x) ?x :ruleset shrink)
(rewrite (Mul ?x (Num 1)) ?x :ruleset shrink)
(rewrite (Mul (Num 1) ?x) ?x :ruleset shrink)
(rewrite (Mul ?x (Num 0)) (Num 0) :ruleset shrink)
(rewrite (Mul (Num 0) ?x) (Num 0) :ruleset shrink)
(rewrite (Pow ?x (Num 1)) ?x :ruleset shrink)
(rewrite (Pow ?x (Num 0)) (Num 1) :ruleset shrink)
(rewrite (Add ?x (Mul (Num -1) ?x)) (Num 0) :ruleset shrink)
(rewrite (Add (Mul (Num -1) ?x) ?x) (Num 0) :ruleset shrink)
(rewrite (Mul ?x (Pow ?x (Num -1))) (Num 1) :ruleset shrink)
(rewrite (Mul (Pow ?x (Num -1)) ?x) (Num 1) :ruleset shrink)

; ── explore ruleset: trig and log/exp identities (default: both enabled) ──────
(ruleset explore)
{trig_rules}
{log_exp_rules}
(rewrite (Mul (Num -1) (Mul (Num -1) ?x)) ?x :ruleset explore)
"#,
                trig_rules = trig_rules,
                log_exp_rules = log_exp_rules,
            );
            let schedule = format!(
                r#"{lets}(let __expr {expr})
(run shrink {si})
(run const-fold {ci})
(run explore {ei})
(run shrink {si})
(run const-fold {ci})
(extract __expr)
"#,
                lets = lets,
                expr = expr_str,
                si = si,
                ei = ei,
                ci = ci,
            );
            (shrink, schedule)
        };

        let prelude = format!(
            r#"
{node_limit_line}{iter_limit_line}(datatype Expr
  (Num i64)
  (Var String)
  (Add Expr Expr)
  (Mul Expr Expr)
  (Pow Expr Expr)
  (Sin Expr)
  (Cos Expr)
  (Exp Expr)
  (Log Expr)
  (Sqrt Expr)
  (Fn String Expr))
{rules_block}
; ── constant folding ──────────────────────────────────────────────────────────
(ruleset const-fold)
(rule ((= e (Add (Num ?a) (Num ?b))))
      ((union e (Num (+ ?a ?b))))
      :ruleset const-fold)
(rule ((= e (Mul (Num ?a) (Num ?b))))
      ((union e (Num (* ?a ?b))))
      :ruleset const-fold)
; Integer Pow folding is done in Rust after extraction: egglog's i64 `^` is XOR.

; ── phased schedule ───────────────────────────────────────────────────────────
"#,
            node_limit_line = node_limit_line,
            iter_limit_line = iter_limit_line,
            rules_block = rules_block,
        );
        (prelude, schedule)
    }

    /// Upper bound on distinct cached preludes per thread. A prelude is keyed
    /// by its full text, so this only fills up if a caller cycles through
    /// many `iter_limit` values; the cache is then simply rebuilt.
    const PRELUDE_CACHE_CAP: usize = 16;

    thread_local! {
        /// Prelude text → an e-graph that has run exactly that prelude.
        static PRELUDE_CACHE: std::cell::RefCell<HashMap<String, egglog::EGraph>> =
            std::cell::RefCell::new(HashMap::new());
    }

    /// A fresh e-graph that has already run `prelude`.
    ///
    /// Parsing and type-checking the datatype and a few dozen rules is most
    /// of the cost of a small `simplify_egraph` call, and it depends only on
    /// the configuration. So the prelude is run once per thread and the
    /// resulting e-graph *cloned* for each call.
    ///
    /// The cache is keyed by the prelude text itself, so every
    /// configuration field that reaches the rules or options is part of the
    /// key by construction — there is no separate key to fall out of sync.
    /// `egglog::EGraph: Clone` copies its tables, union-find, rulesets and
    /// symbol generator by value; the only shared parts are `Arc`'d sorts
    /// and primitives, and the sorts this datatype uses (`i64`, `String`,
    /// the `Expr` eq-sort) carry no mutable state. The cached graph has not
    /// run any rule, holds no e-nodes and has flushed its messages, so a
    /// clone is indistinguishable from a graph that has just parsed the
    /// prelude afresh. `None` if egglog rejects the prelude (the caller then
    /// falls back to the input, as it always did on a failed program).
    fn prelude_egraph(prelude: &str) -> Option<egglog::EGraph> {
        PRELUDE_CACHE.with(|cache| {
            if let Some(eg) = cache.borrow().get(prelude) {
                return Some(eg.clone());
            }
            let mut eg = egglog::EGraph::default();
            eg.parse_and_run_program(None, prelude).ok()?;
            let mut cache = cache.borrow_mut();
            if cache.len() >= PRELUDE_CACHE_CAP {
                cache.clear();
            }
            cache.insert(prelude.to_owned(), eg.clone());
            Some(eg)
        })
    }

    /// Run `schedule` on top of `prelude` and return the extracted term
    /// decoded back to an [`ExprId`].
    ///
    /// Messages are switched off before the schedule runs: egglog would
    /// otherwise render the extracted term with `TermDag::to_string`, which
    /// is tree-sized. The same extraction is read from the extract report as
    /// a hash-consed [`egglog::TermDag`] and decoded node by node instead.
    fn run_and_extract(
        prelude: &str,
        schedule: &str,
        pool: &ExprPool,
        atoms: &HashMap<String, ExprId>,
    ) -> Option<ExprId> {
        let mut egraph = prelude_egraph(prelude)?;
        egraph.disable_messages();
        egraph.parse_and_run_program(None, schedule).ok()?;
        match egraph.get_extract_report() {
            Some(egglog::ExtractReport::Best { termdag, term, .. }) => {
                let root = termdag.lookup(term);
                decode_term(termdag, root, pool, atoms, &mut HashMap::new())
            }
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // 3. Parse egglog output back to ExprId  (RW-1: flatten binary → n-ary)
    // -----------------------------------------------------------------------

    /// Collect all top-level Add children, recursively flattening nested Adds.
    fn flatten_add_args(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        match pool.get(expr) {
            ExprData::Add(args) => args
                .iter()
                .flat_map(|&a| flatten_add_args(a, pool))
                .collect(),
            _ => vec![expr],
        }
    }

    /// Collect all top-level Mul children, recursively flattening nested Muls.
    fn flatten_mul_args(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        match pool.get(expr) {
            ExprData::Mul(args) => args
                .iter()
                .flat_map(|&a| flatten_mul_args(a, pool))
                .collect(),
            _ => vec![expr],
        }
    }

    /// Decode an extracted egglog term back to an [`ExprId`].
    ///
    /// The term lives in a hash-consed [`egglog::TermDag`], so shared
    /// subterms are decoded once each through `memo` — linear in the
    /// extracted DAG, where re-parsing its printed (tree-shaped) text was
    /// exponential in the depth of sharing. Binary `Add`/`Mul` are flattened
    /// back to n-ary on the way out (RW-1).
    ///
    /// `atoms` is the atom-name → subterm table produced by [`Encoder`].
    /// Every `Var` egglog can emit was generated by the encoder, so a miss
    /// means the output is not something we produced; returning `None` makes
    /// the caller fall back to the input expression rather than inventing a
    /// symbol.
    fn decode_term(
        dag: &egglog::TermDag,
        id: egglog::TermId,
        pool: &ExprPool,
        atoms: &HashMap<String, ExprId>,
        memo: &mut HashMap<egglog::TermId, Option<ExprId>>,
    ) -> Option<ExprId> {
        use egglog::ast::Literal;
        use egglog::Term;
        if let Some(&done) = memo.get(&id) {
            return done;
        }
        let string_lit = |t: egglog::TermId| match dag.get(t) {
            Term::Lit(Literal::String(s)) => Some(s.as_str()),
            _ => None,
        };
        let Term::App(head, kids) = dag.get(id) else {
            return None;
        };
        let mut sub = |t: egglog::TermId| decode_term(dag, t, pool, atoms, memo);
        let result = match (head.as_str(), kids.as_slice()) {
            ("Num", &[k]) => match dag.get(k) {
                Term::Lit(Literal::Int(n)) => Some(pool.integer(*n)),
                _ => None,
            },
            ("Var", &[k]) => string_lit(k).and_then(|name| atoms.get(name).copied()),
            ("Add", &[a, b]) => (|| {
                let a = sub(a)?;
                let b = sub(b)?;
                let mut children = flatten_add_args(a, pool);
                children.extend(flatten_add_args(b, pool));
                Some(pool.add(children))
            })(),
            ("Mul", &[a, b]) => (|| {
                let a = sub(a)?;
                let b = sub(b)?;
                let mut children = flatten_mul_args(a, pool);
                children.extend(flatten_mul_args(b, pool));
                Some(pool.mul(children))
            })(),
            ("Pow", &[a, b]) => (|| {
                let a = sub(a)?;
                let b = sub(b)?;
                Some(pool.pow(a, b))
            })(),
            ("Fn", &[name, a]) => (|| {
                let name = string_lit(name)?;
                let arg = sub(a)?;
                Some(pool.func(name, vec![arg]))
            })(),
            ("Sin", &[a]) => sub(a).map(|a| pool.func("sin", vec![a])),
            ("Cos", &[a]) => sub(a).map(|a| pool.func("cos", vec![a])),
            ("Exp", &[a]) => sub(a).map(|a| pool.func("exp", vec![a])),
            ("Log", &[a]) => sub(a).map(|a| pool.func("log", vec![a])),
            ("Sqrt", &[a]) => sub(a).map(|a| pool.func("sqrt", vec![a])),
            _ => None,
        };
        memo.insert(id, result);
        result
    }

    /// Parse an extracted egglog term, printed as text, back to an
    /// [`ExprId`] — the pre-`TermDag` decoder, kept as the reference that
    /// [`decode_term`] is tested against.
    #[cfg(test)]
    fn parse_egglog_term(
        s: &str,
        pool: &ExprPool,
        atoms: &HashMap<String, ExprId>,
    ) -> Option<ExprId> {
        let s = s.trim();
        if s.starts_with('(') && s.ends_with(')') {
            let inner = &s[1..s.len() - 1];
            let (head, rest) = split_head(inner)?;
            match head {
                "Num" => {
                    let n: i64 = rest.trim().parse().ok()?;
                    Some(pool.integer(n))
                }
                "Var" => {
                    let name = unquote(rest.trim())?;
                    atoms.get(name).copied()
                }
                "Add" => {
                    let (a_str, b_str) = split_two_args(rest)?;
                    let a = parse_egglog_term(&a_str, pool, atoms)?;
                    let b = parse_egglog_term(&b_str, pool, atoms)?;
                    // RW-1: flatten binary tree back to n-ary on the way out.
                    let mut children = flatten_add_args(a, pool);
                    children.extend(flatten_add_args(b, pool));
                    Some(pool.add(children))
                }
                "Mul" => {
                    let (a_str, b_str) = split_two_args(rest)?;
                    let a = parse_egglog_term(&a_str, pool, atoms)?;
                    let b = parse_egglog_term(&b_str, pool, atoms)?;
                    let mut children = flatten_mul_args(a, pool);
                    children.extend(flatten_mul_args(b, pool));
                    Some(pool.mul(children))
                }
                "Pow" => {
                    let (a_str, b_str) = split_two_args(rest)?;
                    let a = parse_egglog_term(&a_str, pool, atoms)?;
                    let b = parse_egglog_term(&b_str, pool, atoms)?;
                    Some(pool.pow(a, b))
                }
                "Fn" => {
                    let (name_tok, remainder) = consume_term(rest)?;
                    let name = unquote(name_tok)?;
                    let arg = parse_egglog_term(remainder.trim(), pool, atoms)?;
                    Some(pool.func(name, vec![arg]))
                }
                "Sin" => Some(pool.func("sin", vec![parse_egglog_term(rest, pool, atoms)?])),
                "Cos" => Some(pool.func("cos", vec![parse_egglog_term(rest, pool, atoms)?])),
                "Exp" => Some(pool.func("exp", vec![parse_egglog_term(rest, pool, atoms)?])),
                "Log" => Some(pool.func("log", vec![parse_egglog_term(rest, pool, atoms)?])),
                "Sqrt" => Some(pool.func("sqrt", vec![parse_egglog_term(rest, pool, atoms)?])),
                _ => None,
            }
        } else {
            let n: i64 = s.parse().ok()?;
            Some(pool.integer(n))
        }
    }

    /// Strip the surrounding double quotes of an egglog string literal.
    ///
    /// Deliberately stricter than `trim_matches('"')`: an unquoted token is
    /// rejected rather than silently accepted.
    #[cfg(test)]
    fn unquote(s: &str) -> Option<&str> {
        s.trim().strip_prefix('"')?.strip_suffix('"')
    }

    #[cfg(test)]
    fn split_head(s: &str) -> Option<(&str, &str)> {
        let s = s.trim();
        let pos = s.find(|c: char| c.is_whitespace())?;
        Some((&s[..pos], &s[pos + 1..]))
    }

    #[cfg(test)]
    fn split_two_args(s: &str) -> Option<(String, String)> {
        let s = s.trim();
        let (first, remainder) = consume_term(s)?;
        let second = remainder.trim();
        Some((first.to_string(), second.to_string()))
    }

    #[cfg(test)]
    fn consume_term(s: &str) -> Option<(&str, &str)> {
        let s = s.trim_start();
        if s.starts_with('(') {
            let mut depth = 0usize;
            let mut in_string = false;
            for (i, c) in s.char_indices() {
                match c {
                    '"' => in_string = !in_string,
                    '(' if !in_string => depth += 1,
                    ')' if !in_string => {
                        depth -= 1;
                        if depth == 0 {
                            return Some((&s[..=i], &s[i + 1..]));
                        }
                    }
                    _ => {}
                }
            }
            None
        } else {
            let end = s
                .find(|c: char| c.is_whitespace() || c == ')')
                .unwrap_or(s.len());
            Some((&s[..end], &s[end..]))
        }
    }

    // -----------------------------------------------------------------------
    // RW-3: Linear-expression canonizer (post-extraction pass)
    // -----------------------------------------------------------------------

    /// Try to extract a linear term as `(integer_coefficient, base_expr)`.
    ///
    /// Recognises: bare symbols (coeff = 1) and `Mul(Integer, Symbol)`.
    fn extract_linear_term(expr: ExprId, pool: &ExprPool) -> Option<(i64, ExprId)> {
        match pool.get(expr) {
            ExprData::Symbol { .. } => Some((1, expr)),
            ExprData::Mul(args) if args.len() == 2 => {
                let (a, b) = (args[0], args[1]);
                if let ExprData::Integer(n) = pool.get(a) {
                    if matches!(pool.get(b), ExprData::Symbol { .. }) {
                        return n.0.to_i64().map(|c| (c, b));
                    }
                }
                if let ExprData::Integer(n) = pool.get(b) {
                    if matches!(pool.get(a), ExprData::Symbol { .. }) {
                        return n.0.to_i64().map(|c| (c, a));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Fold `Pow` with numeric integer base and non-negative integer exponent.
    ///
    /// Egglog's i64 `^` is bitwise XOR, so Pow constant folding cannot be done
    /// inside the egglog program without polluting the e-graph.
    pub(super) fn fold_numeric_pow(expr: ExprId, pool: &ExprPool) -> ExprId {
        fold_numeric_pow_memo(expr, pool, &mut HashMap::new())
    }

    /// Run `pass` on `expr` through `memo`: each post-pass below is a pure
    /// function of the `ExprId` it is given, so a subterm shared by many
    /// parents is rewritten once. Without this a DAG-shaped result — which is
    /// what a shared input extracts to — is walked as a tree.
    fn memoized(
        expr: ExprId,
        memo: &mut HashMap<ExprId, ExprId>,
        pass: impl FnOnce(&mut HashMap<ExprId, ExprId>) -> ExprId,
    ) -> ExprId {
        if let Some(&done) = memo.get(&expr) {
            return done;
        }
        let out = pass(memo);
        memo.insert(expr, out);
        out
    }

    fn fold_numeric_pow_memo(
        expr: ExprId,
        pool: &ExprPool,
        memo: &mut HashMap<ExprId, ExprId>,
    ) -> ExprId {
        memoized(expr, memo, |memo| fold_numeric_pow_node(expr, pool, memo))
    }

    fn fold_numeric_pow_node(
        expr: ExprId,
        pool: &ExprPool,
        memo: &mut HashMap<ExprId, ExprId>,
    ) -> ExprId {
        use crate::simplify::rules::RewriteRule;
        use rug::ops::Pow;
        let mut rec = |a: ExprId| fold_numeric_pow_memo(a, pool, memo);
        match pool.get(expr) {
            ExprData::Add(args) => {
                let args: Vec<ExprId> = args.iter().map(|&a| rec(a)).collect();
                pool.add(args)
            }
            ExprData::Mul(args) => {
                let args: Vec<ExprId> = args.iter().map(|&a| rec(a)).collect();
                pool.mul(args)
            }
            ExprData::Pow { base, exp } => {
                let base = rec(base);
                let exp = rec(exp);
                if let (ExprData::Integer(b), ExprData::Integer(e)) =
                    (pool.get(base), pool.get(exp))
                {
                    if b.0 == 1 {
                        return pool.integer(1_i32);
                    }
                    if b.0 == -1 {
                        let sign: i64 = if e.0.is_even() { 1 } else { -1 };
                        return pool.integer(sign);
                    }
                    if e.0 >= 0 {
                        if let Some(e_u32) = e.0.to_u32() {
                            let result: rug::Integer = b.0.clone().pow(e_u32);
                            return pool.integer(result);
                        }
                    }
                }
                pool.pow(base, exp)
            }
            ExprData::Func { name, args } => {
                let args: Vec<ExprId> = args.iter().map(|&a| rec(a)).collect();
                let folded = pool.func(&name, args);
                if name == "sqrt" {
                    if let Some((after, _)) =
                        crate::simplify::rules::SqrtInteger.apply(folded, pool)
                    {
                        return after;
                    }
                }
                folded
            }
            _ => expr,
        }
    }

    /// Canonicalize linear combinations in an expression.
    ///
    /// At each `Add` node, collects `(coefficient, symbol)` pairs and sums
    /// coefficients for identical bases, eliminating zero terms.
    ///
    /// Example: `2*x + 3*x + y` → `5*x + y`.
    pub(super) fn canonicalize_linear(expr: ExprId, pool: &ExprPool) -> ExprId {
        canonicalize_linear_memo(expr, pool, &mut HashMap::new())
    }

    fn canonicalize_linear_memo(
        expr: ExprId,
        pool: &ExprPool,
        memo: &mut HashMap<ExprId, ExprId>,
    ) -> ExprId {
        memoized(expr, memo, |memo| {
            canonicalize_linear_node(expr, pool, memo)
        })
    }

    fn canonicalize_linear_node(
        expr: ExprId,
        pool: &ExprPool,
        memo: &mut HashMap<ExprId, ExprId>,
    ) -> ExprId {
        let mut rec = |a: ExprId| canonicalize_linear_memo(a, pool, memo);
        match pool.get(expr) {
            ExprData::Add(args) => {
                let args: Vec<ExprId> = args.iter().map(|&a| rec(a)).collect();

                let mut coeff_map: HashMap<ExprId, i64> = HashMap::new();
                let mut non_linear: Vec<ExprId> = Vec::new();
                let mut found_linear = false;

                for &arg in &args {
                    if let Some((coeff, base)) = extract_linear_term(arg, pool) {
                        *coeff_map.entry(base).or_insert(0) += coeff;
                        found_linear = true;
                    } else {
                        non_linear.push(arg);
                    }
                }

                if !found_linear {
                    return pool.add(args);
                }

                let mut result: Vec<ExprId> = non_linear;
                // Sort by key for determinism
                let mut pairs: Vec<(ExprId, i64)> = coeff_map.into_iter().collect();
                pairs.sort_by_key(|(id, _)| *id);
                for (base, coeff) in pairs {
                    match coeff {
                        0 => {}
                        1 => result.push(base),
                        c => result.push(pool.mul(vec![pool.integer(c), base])),
                    }
                }

                match result.len() {
                    0 => pool.integer(0_i32),
                    1 => result[0],
                    _ => pool.add(result),
                }
            }
            ExprData::Mul(args) => {
                let args: Vec<ExprId> = args.iter().map(|&a| rec(a)).collect();
                pool.mul(args)
            }
            ExprData::Pow { base, exp } => {
                let base = rec(base);
                let exp = rec(exp);
                pool.pow(base, exp)
            }
            ExprData::Func { name, args } => {
                let args: Vec<ExprId> = args.iter().map(|&a| rec(a)).collect();
                pool.func(&name, args)
            }
            _ => expr,
        }
    }

    // -----------------------------------------------------------------------
    // Final post-extraction constant-fold pass
    // -----------------------------------------------------------------------

    /// Apply only the cheap constant-folding rules to `expr`, bottom-up, to a
    /// per-node fixpoint.
    ///
    /// This covers the folds not modeled inside the egglog program itself:
    /// elementary functions at 0/1, `x^0`/`x^1`, `1^r`, power-of-power,
    /// even-power sign folding, distribution of `pow` over a literal `Mul`
    /// coefficient, `Rational(n/1)` canonicalization, and `0`/`1`
    /// identities for `Add`/`Mul` — all via [`ConstFold`], [`PowZero`],
    /// [`PowOne`], [`AddZero`], [`MulOne`], and [`MulZero`].
    ///
    /// Nested `Add`/`Mul` trees are flattened before the local fold so that
    /// coefficients introduced by earlier post-passes (e.g. linear
    /// canonization turning `x+x` into `2·x` under an outer `·½`) meet in
    /// one n-ary product and fold (`2·½ → 1`). Without flattening,
    /// `ConstFold` only sees one numeric factor per nested `Mul` and leaves
    /// `((x * 2) * 1/2)` untouched.
    ///
    /// Unlike [`super::super::engine::simplify`], this does **not** run the
    /// full rule engine (no `SubSelf`/`DivSelf`, no discrimination-net
    /// pattern rules, no fixed-point loop over the whole tree) — each node
    /// is visited once and folded to a local fixpoint, so the pass is
    /// `O(n)` in the size of the extracted term rather than
    /// `O(n * iterations)`. The extracted term is already near-normal-form,
    /// so this bounded local fold is sufficient to pick up the constant
    /// folds above without re-running the whole simplifier.
    pub(super) fn apply_const_folds(expr: ExprId, pool: &ExprPool) -> ExprId {
        apply_const_folds_memo(expr, pool, &mut HashMap::new())
    }

    fn apply_const_folds_memo(
        expr: ExprId,
        pool: &ExprPool,
        memo: &mut HashMap<ExprId, ExprId>,
    ) -> ExprId {
        memoized(expr, memo, |memo| apply_const_folds_node(expr, pool, memo))
    }

    fn apply_const_folds_node(
        expr: ExprId,
        pool: &ExprPool,
        memo: &mut HashMap<ExprId, ExprId>,
    ) -> ExprId {
        use crate::simplify::rules::{
            AddZero, ConstFold, MulOne, MulZero, PowOne, PowZero, RewriteRule,
        };
        let mut rec = |a: ExprId| apply_const_folds_memo(a, pool, memo);

        // Recurse into children first, then flatten nested Add/Mul so
        // numeric factors from sibling subtrees share one n-ary node.
        let rebuilt = match pool.get(expr) {
            ExprData::Add(args) => {
                let args: Vec<ExprId> = args
                    .iter()
                    .map(|&a| rec(a))
                    .flat_map(|a| flatten_add_args(a, pool))
                    .collect();
                match args.len() {
                    0 => pool.integer(0_i32),
                    1 => args[0],
                    _ => pool.add(args),
                }
            }
            ExprData::Mul(args) => {
                let args: Vec<ExprId> = args
                    .iter()
                    .map(|&a| rec(a))
                    .flat_map(|a| flatten_mul_args(a, pool))
                    .collect();
                match args.len() {
                    0 => pool.integer(1_i32),
                    1 => args[0],
                    _ => pool.mul(args),
                }
            }
            ExprData::Pow { base, exp } => {
                let base = rec(base);
                let exp = rec(exp);
                pool.pow(base, exp)
            }
            ExprData::Func { name, args } => {
                let args: Vec<ExprId> = args.iter().map(|&a| rec(a)).collect();
                pool.func(&name, args)
            }
            _ => expr,
        };

        // Fold the rebuilt node to a local fixpoint with the cheap rules
        // only. Each rule either strictly shrinks the term or returns
        // `None`, so this loop terminates quickly.
        let mut current = rebuilt;
        loop {
            let next = AddZero
                .apply(current, pool)
                .or_else(|| MulZero.apply(current, pool))
                .or_else(|| MulOne.apply(current, pool))
                .or_else(|| PowZero.apply(current, pool))
                .or_else(|| PowOne.apply(current, pool))
                .or_else(|| ConstFold.apply(current, pool));
            match next {
                Some((after, _)) if after != current => {
                    // ConstFold / MulOne may reintroduce nesting; flatten
                    // again so a subsequent ConstFold can merge coefficients.
                    current = match pool.get(after) {
                        ExprData::Add(_) => {
                            let flat = flatten_add_args(after, pool);
                            match flat.len() {
                                0 => pool.integer(0_i32),
                                1 => flat[0],
                                _ => pool.add(flat),
                            }
                        }
                        ExprData::Mul(_) => {
                            let flat = flatten_mul_args(after, pool);
                            match flat.len() {
                                0 => pool.integer(1_i32),
                                1 => flat[0],
                                _ => pool.mul(flat),
                            }
                        }
                        _ => after,
                    };
                }
                _ => break,
            }
        }
        current
    }

    // -----------------------------------------------------------------------
    // 4. Public implementation
    // -----------------------------------------------------------------------

    pub fn simplify_egraph_impl(
        expr: ExprId,
        pool: &ExprPool,
        config: &super::EgraphConfig,
    ) -> crate::deriv::log::DerivedExpr<ExprId> {
        use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
        use crate::kernel::expr_props::expr_contains_noncommutative_symbol;

        if expr_contains_noncommutative_symbol(pool, expr) {
            return super::super::engine::simplify(expr, pool);
        }

        // `0⁻¹` in the e-graph makes `(Num 0)` and `(Num 1)` the same e-class
        // (see `has_provably_zero_denominator`); hand it to the rule engine,
        // which leaves the undefined product unevaluated.
        if has_provably_zero_denominator(expr, pool) {
            return super::super::engine::simplify(expr, pool);
        }

        // Enforce the node limit before handing the expression to egglog.
        // Saturation can materialise exponentially many equivalent forms, so a
        // hard pre-check on input size prevents OOM on pathological inputs.
        //
        // Counting *distinct* nodes bounds the real work because every stage
        // after this is DAG-sized: the encoder binds shared nodes once, the
        // extracted term is decoded from egglog's hash-consed `TermDag`, and
        // the post-passes are memoised per `ExprId`. (When the encoder wrote
        // the DAG out as a tree, a 120-node input passed this check and then
        // tried to build ~10⁸ terms.)
        if let Some(limit) = config.node_limit {
            let n = count_dag_nodes(expr, pool);
            if n > limit {
                let mut log = DerivationLog::new();
                log.push(RewriteStep::simple(
                    "egraph_node_limit_exceeded",
                    expr,
                    expr,
                ));
                return DerivedExpr::with_log(expr, log);
            }
        }

        let mut encoder = Encoder::new(pool);
        let encoded = encoder.encode(expr);
        let (prelude, schedule) = egglog_program(&encoded.lets, &encoded.root, config);
        let result = run_and_extract(&prelude, &schedule, pool, encoder.atoms());

        post_process(result.unwrap_or(expr), expr, pool)
    }

    /// The post-extraction passes and derivation log shared by the
    /// implementation and its test reference.
    fn post_process(
        extracted: ExprId,
        expr: ExprId,
        pool: &ExprPool,
    ) -> crate::deriv::log::DerivedExpr<ExprId> {
        use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
        let simplified = fold_numeric_pow(extracted, pool);
        // RW-3: apply linear canonizer as a post-extraction pass.
        let simplified = canonicalize_linear(simplified, pool);
        // Final post-extraction pass: apply only the cheap constant-folding
        // rules (elementary functions at 0/1, x^0/x^1, 1^r, power-of-power,
        // even-power sign folding, distribution of pow over a literal Mul
        // coefficient, Rational(n/1) canonicalization, and Add/Mul 0/1
        // identities) to the extracted term. See `apply_const_folds` — this
        // is a single bottom-up O(n) pass, not a full re-run of the rule
        // engine.
        let simplified = apply_const_folds(simplified, pool);

        let mut log = DerivationLog::new();
        if simplified != expr {
            log.push(RewriteStep::simple("egraph_simplify", expr, simplified));
        }
        DerivedExpr::with_log(simplified, log)
    }

    // -----------------------------------------------------------------------
    // Test reference: the pre-DAG pipeline
    // -----------------------------------------------------------------------

    /// The tree serialiser exactly as it was before the DAG encoder:
    /// `format!`-folded, no sharing. Kept to pin [`Encoder::encode`]'s tree
    /// path byte-for-byte.
    #[cfg(test)]
    pub(super) fn encode_tree_reference(enc: &mut Encoder<'_>, expr: ExprId) -> String {
        match classify(enc.pool, expr) {
            Node::Num(n) => format!("(Num {n})"),
            Node::Atom => {
                let mut s = String::new();
                enc.write_atom(expr, &mut s);
                s
            }
            Node::Add(args) => {
                let mut it = args.into_iter();
                let init = encode_tree_reference(enc, it.next().unwrap());
                it.fold(init, |acc, id| {
                    format!("(Add {acc} {})", encode_tree_reference(enc, id))
                })
            }
            Node::Mul(args) => {
                let mut it = args.into_iter();
                let init = encode_tree_reference(enc, it.next().unwrap());
                it.fold(init, |acc, id| {
                    format!("(Mul {acc} {})", encode_tree_reference(enc, id))
                })
            }
            Node::Pow(base, exp) => format!(
                "(Pow {} {})",
                encode_tree_reference(enc, base),
                encode_tree_reference(enc, exp)
            ),
            Node::Known(ctor, arg) => format!("({ctor} {})", encode_tree_reference(enc, arg)),
            Node::Fn(name, arg) => format!("(Fn \"{name}\" {})", encode_tree_reference(enc, arg)),
        }
    }

    /// Test-only: the pipeline as it was before this module went DAG-sized —
    /// the input written out as a tree, a fresh `EGraph` running the whole
    /// program, and the printed extraction re-parsed as text. Exponential on
    /// shared inputs; the differential tests keep them small.
    #[cfg(test)]
    pub(super) fn simplify_egraph_reference(
        expr: ExprId,
        pool: &ExprPool,
        config: &super::EgraphConfig,
    ) -> ExprId {
        use crate::kernel::expr_props::expr_contains_noncommutative_symbol;
        if expr_contains_noncommutative_symbol(pool, expr)
            || has_provably_zero_denominator(expr, pool)
        {
            return super::super::engine::simplify(expr, pool).value;
        }
        let mut encoder = Encoder::new(pool);
        let expr_str = encode_tree_reference(&mut encoder, expr);
        let (prelude, schedule) = egglog_program("", &expr_str, config);
        let result: Option<ExprId> = (|| {
            let mut egraph = egglog::EGraph::default();
            let outputs = egraph
                .parse_and_run_program(None, &format!("{prelude}{schedule}"))
                .ok()?;
            let term_str = outputs.into_iter().last()?;
            parse_egglog_term(&term_str, pool, encoder.atoms())
        })();
        post_process(result.unwrap_or(expr), expr, pool).value
    }

    /// Test-only: the new encoder's output, for byte-level comparison.
    #[cfg(test)]
    pub(super) fn encode_for_test(pool: &ExprPool, expr: ExprId) -> (String, String) {
        let mut enc = Encoder::new(pool);
        let e = enc.encode(expr);
        (e.lets, e.root)
    }

    #[cfg(test)]
    pub(super) fn encode_tree_reference_for_test(pool: &ExprPool, expr: ExprId) -> String {
        encode_tree_reference(&mut Encoder::new(pool), expr)
    }
}

// ---------------------------------------------------------------------------
// PA-6 / RW-4 — Pluggable e-graph cost functions
// ---------------------------------------------------------------------------

use crate::deriv::log::DerivedExpr;
use crate::kernel::{ExprId, ExprPool};

/// Cost model used when extracting from the e-graph.
///
/// The extractor chooses the expression with the *lowest* total cost.
/// Implement this trait to define custom extraction objectives.
///
/// # Built-in implementations
///
/// | Type | Description |
/// |------|-------------|
/// | [`SizeCost`] | Every node costs 1 (tree size). Default. |
/// | [`OpCost`]   | Operators weighted by evaluation cost. |
/// | [`DepthCost`]| Cost = max child depth + 1. |
/// | [`StabilityCost`] | Penalises catastrophic cancellation. |
/// | [`NoncommutativeCost`] | Tie-break for non-commutative `Mul` chains (V3-2). |
pub trait EgraphCost: Send + Sync {
    /// Compute the cost of a node given its operator name and its children's costs.
    fn cost(&self, op: &str, child_costs: &[f64]) -> f64;
}

/// Every node costs 1 (tree-size cost). This is the egglog default.
pub struct SizeCost;
impl EgraphCost for SizeCost {
    fn cost(&self, _op: &str, child_costs: &[f64]) -> f64 {
        1.0 + child_costs.iter().sum::<f64>()
    }
}

/// Operators weighted by their numerical evaluation cost.
pub struct OpCost;
impl EgraphCost for OpCost {
    fn cost(&self, op: &str, child_costs: &[f64]) -> f64 {
        let w = match op {
            "Num" | "Var" => 0.1,
            "Add" => 1.0,
            "Mul" => 1.5,
            "Pow" => 3.0,
            "Sin" | "Cos" | "Exp" | "Log" | "Sqrt" => 5.0,
            _ => 2.0,
        };
        w + child_costs.iter().sum::<f64>()
    }
}

/// Cost = max child depth + 1.
///
/// Minimises the critical-path length; useful for GPU / parallel evaluation
/// where depth determines the number of synchronisation barriers.
pub struct DepthCost;
impl EgraphCost for DepthCost {
    fn cost(&self, _op: &str, child_costs: &[f64]) -> f64 {
        1.0 + child_costs.iter().cloned().fold(0.0_f64, f64::max)
    }
}

/// Penalises catastrophic cancellation.
///
/// Applies a `3×` multiplier to binary `Add`/`Sub` nodes whose both children
/// have non-trivial cost (i.e. not a bare literal), discouraging expressions
/// of the form `large_expr - large_expr` in favour of Horner form or
/// log-sum-exp style rewrites.
pub struct StabilityCost;
impl EgraphCost for StabilityCost {
    fn cost(&self, op: &str, child_costs: &[f64]) -> f64 {
        let base = 1.0 + child_costs.iter().sum::<f64>();
        match op {
            // Penalise binary add/sub between two non-trivial children.
            "Add" | "Sub"
                if child_costs.len() == 2 && child_costs[0] > 1.0 && child_costs[1] > 1.0 =>
            {
                base * 3.0
            }
            "Pow" => base * 2.0,
            _ => base,
        }
    }
}

/// Extraction cost biased toward **left-to-right** (`Mul`) products (V3-2).
///
/// When egglog gains a fully pluggable extractor, this can rank
/// normal-ordered operator strings (Pauli / Clifford) lower than scrambled
/// permutations. Today it adds a small tie-break on `Mul` so experiments
/// with non-commuting `Var` encodings stay deterministic.
pub struct NoncommutativeCost;
impl EgraphCost for NoncommutativeCost {
    fn cost(&self, op: &str, child_costs: &[f64]) -> f64 {
        let base = SizeCost.cost(op, child_costs);
        match op {
            "Mul" => base + 1.0e-6 * child_costs.len() as f64,
            _ => base,
        }
    }
}

// ---------------------------------------------------------------------------
// PA-6 — Schedule configuration  (RW-2: node_limit / iter_limit)
// ---------------------------------------------------------------------------

/// Configuration for the e-graph schedule and extraction strategy.
///
/// Pass to [`simplify_egraph_with`] to customise iteration counts and
/// resource limits.
///
/// # Rule flags
///
/// By default `include_trig_rules` is `true` so `simplify_egraph` reduces
/// `sin²(x)+cos²(x)→1`. Log/exp cancellation is **not** loaded in egglog
/// (no domain check); use [`crate::simplify::engine::simplify_log_exp`].
/// without any extra configuration.  Set either flag to `false` to suppress
/// the corresponding rule set (useful when you need to benchmark rule impact or
/// avoid domain-sensitive rewrites).
#[derive(Debug, Clone)]
pub struct EgraphConfig {
    /// Saturation iterations in the *shrinking* phase. Default 5.
    pub shrink_iters: usize,
    /// Saturation iterations in the *exploring* phase. Default 3.
    pub explore_iters: usize,
    /// Constant-folding iterations appended after each phase. Default 3.
    pub const_fold_iters: usize,
    /// Abort if the e-graph exceeds this many nodes. `None` = unlimited.
    pub node_limit: Option<usize>,
    /// Per-ruleset iteration cap passed to egglog's scheduler. `None` = unlimited.
    pub iter_limit: Option<usize>,
    /// Include the Pythagorean trig identity (`sin²+cos²→1`) in the explore phase.
    /// Default `true`.
    pub include_trig_rules: bool,
    /// Reserved for log/exp egglog rules. Currently a no-op: principal-branch
    /// log/exp cancellation needs a domain check egglog cannot express.
    /// Prefer [`crate::simplify::engine::simplify_log_exp`]. Default `true`.
    pub include_log_exp_rules: bool,
    /// Schedule match-disjoint egglog rule groups (Add / Mul / Pow / trig / log)
    /// as separate `(run …)` steps within each phase. Default `true`.
    pub disjoint_schedule: bool,
}

impl Default for EgraphConfig {
    fn default() -> Self {
        EgraphConfig {
            shrink_iters: 5,
            explore_iters: 3,
            const_fold_iters: 3,
            node_limit: None,
            iter_limit: None,
            include_trig_rules: true,
            include_log_exp_rules: true,
            disjoint_schedule: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Simplify `expr` using the e-graph backend with default settings.
///
/// Falls back to the rule-based simplifier when `egraph` feature is off.
pub fn simplify_egraph(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    #[cfg(feature = "egraph")]
    {
        backend::simplify_egraph_impl(expr, pool, &EgraphConfig::default())
    }
    #[cfg(not(feature = "egraph"))]
    {
        super::engine::simplify(expr, pool)
    }
}

/// Simplify `expr` using the e-graph backend with a custom configuration.
///
/// The `cost` parameter documents the intended extraction preference; full
/// pluggable-extractor support requires a future egglog API.  The config
/// schedule limits (`node_limit`, `iter_limit`, phase iters) are wired
/// into the egglog program today.
pub fn simplify_egraph_with(
    expr: ExprId,
    pool: &ExprPool,
    config: &EgraphConfig,
    _cost: &dyn EgraphCost,
) -> DerivedExpr<ExprId> {
    #[cfg(feature = "egraph")]
    {
        backend::simplify_egraph_impl(expr, pool, config)
    }
    #[cfg(not(feature = "egraph"))]
    {
        let _ = config;
        super::engine::simplify(expr, pool)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn egraph_simplify_x_plus_y_minus_x() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.add(vec![x, y, neg_x]);
        let result = simplify_egraph(expr, &pool);
        assert_ne!(result.value, pool.integer(0_i32), "should not be zero");
    }

    #[test]
    fn egraph_simplify_const_fold() {
        let pool = ExprPool::new();
        let expr = pool.add(vec![pool.integer(3_i32), pool.integer(4_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, pool.integer(7_i32));
    }

    #[test]
    fn egraph_simplify_add_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.integer(0_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, x);
    }

    #[test]
    fn egraph_simplify_mul_one() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.integer(1_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, x);
    }

    #[test]
    fn egraph_simplify_mul_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.integer(0_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, pool.integer(0_i32));
    }

    #[test]
    fn egraph_fallback_no_panic_on_rational() {
        let pool = ExprPool::new();
        let r = pool.rational(1, 3);
        let _ = simplify_egraph(r, &pool);
    }

    // RW-1: flattening round-trip
    #[test]
    fn egraph_round_trips_nary_add() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        // x + y + z should survive the egglog round-trip as a 3-arg Add
        let expr = pool.add(vec![x, y, z]);
        let result = simplify_egraph(expr, &pool);
        // Must still be an Add (not a nested binary tree)
        if let crate::kernel::ExprData::Add(args) =
            crate::kernel::ExprPool::get(&pool, result.value)
        {
            assert_eq!(args.len(), 3);
        }
    }

    // RW-3: linear canonizer
    #[test]
    fn linear_canonizer_combines_like_terms() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // 2*x + 3*x = 5*x
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let three_x = pool.mul(vec![pool.integer(3_i32), x]);
        let expr = pool.add(vec![two_x, three_x]);
        #[cfg(feature = "egraph")]
        {
            let result = backend::canonicalize_linear(expr, &pool);
            let five_x = pool.mul(vec![pool.integer(5_i32), x]);
            assert_eq!(result, five_x);
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    /// report7-20: nested `(2·x)·(1/2)` must fold after linear canonization.
    #[test]
    fn apply_const_folds_flattens_nested_mul_coefficients() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let half = pool.rational(1, 2);
        let nested = pool.mul(vec![two_x, half]);
        #[cfg(feature = "egraph")]
        {
            let folded = backend::apply_const_folds(nested, &pool);
            assert_eq!(folded, x, "expected (2*x)*(1/2) → x, got {folded:?}");
        }
        #[cfg(not(feature = "egraph"))]
        let _ = nested;
    }

    /// report7-20 headline: `simplify_egraph((x+x)/2)` → `x`.
    #[test]
    fn egraph_folds_double_over_two() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let inv2 = pool.pow(pool.integer(2_i32), pool.integer(-1_i32));
        let expr = pool.mul(vec![pool.add(vec![x, x]), inv2]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(
            result.value, x,
            "expected (x+x)/2 → x, got {:?}",
            result.value
        );
    }

    // RW-2: config wiring compiles and does not panic
    #[test]
    fn egraph_with_node_limit() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.integer(0_i32)]);
        let config = EgraphConfig {
            node_limit: Some(10_000),
            ..EgraphConfig::default()
        };
        let result = simplify_egraph_with(expr, &pool, &config, &SizeCost);
        assert_eq!(result.value, x);
    }

    #[test]
    fn egraph_noncommutative_falls_back_to_rules() {
        let pool = ExprPool::new();
        let a = pool.symbol_commutative("A", Domain::Real, false);
        let expr = pool.add(vec![a, pool.integer(0_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, a);
    }

    // V3-2: NoncommutativeCost is callable
    #[test]
    fn noncommutative_cost_is_callable() {
        let nc = NoncommutativeCost;
        let v = nc.cost("Mul", &[1.0, 1.0]);
        assert!(v.is_finite());
    }

    // RW-4: StabilityCost is callable
    #[test]
    fn stability_cost_penalises_binary_add() {
        let sc = StabilityCost;
        let penalised = sc.cost("Add", &[2.0, 2.0]);
        let normal = sc.cost("Add", &[0.1, 2.0]);
        assert!(penalised > normal);
    }

    #[test]
    fn egraph_sqrt_trig_identity_squared() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let inner = pool.add(vec![
            pool.pow(sin_x, pool.integer(2_i32)),
            pool.pow(cos_x, pool.integer(2_i32)),
        ]);
        let expr = pool.func("sqrt", vec![pool.pow(inner, pool.integer(2_i32))]);
        #[cfg(feature = "egraph")]
        {
            let result = simplify_egraph(expr, &pool);
            assert_eq!(
                result.value,
                pool.integer(1_i32),
                "got {}",
                pool.display(result.value)
            );
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    #[test]
    fn egraph_pow_one_squared_is_one() {
        let pool = ExprPool::new();
        let expr = pool.pow(pool.integer(1_i32), pool.integer(2_i32));
        #[cfg(feature = "egraph")]
        {
            let result = simplify_egraph(expr, &pool);
            assert_eq!(
                result.value,
                pool.integer(1_i32),
                "got {}",
                pool.display(result.value)
            );
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    // V1-15: trig identity via Pow form (sin(x)^2 + cos(x)^2 → 1)
    #[test]
    fn egraph_trig_identity_pow_form() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let sin2 = pool.pow(sin_x, pool.integer(2_i32));
        let cos2 = pool.pow(cos_x, pool.integer(2_i32));
        let expr = pool.add(vec![sin2, cos2]);
        #[cfg(feature = "egraph")]
        {
            let result = simplify_egraph(expr, &pool);
            assert_eq!(result.value, pool.integer(1_i32));
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    // exp(log(x)) → x requires positivity; egglog must not fire it ungated.
    #[test]
    fn egraph_exp_of_log_stays_without_assumptions() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![pool.func("log", vec![x])]);
        #[cfg(feature = "egraph")]
        {
            let result = simplify_egraph(expr, &pool);
            assert_eq!(result.value, expr, "got {}", pool.display(result.value));
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    // log(exp(x)) must not fire in egglog (no domain check).
    #[test]
    fn egraph_log_of_exp_stays() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("log", vec![pool.func("exp", vec![x])]);
        #[cfg(feature = "egraph")]
        {
            let result = simplify_egraph(expr, &pool);
            assert_eq!(result.value, expr, "got {}", pool.display(result.value));
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    // V1-15: opt-out trig rules via config
    #[test]
    fn egraph_opt_out_trig_rules() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let sin2 = pool.pow(sin_x, pool.integer(2_i32));
        let cos2 = pool.pow(cos_x, pool.integer(2_i32));
        let expr = pool.add(vec![sin2, cos2]);
        let config = EgraphConfig {
            include_trig_rules: false,
            ..EgraphConfig::default()
        };
        let result = simplify_egraph_with(expr, &pool, &config, &SizeCost);
        assert_ne!(result.value, pool.integer(1_i32));
    }

    // Opt-out flag remains API-stable; log/exp egglog rules are already empty.
    #[test]
    fn egraph_opt_out_log_exp_rules() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("log", vec![pool.func("exp", vec![x])]);
        let config = EgraphConfig {
            include_log_exp_rules: false,
            ..EgraphConfig::default()
        };
        let result = simplify_egraph_with(expr, &pool, &config, &SizeCost);
        assert_eq!(result.value, expr);
    }
}

// ---------------------------------------------------------------------------
// Opaque-atom soundness regression tests
//
// Before the opaque-atom encoding, every node kind the egglog datatype could
// not express was serialised as the *literal number zero*.  The arithmetic
// rules then fired on it correctly but on a corrupted term, so
// `simplify_egraph` silently returned wrong answers: `1/2 → 0`,
// `x + 1/2 → x`, `x^(1/2) → 1`, and so on for rationals, floats,
// out-of-`i64` integers, n-ary `Func`, `Piecewise`, `Predicate`, `Forall`,
// `Exists`, `RootSum` and `BigO`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod opaque_atom_tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use rug::ops::Pow as _;

    /// A node that the egglog datatype cannot express must survive
    /// `simplify_egraph` untouched — bare and in every position where an
    /// arithmetic rule could have matched a literal `0`.
    fn assert_opaque_preserved(pool: &ExprPool, node: ExprId, label: &str) {
        let x = pool.symbol("x_probe", Domain::Real);

        let got = simplify_egraph(node, pool).value;
        assert_eq!(
            got,
            node,
            "{label}: bare — expected {}, got {}",
            pool.display(node),
            pool.display(got)
        );

        // `(Add ?x (Num 0)) → ?x` must not swallow it.
        let sum = pool.add(vec![x, node]);
        let got = simplify_egraph(sum, pool).value;
        assert_eq!(
            got,
            sum,
            "{label}: x + node — expected {}, got {}",
            pool.display(sum),
            pool.display(got)
        );

        // `(Mul ?x (Num 0)) → (Num 0)` must not absorb it.
        let prod = pool.mul(vec![x, node]);
        let got = simplify_egraph(prod, pool).value;
        assert_eq!(
            got,
            prod,
            "{label}: x * node — expected {}, got {}",
            pool.display(prod),
            pool.display(got)
        );

        // `(Pow ?x (Num 0)) → (Num 1)` must not fire on it.
        let power = pool.pow(x, node);
        let got = simplify_egraph(power, pool).value;
        assert_eq!(
            got,
            power,
            "{label}: x^node — expected {}, got {}",
            pool.display(power),
            pool.display(got)
        );
    }

    // -- the six reported reproduction cases --------------------------------

    #[test]
    fn egraph_preserves_bare_rational() {
        let pool = ExprPool::new();
        let half = pool.rational(1, 2);
        assert_eq!(simplify_egraph(half, &pool).value, half);
    }

    #[test]
    fn egraph_preserves_symbol_over_rational() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.rational(1, 2)]);
        assert_eq!(simplify_egraph(expr, &pool).value, expr);
    }

    #[test]
    fn egraph_preserves_symbol_plus_rational() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.rational(1, 2)]);
        assert_eq!(simplify_egraph(expr, &pool).value, expr);
    }

    #[test]
    fn egraph_preserves_rational_exponent() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.rational(1, 2));
        assert_eq!(simplify_egraph(expr, &pool).value, expr);
    }

    #[test]
    fn egraph_preserves_integer_to_rational_power() {
        let pool = ExprPool::new();
        let expr = pool.pow(pool.integer(2_i32), pool.rational(1, 2));
        assert_eq!(simplify_egraph(expr, &pool).value, expr);
    }

    #[test]
    fn egraph_preserves_symbol_plus_float() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.float(0.5, 53)]);
        assert_eq!(simplify_egraph(expr, &pool).value, expr);
    }

    // -- one case per node kind routed through the opaque path ---------------

    #[test]
    fn egraph_preserves_rational_literal() {
        let pool = ExprPool::new();
        assert_opaque_preserved(&pool, pool.rational(2, 3), "Rational");
    }

    #[test]
    fn egraph_preserves_float_literal() {
        let pool = ExprPool::new();
        assert_opaque_preserved(&pool, pool.float(0.5, 53), "Float");
    }

    /// Integers wider than `i64` used to be clamped to `i64::MAX`/`MIN`.
    #[test]
    fn egraph_preserves_out_of_range_integer() {
        let pool = ExprPool::new();
        let big = pool.integer(rug::Integer::from(2).pow(100_u32));
        assert_opaque_preserved(&pool, big, "Integer(2^100)");
    }

    #[test]
    fn egraph_preserves_binary_func() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        assert_opaque_preserved(&pool, pool.func("atan2", vec![x, y]), "Func/2");
    }

    #[test]
    fn egraph_preserves_nullary_func() {
        let pool = ExprPool::new();
        assert_opaque_preserved(&pool, pool.func("rand", vec![]), "Func/0");
    }

    #[test]
    fn egraph_preserves_piecewise() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let pw = pool.piecewise(
            vec![(pool.pred_gt(x, pool.integer(0_i32)), x)],
            pool.integer(-1_i32),
        );
        assert_opaque_preserved(&pool, pw, "Piecewise");
    }

    #[test]
    fn egraph_preserves_predicate() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        assert_opaque_preserved(&pool, pool.pred_gt(x, y), "Predicate");
    }

    #[test]
    fn egraph_preserves_forall() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let body = pool.pred_ge(pool.pow(x, pool.integer(2_i32)), pool.integer(0_i32));
        assert_opaque_preserved(&pool, pool.forall(x, body), "Forall");
    }

    #[test]
    fn egraph_preserves_exists() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let body = pool.pred_eq(pool.pow(x, pool.integer(2_i32)), pool.integer(2_i32));
        assert_opaque_preserved(&pool, pool.exists(x, body), "Exists");
    }

    #[test]
    fn egraph_preserves_root_sum() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let c = pool.symbol("c", Domain::Real);
        // Σ_{c : c² − 2 = 0} log(x − c)
        let poly = pool.add(vec![pool.pow(c, pool.integer(2_i32)), pool.integer(-2_i32)]);
        let body = pool.func(
            "log",
            vec![pool.add(vec![x, pool.mul(vec![pool.integer(-1_i32), c])])],
        );
        assert_opaque_preserved(&pool, pool.root_sum(poly, c, body), "RootSum");
    }

    #[test]
    fn egraph_preserves_big_o() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        assert_opaque_preserved(&pool, pool.big_o(x), "BigO");
    }

    /// A unary function whose name is not a plain identifier cannot be
    /// embedded in an egglog string literal, so it also takes the opaque path.
    #[test]
    fn egraph_preserves_func_with_exotic_name() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        assert_opaque_preserved(&pool, pool.func("weird name\"", vec![x]), "Func/exotic");
    }

    // -- uninterpreted unary functions --------------------------------------

    /// Unknown unary functions used to be mangled into a symbol literally
    /// named `"tan_(Num 3)"`.
    #[test]
    fn egraph_preserves_unknown_unary_func() {
        let pool = ExprPool::new();
        let expr = pool.func("tan", vec![pool.integer(3_i32)]);
        let got = simplify_egraph(expr, &pool).value;
        assert_eq!(got, expr, "got {}", pool.display(got));
    }

    /// `Fn` keeps unknown functions *structured*, so rules still fire inside
    /// the argument.
    #[test]
    fn egraph_simplifies_inside_unknown_unary_func() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("tan", vec![pool.add(vec![x, pool.integer(0_i32)])]);
        let got = simplify_egraph(expr, &pool).value;
        assert_eq!(
            got,
            pool.func("tan", vec![x]),
            "expected tan(x), got {}",
            pool.display(got)
        );
    }

    /// Distinct unknown functions of the same argument must stay distinct.
    #[test]
    fn egraph_distinct_unknown_funcs_do_not_collide() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("tan", vec![x]);
        let g = pool.func("erf", vec![x]);
        let expr = pool.add(vec![f, pool.mul(vec![pool.integer(-1_i32), g])]);
        let got = simplify_egraph(expr, &pool).value;
        assert_ne!(got, pool.integer(0_i32), "tan(x) − erf(x) must not be 0");
    }

    // -- atom identity: same subterm unifies, different subterms do not ------

    /// A subterm containing an opaque atom still cancels against itself,
    /// because equal subterms share one atom.
    ///
    /// `-1` is interned first on purpose: `pool.mul` sorts its children by
    /// `ExprId`, and the cancellation rule is spelled `(Mul (Num -1) ?x)`,
    /// so it only matches when `-1` sorts to the front.  That ordering
    /// sensitivity is a pre-existing property of the rule set — plain
    /// `(x + y) − (x + y)` behaves exactly the same way — and is independent
    /// of the atom encoding under test here.
    ///
    /// Gated on the `egraph` feature: without it `simplify_egraph` returns the
    /// expression unchanged, so there is no rewrite to cancel. The sibling
    /// `egraph_preserves_*` tests are deliberately left ungated — they assert
    /// that nothing is *corrupted*, which the no-op fallback satisfies too.
    #[cfg(feature = "egraph")]
    #[test]
    fn egraph_identical_rational_subterms_cancel() {
        let pool = ExprPool::new();
        let neg = pool.integer(-1_i32);
        let x = pool.symbol("x", Domain::Real);
        let t = pool.add(vec![x, pool.rational(1, 2)]);
        let expr = pool.add(vec![t, pool.mul(vec![neg, t])]);
        let got = simplify_egraph(expr, &pool).value;
        assert_eq!(
            got,
            pool.integer(0_i32),
            "(x + 1/2) − (x + 1/2) should be 0, got {}",
            pool.display(got)
        );
    }

    /// The same, for a wholly opaque subterm.
    #[test]
    fn egraph_identical_opaque_subterms_cancel() {
        let pool = ExprPool::new();
        let neg = pool.integer(-1_i32);
        let x = pool.symbol("x", Domain::Real);
        let pw = pool.piecewise(vec![(pool.pred_gt(x, pool.integer(0_i32)), x)], neg);
        let expr = pool.add(vec![pw, pool.mul(vec![neg, pw])]);
        let got = simplify_egraph(expr, &pool).value;
        assert_eq!(
            got,
            pool.integer(0_i32),
            "p − p should be 0 for opaque p, got {}",
            pool.display(got)
        );
    }

    /// Two occurrences of the same rational literal share one atom.
    #[test]
    fn egraph_identical_rational_atoms_cancel() {
        let pool = ExprPool::new();
        let neg = pool.integer(-1_i32);
        let half = pool.rational(1, 2);
        let expr = pool.add(vec![half, pool.mul(vec![neg, half])]);
        let got = simplify_egraph(expr, &pool).value;
        assert_eq!(
            got,
            pool.integer(0_i32),
            "1/2 − 1/2 should be 0, got {}",
            pool.display(got)
        );
    }

    #[test]
    fn egraph_distinct_rationals_do_not_collide() {
        let pool = ExprPool::new();
        let a = pool.rational(1, 2);
        let b = pool.rational(1, 3);
        let expr = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        let got = simplify_egraph(expr, &pool).value;
        assert_ne!(got, pool.integer(0_i32), "1/2 − 1/3 must not be 0");
    }

    #[test]
    fn egraph_distinct_opaque_subterms_do_not_collide() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let ox = pool.big_o(x);
        let oy = pool.big_o(y);
        let expr = pool.add(vec![ox, pool.mul(vec![pool.integer(-1_i32), oy])]);
        let got = simplify_egraph(expr, &pool).value;
        assert_ne!(got, pool.integer(0_i32), "O(x) − O(y) must not be 0");
    }

    // -- user symbols cannot collide with generated atom names ---------------

    /// A user is allowed to name a symbol `a0` / `a1` — exactly the shape of
    /// a generated opaque atom.  Symbol names are never emitted into the
    /// egglog program, so a collision is impossible by construction.
    #[test]
    fn egraph_user_symbol_shaped_like_atom_does_not_collide() {
        let pool = ExprPool::new();
        let a0 = pool.symbol("a0", Domain::Real);
        let a1 = pool.symbol("a1", Domain::Real);
        let a2 = pool.symbol("a2", Domain::Real);
        let neg = pool.integer(-1_i32);
        let zero = pool.integer(0_i32);

        for (lhs, rhs, label) in [
            (a0, pool.rational(1, 2), "a0 − 1/2"),
            (a1, pool.rational(1, 2), "a1 − 1/2"),
            (a2, pool.float(0.25, 53), "a2 − 0.25"),
            (a0, pool.big_o(a1), "a0 − O(a1)"),
            (a0, a1, "a0 − a1"),
        ] {
            let expr = pool.add(vec![lhs, pool.mul(vec![neg, rhs])]);
            let got = simplify_egraph(expr, &pool).value;
            assert_ne!(got, zero, "{label} must not cancel");
        }

        // …while genuine self-cancellation still works.
        let expr = pool.add(vec![a0, pool.mul(vec![neg, a0])]);
        assert_eq!(simplify_egraph(expr, &pool).value, zero, "a0 − a0 = 0");
    }

    /// Symbols round-trip as themselves, keeping their domain and
    /// commutativity flag instead of being re-interned as `Domain::Real`.
    #[test]
    fn egraph_preserves_symbol_domain() {
        let pool = ExprPool::new();
        let z = pool.symbol("z", Domain::Complex);
        let expr = pool.add(vec![z, pool.integer(0_i32)]);
        let got = simplify_egraph(expr, &pool).value;
        assert_eq!(got, z, "z:Complex + 0 must stay the Complex symbol");
    }
}

// ---------------------------------------------------------------------------
// Property test: simplify_egraph must preserve semantics
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "egraph"))]
mod opaque_atom_proptests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::{Domain, ExprPool};
    use proptest::prelude::*;
    use std::collections::HashMap;

    /// A small expression grammar covering every serialiser path: integer
    /// literals (`Num`), rationals and floats (opaque atoms), symbols
    /// (atoms), `Add`/`Mul`/`Pow`, known unary functions, an uninterpreted
    /// unary function (`Fn`), and a binary function (opaque atom).
    #[derive(Debug, Clone)]
    enum Ast {
        Int(i64),
        Rat(i64, i64),
        Flt(i32),
        X,
        Y,
        Add(Box<Ast>, Box<Ast>),
        Mul(Box<Ast>, Box<Ast>),
        Neg(Box<Ast>),
        PowI(Box<Ast>, i32),
        /// `base^(1/2)` — a rational in exponent position.
        PowHalf(Box<Ast>),
        Sin(Box<Ast>),
        Cos(Box<Ast>),
        Sqrt(Box<Ast>),
        /// Uninterpreted unary function.
        Tan(Box<Ast>),
    }

    fn build(ast: &Ast, pool: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
        match ast {
            Ast::Int(n) => pool.integer(*n),
            Ast::Rat(n, d) => pool.rational(*n, *d),
            Ast::Flt(k) => pool.float(f64::from(*k) / 8.0, 53),
            Ast::X => x,
            Ast::Y => y,
            Ast::Add(a, b) => pool.add(vec![build(a, pool, x, y), build(b, pool, x, y)]),
            Ast::Mul(a, b) => pool.mul(vec![build(a, pool, x, y), build(b, pool, x, y)]),
            Ast::Neg(a) => pool.mul(vec![pool.integer(-1_i32), build(a, pool, x, y)]),
            Ast::PowI(a, e) => pool.pow(build(a, pool, x, y), pool.integer(*e)),
            Ast::PowHalf(a) => pool.pow(build(a, pool, x, y), pool.rational(1, 2)),
            Ast::Sin(a) => pool.func("sin", vec![build(a, pool, x, y)]),
            Ast::Cos(a) => pool.func("cos", vec![build(a, pool, x, y)]),
            Ast::Sqrt(a) => pool.func("sqrt", vec![build(a, pool, x, y)]),
            Ast::Tan(a) => pool.func("tan", vec![build(a, pool, x, y)]),
        }
    }

    fn ast_strategy() -> impl Strategy<Value = Ast> {
        let leaf = prop_oneof![
            (-6i64..=6).prop_map(Ast::Int),
            (-6i64..=6, 1i64..=6).prop_map(|(n, d)| Ast::Rat(n, d)),
            (-40i32..=40).prop_map(Ast::Flt),
            Just(Ast::X),
            Just(Ast::Y),
        ];
        leaf.prop_recursive(4, 32, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Ast::Add(Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Ast::Mul(Box::new(a), Box::new(b))),
                inner.clone().prop_map(|a| Ast::Neg(Box::new(a))),
                (inner.clone(), -2i32..=3).prop_map(|(a, e)| Ast::PowI(Box::new(a), e)),
                inner.clone().prop_map(|a| Ast::PowHalf(Box::new(a))),
                inner.clone().prop_map(|a| Ast::Sin(Box::new(a))),
                inner.clone().prop_map(|a| Ast::Cos(Box::new(a))),
                inner.clone().prop_map(|a| Ast::Sqrt(Box::new(a))),
                // `Ast::Atan2` is deliberately absent from this *numeric*
                // property. `x·0 → 0` discards the sign of IEEE zero — for
                // negative `x`, `x·0` is `-0.0` — and `atan2` is discontinuous
                // across signed zero, so the original and simplified forms land
                // on opposite sides of its branch cut: `atan2(-0.0, -1) = -π`
                // vs `atan2(0.0, -1) = +π`, and `atan2(0, sqrt(-0.0)) = π` vs
                // `atan2(0, 0) = 0`. The gap is not a fixed multiple of π, so
                // no tolerance rescues it.
                //
                // This is not an egraph artifact — mainline `simplify` and
                // `simplify_expanded` perform the identical rewrite, as does
                // every mainstream CAS. Float equivalence is simply not a sound
                // oracle for `atan2` under a rewrite that holds only up to
                // signed zero. Binary-`Func` preservation (the arity-2 case the
                // `(Num 0)` bug corrupted) is covered structurally and exactly
                // by `egraph_preserves_binary_func` instead.
                inner.prop_map(|a| Ast::Tan(Box::new(a))),
            ]
        })
    }

    /// Generic sample points — deliberately not small integers, so an
    /// accidental agreement is unlikely.
    const SAMPLE_POINTS: [(f64, f64); 5] = [
        (0.7137, 1.2911),
        (2.3049, -0.8513),
        (-1.4471, 0.6301),
        (3.7219, 2.2087),
        (0.1303, -2.9411),
    ];

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// `simplify_egraph(e)` must be *semantically* equal to `e`.
        ///
        /// This is the property that would have caught the `Node::Unsupported
        /// → (Num 0)` bug: `x + 1/2 → x` disagrees numerically everywhere.
        #[test]
        fn egraph_simplify_preserves_value(ast in ast_strategy()) {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let expr = build(&ast, &pool, x, y);
            let simplified = simplify_egraph(expr, &pool).value;

            // Structural identity implies semantic identity.
            if simplified != expr {
                for (xv, yv) in SAMPLE_POINTS {
                    let mut env = HashMap::new();
                    env.insert(x, xv);
                    env.insert(y, yv);
                    let (Some(a), Some(b)) = (
                        eval_interp(expr, &env, &pool),
                        eval_interp(simplified, &env, &pool),
                    ) else {
                        continue;
                    };
                    // Undefined / overflowing points carry no information.
                    if !a.is_finite() || !b.is_finite() {
                        continue;
                    }
                    let tol = 1e-6 * (1.0 + a.abs().max(b.abs()));
                    prop_assert!(
                        (a - b).abs() <= tol,
                        "at (x, y) = ({xv}, {yv}): {} = {a} but simplify_egraph gave {} = {b}",
                        pool.display(expr),
                        pool.display(simplified),
                    );
                }
            }
        }

        /// The single-buffer tree writer is byte-for-byte the old
        /// `format!`-folded encoder whenever the input is tree-shaped.
        #[test]
        fn tree_encoding_is_byte_identical(ast in ast_strategy()) {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let expr = build(&ast, &pool, x, y);
            let (lets, root) = backend::encode_for_test(&pool, expr);
            if lets.is_empty() {
                prop_assert_eq!(root, backend::encode_tree_reference_for_test(&pool, expr));
            }
        }

        /// Differential: the DAG-sized pipeline (let-bound sharing, cached
        /// prelude, `TermDag` decoding, memoised post-passes) returns exactly
        /// the `ExprId` the old tree-text pipeline did — on plain inputs and
        /// on inputs built to share interior subterms.
        #[test]
        fn dag_pipeline_matches_tree_reference(
            a in ast_strategy(),
            b in ast_strategy(),
            shape in 0u8..4,
            disjoint in any::<bool>(),
            trig in any::<bool>(),
        ) {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let ea = build(&a, &pool, x, y);
            let eb = build(&b, &pool, x, y);
            let expr = match shape {
                0 => ea,
                // `ea` reached along two parent edges.
                1 => pool.add(vec![pool.mul(vec![ea, eb]), pool.func("sin", vec![ea])]),
                2 => pool.mul(vec![pool.func("sin", vec![ea]), pool.func("cos", vec![ea])]),
                _ => pool.add(vec![
                    pool.pow(pool.add(vec![ea, eb]), pool.integer(2_i32)),
                    pool.mul(vec![pool.integer(-1_i32), pool.add(vec![ea, eb])]),
                    ea,
                ]),
            };
            let config = EgraphConfig {
                disjoint_schedule: disjoint,
                include_trig_rules: trig,
                ..EgraphConfig::default()
            };
            let new = simplify_egraph_with(expr, &pool, &config, &SizeCost).value;
            let old = backend::simplify_egraph_reference(expr, &pool, &config);
            prop_assert_eq!(
                new,
                old,
                "{}: new {} vs reference {}",
                pool.display(expr),
                pool.display(new),
                pool.display(old)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// DAG-sized encoding, prelude cache and node-limit guard
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "egraph"))]
mod dag_tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use std::time::{Duration, Instant};

    /// `e_{k+1} = sin(e_k)·cos(e_k)`: `3k + 3` distinct nodes, `~2^k` as a tree.
    fn sin_cos_dag(pool: &ExprPool, depth: usize) -> ExprId {
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let mut e = pool.add(vec![x, y]);
        for _ in 0..depth {
            e = pool.mul(vec![pool.func("sin", vec![e]), pool.func("cos", vec![e])]);
        }
        e
    }

    /// Chebyshev recurrence `T_{n+1} = 2x·T_n − T_{n−1}`, unexpanded: every
    /// `T_n` is shared by the next two.
    fn chebyshev_dag(pool: &ExprPool, n: usize) -> ExprId {
        let x = pool.symbol("x", Domain::Real);
        let (mut prev, mut cur) = (pool.integer(1_i32), x);
        for _ in 1..n {
            let next = pool.add(vec![
                pool.mul(vec![pool.integer(2_i32), x, cur]),
                pool.mul(vec![pool.integer(-1_i32), prev]),
            ]);
            prev = cur;
            cur = next;
        }
        cur
    }

    fn dag_size(pool: &ExprPool, root: ExprId) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            match pool.get(id) {
                crate::kernel::ExprData::Add(a) | crate::kernel::ExprData::Mul(a) => {
                    stack.extend(a)
                }
                crate::kernel::ExprData::Func { args, .. } => stack.extend(args),
                crate::kernel::ExprData::Pow { base, exp } => stack.extend([base, exp]),
                _ => {}
            }
        }
        seen.len()
    }

    /// Depth 30 is a ~2³⁰-node tree. Before the DAG encoder, depth 13 already
    /// took ~190 ms and each level multiplied that by ~4.
    #[test]
    fn deep_sin_cos_dag_is_polynomial() {
        let pool = ExprPool::new();
        let e = sin_cos_dag(&pool, 30);
        let t = Instant::now();
        let out = simplify_egraph(e, &pool).value;
        let took = t.elapsed();
        assert!(
            took < Duration::from_secs(1),
            "depth-30 sin·cos DAG took {took:?}"
        );
        // Nothing to simplify: the input comes back as itself.
        assert_eq!(out, e);
    }

    #[test]
    fn deep_chebyshev_dag_is_polynomial() {
        let pool = ExprPool::new();
        let e = chebyshev_dag(&pool, 40);
        let t = Instant::now();
        let out = simplify_egraph(e, &pool).value;
        let took = t.elapsed();
        assert!(took < Duration::from_secs(1), "T_40 DAG took {took:?}");
        assert!(dag_size(&pool, out) <= 2 * dag_size(&pool, e));
    }

    /// The shared encoding still simplifies *inside* shared subterms, and
    /// the rewrite reaches every parent: `s = x + 0` appears under both
    /// `sin` and `cos`.
    #[test]
    fn shared_subterm_simplifies_everywhere() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let s = pool.add(vec![x, pool.integer(0_i32)]);
        let mut e = s;
        for _ in 0..20 {
            e = pool.mul(vec![pool.func("sin", vec![e]), pool.func("cos", vec![e])]);
        }
        let mut want = x;
        for _ in 0..20 {
            want = pool.mul(vec![
                pool.func("sin", vec![want]),
                pool.func("cos", vec![want]),
            ]);
        }
        assert_eq!(simplify_egraph(e, &pool).value, want);
    }

    /// A shared `sin²+cos²` collapses to 1 in every position.
    #[test]
    fn shared_trig_identity_collapses() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let one_ish = pool.add(vec![
            pool.pow(pool.func("sin", vec![x]), two),
            pool.pow(pool.func("cos", vec![x]), two),
        ]);
        let e = pool.add(vec![
            pool.func("tan", vec![one_ish]),
            pool.pow(one_ish, two),
        ]);
        let want = pool.add(vec![
            pool.func("tan", vec![pool.integer(1_i32)]),
            pool.integer(1_i32),
        ]);
        assert_eq!(simplify_egraph(e, &pool).value, want);
    }

    /// The node-limit guard counts distinct nodes, and that is now the real
    /// size of the work: a 120-node DAG whose tree has ~2³⁹ nodes both
    /// passes a limit of 200 *and* finishes promptly…
    #[test]
    fn node_limit_bounds_actual_work() {
        let pool = ExprPool::new();
        let e = sin_cos_dag(&pool, 39);
        let n = dag_size(&pool, e);
        assert!(n <= 125, "{n}");
        let config = EgraphConfig {
            node_limit: Some(200),
            ..EgraphConfig::default()
        };
        let t = Instant::now();
        let r = simplify_egraph_with(e, &pool, &config, &SizeCost);
        assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
        assert_eq!(r.value, e);
        assert!(r
            .log
            .steps()
            .iter()
            .all(|s| s.rule_name != "egraph_node_limit_exceeded"));

        // …while a limit below the DAG size still refuses it up front.
        let tight = EgraphConfig {
            node_limit: Some(n - 1),
            ..EgraphConfig::default()
        };
        let r = simplify_egraph_with(e, &pool, &tight, &SizeCost);
        assert_eq!(r.value, e);
        assert!(r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "egraph_node_limit_exceeded"));
    }

    /// The prelude cache is keyed by everything that changes the program,
    /// and a cached e-graph carries nothing from one call into the next.
    #[test]
    fn prelude_cache_does_not_leak_between_configs_or_calls() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let pyth = pool.add(vec![
            pool.pow(pool.func("sin", vec![x]), two),
            pool.pow(pool.func("cos", vec![x]), two),
        ]);
        let one = pool.integer(1_i32);
        let on = EgraphConfig::default();
        let off = EgraphConfig {
            include_trig_rules: false,
            ..EgraphConfig::default()
        };
        let joint = EgraphConfig {
            disjoint_schedule: false,
            ..EgraphConfig::default()
        };
        let joint_off = EgraphConfig {
            disjoint_schedule: false,
            include_trig_rules: false,
            ..EgraphConfig::default()
        };
        for _ in 0..3 {
            assert_eq!(simplify_egraph_with(pyth, &pool, &on, &SizeCost).value, one);
            assert_ne!(
                simplify_egraph_with(pyth, &pool, &off, &SizeCost).value,
                one
            );
            assert_eq!(
                simplify_egraph_with(pyth, &pool, &joint, &SizeCost).value,
                one
            );
            assert_ne!(
                simplify_egraph_with(pyth, &pool, &joint_off, &SizeCost).value,
                one
            );
        }
        // An unrelated expression right after must not see the previous
        // call's `__expr` or e-classes: `x + 0` is still just `x`, and a
        // bare `sin(x)` is not unified with anything.
        let x0 = pool.add(vec![x, pool.integer(0_i32)]);
        assert_eq!(simplify_egraph(x0, &pool).value, x);
        let sx = pool.func("sin", vec![x]);
        assert_eq!(simplify_egraph(sx, &pool).value, sx);
    }

    /// Timing harness for the PR description; run with
    /// `cargo test --release --features egraph -- --ignored egraph_dag_timings --nocapture`.
    #[test]
    #[ignore]
    fn egraph_dag_timings() {
        fn best<F: FnMut()>(reps: usize, mut f: F) -> f64 {
            let mut b = f64::MAX;
            for _ in 0..reps {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b * 1e3
        }
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e0 = pool.add(vec![x, pool.integer(0_i32)]);
        let t = best(50, || {
            std::hint::black_box(simplify_egraph(e0, &pool));
        });
        println!("x + 0: {t:.3} ms");
        let two = pool.integer(2_i32);
        let pyth = pool.add(vec![
            pool.pow(pool.func("sin", vec![x]), two),
            pool.pow(pool.func("cos", vec![x]), two),
        ]);
        let t = best(20, || {
            std::hint::black_box(simplify_egraph(pyth, &pool));
        });
        println!("sin^2 x + cos^2 x: {t:.3} ms");
        for k in [8usize, 10, 12, 20, 30] {
            let e = sin_cos_dag(&pool, k);
            let t = best(3, || {
                std::hint::black_box(simplify_egraph(e, &pool));
            });
            println!(
                "sin·cos DAG depth {k} ({} nodes): {t:.3} ms",
                dag_size(&pool, e)
            );
        }
        for n in [8usize, 10, 12, 14, 20, 40] {
            let e = chebyshev_dag(&pool, n);
            let t = best(3, || {
                std::hint::black_box(simplify_egraph(e, &pool));
            });
            println!(
                "Chebyshev DAG n={n} ({} nodes): {t:.3} ms",
                dag_size(&pool, e)
            );
        }
    }
}
