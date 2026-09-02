//! Monte Carlo tree search with a policy-value network at the leaves.
//!
//! Phase 3 of Minihouse Zero. The tree lives here and the network lives in Python: Rust
//! descends until it has a batch of leaves that need evaluating, hands their encoded
//! planes across one call, and takes back priors and values. That inversion is the whole
//! design (see docs/ZERO.md) -- Rust threads calling *into* torch would serialise on the
//! GIL and give up the parallelism they were spawned for.
//!
//! The batch comes from **virtual loss**: a descent that reaches an unevaluated leaf
//! marks the path as if it had lost, so the next descent in the same `collect` goes
//! somewhere else. Without it every descent in a batch returns the same leaf and the
//! batch is worth one simulation.
//!
//! Nodes and edges are flat `Vec`s addressed by index. No `Rc`, no `RefCell`, and no
//! `GameState` per node: a descent walks a single scratch board with `make_ai_move` and
//! rewinds it with `undo_ai_move`, which is why a simulation costs two cheap move
//! applications per ply rather than a board clone.
//!
//! **Values are always in the frame of the side to move at that node**, matching the
//! network's own convention, so `backup` flips sign at every step up the path.

use crate::encode;
use crate::gamestate::GameState;
use crate::types::*;

pub const NO_NODE: u32 = u32::MAX;

/// Prior mass given to a child that has never been visited. AlphaZero's original choice
/// is 0 -- neutral -- which is what this is; a first-play-urgency reduction relative to
/// the parent is the usual refinement and belongs with tuning, not with correctness.
pub const DEFAULT_FPU: f32 = 0.0;
pub const DEFAULT_C_PUCT: f32 = 1.5;

#[derive(Clone, Copy)]
pub struct Config {
    pub c_puct: f32,
    pub fpu: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config { c_puct: DEFAULT_C_PUCT, fpu: DEFAULT_FPU }
    }
}

#[derive(Clone, Copy)]
pub struct Edge {
    pub action: u32,
    pub mv: Move,
    pub prior: f32,
    pub child: u32,
}

pub struct Node {
    pub edge_start: u32,
    pub edge_len: u32,
    pub visits: u32,
    pub value_sum: f64,
    pub virtual_loss: u32,
    pub expanded: bool,
    pub pending: bool,
    /// Some(v) when the position is over: v is the result for the side to move here.
    pub terminal: Option<f32>,
}

impl Node {
    fn new() -> Node {
        Node {
            edge_start: 0,
            edge_len: 0,
            visits: 0,
            value_sum: 0.0,
            virtual_loss: 0,
            expanded: false,
            pending: false,
            terminal: None,
        }
    }
}

/// A leaf waiting for the network. `moves` is carried rather than recomputed: expanding
/// it later would otherwise mean walking back down to the position to regenerate them.
pub struct Pending {
    pub node: u32,
    pub path: Vec<u32>,
    pub moves: Vec<Move>,
    pub side: Color,
}

pub struct Mcts {
    root: GameState,
    scratch: GameState,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pending: Vec<Pending>,
    cfg: Config,
    simulations: usize,
}

impl Mcts {
    pub fn new(position: &GameState, cfg: Config) -> Mcts {
        let mut root = position.clone();
        // Pins where the game's history stops and the tree's begins, so a repetition
        // inside the tree is scored against the same rule the alpha-beta search uses.
        root.set_search_root();
        let scratch = root.clone();
        let mut tree = Mcts {
            root,
            scratch,
            nodes: vec![Node::new()],
            edges: Vec::new(),
            pending: Vec::new(),
            cfg,
            simulations: 0,
        };
        // The root needs its terminal verdict like any other node. Only child nodes get
        // one during a descent, so without this a search started from a finished game
        // expands a node with no edges and every later descent selects out of an empty
        // range.
        tree.nodes[0].terminal = Self::terminal_value(&mut tree.scratch);
        tree
    }

    pub fn root_is_terminal(&self) -> bool {
        self.nodes[0].terminal.is_some()
    }

    pub fn simulations(&self) -> usize {
        self.simulations
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// The result for the side to move, if this position is over. `None` if play continues.
    fn terminal_value(gs: &mut GameState) -> Option<f32> {
        if gs.get_legal_moves_vec().is_empty() {
            // Mated or stalemated: the side to move has lost or drawn, never won.
            return Some(if gs.side_to_move_in_check() { -1.0 } else { 0.0 });
        }
        if gs.is_terminal_draw() {
            return Some(0.0);
        }
        None
    }

    /// PUCT: argmax over Q + c_puct * P * sqrt(N_parent) / (1 + N_child).
    ///
    /// A child's statistics are in the *child's* frame, so Q as the parent sees it is
    /// their negation. Virtual loss is added to the child's visit count and to its score,
    /// which is a win for the child and therefore a loss for the parent doing the
    /// choosing -- exactly the discouragement it is there to provide.
    fn select(&self, node: u32) -> u32 {
        let n = &self.nodes[node as usize];
        debug_assert!(n.edge_len > 0, "selecting out of an expanded node with no moves");
        let parent_n = (n.visits + n.virtual_loss).max(1) as f32;
        let sqrt_parent = parent_n.sqrt();

        let mut best = n.edge_start;
        let mut best_score = f32::NEG_INFINITY;

        for i in n.edge_start..n.edge_start + n.edge_len {
            let e = &self.edges[i as usize];
            let (q, child_n) = if e.child == NO_NODE {
                (self.cfg.fpu, 0u32)
            } else {
                let c = &self.nodes[e.child as usize];
                let visits = c.visits + c.virtual_loss;
                if visits == 0 {
                    (self.cfg.fpu, 0)
                } else {
                    let w = c.value_sum + c.virtual_loss as f64;
                    (-(w / visits as f64) as f32, visits)
                }
            };
            let u = self.cfg.c_puct * e.prior * sqrt_parent / (1.0 + child_n as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    fn backup(&mut self, path: &[u32], leaf_value: f32) {
        let mut v = leaf_value;
        for &node in path.iter().rev() {
            let n = &mut self.nodes[node as usize];
            n.visits += 1;
            n.value_sum += v as f64;
            v = -v;
        }
        self.simulations += 1;
    }

    fn apply_virtual_loss(&mut self, path: &[u32]) {
        for &node in path {
            self.nodes[node as usize].virtual_loss += 1;
        }
    }

    fn clear_virtual_loss(&mut self, path: &[u32]) {
        for &node in path {
            let n = &mut self.nodes[node as usize];
            n.virtual_loss = n.virtual_loss.saturating_sub(1);
        }
    }

    /// Descend until a batch of `max_leaves` unevaluated positions has been collected,
    /// resolving terminal positions in place. Returns their encoded planes, one
    /// `encode::INPUT_SIZE` block each, in the order `expand` must answer them.
    pub fn collect(&mut self, max_leaves: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(max_leaves * encode::INPUT_SIZE);
        if self.root_is_terminal() {
            return out;
        }

        // A descent that ends on a terminal position backs up without adding to the
        // batch, so `pending.len()` is not on its own a guarantee of progress: a node
        // whose every move ends the game would loop here forever. The budget bounds the
        // work per call instead. Simulations still accumulate, so the caller's own
        // "have I run enough" condition is what ends the search.
        let mut budget = max_leaves.saturating_mul(4) + 32;

        while self.pending.len() < max_leaves && budget > 0 {
            budget -= 1;
            let mut path = vec![0u32];
            let mut node = 0u32;
            let mut depth = 0usize;
            let mut resolved: Option<f32> = None;

            loop {
                let current = &self.nodes[node as usize];
                if let Some(v) = current.terminal {
                    resolved = Some(v);
                    break;
                }
                if current.pending {
                    // Another descent in this batch is already waiting on this leaf and
                    // virtual loss did not steer us away -- usually a node with very few
                    // children. Abandoning the descent is right: adding it twice would
                    // have the network answer the same question twice and would double
                    // count its backup.
                    break;
                }
                if !current.expanded {
                    break;
                }

                let edge_idx = self.select(node);
                let edge = self.edges[edge_idx as usize];
                self.scratch.make_ai_move(edge.mv);
                depth += 1;

                if edge.child == NO_NODE {
                    self.nodes.push(Node::new());
                    let fresh = (self.nodes.len() - 1) as u32;
                    self.edges[edge_idx as usize].child = fresh;
                    node = fresh;
                    path.push(node);
                    // A new node's terminal status is decided once, here, and never
                    // costs the network an evaluation.
                    self.nodes[node as usize].terminal = Self::terminal_value(&mut self.scratch);
                    if let Some(v) = self.nodes[node as usize].terminal {
                        resolved = Some(v);
                    }
                    break;
                }
                node = edge.child;
                path.push(node);
            }

            let stalled = resolved.is_none() && self.nodes[node as usize].pending;

            if let Some(v) = resolved {
                self.backup(&path, v);
            } else if !stalled {
                let moves = self.scratch.get_legal_moves_vec();
                let side = self.scratch.current_turn;
                let at = out.len();
                out.resize(at + encode::INPUT_SIZE, 0.0);
                encode::encode_position_into(&self.scratch, &mut out[at..]);
                self.nodes[node as usize].pending = true;
                self.apply_virtual_loss(&path);
                self.pending.push(Pending { node, path, moves, side });
            }

            for _ in 0..depth {
                self.scratch.undo_ai_move();
            }

            if stalled {
                break;
            }
        }
        out
    }

    /// Answer the leaves from the last `collect`. `priors` is one row of
    /// `encode::ACTION_SPACE` per leaf, already masked and normalised over that
    /// position's legal actions; `values` is one scalar per leaf in its own frame.
    pub fn expand(&mut self, priors: &[f32], values: &[f32]) -> Result<(), String> {
        if values.len() != self.pending.len() {
            return Err(format!(
                "expand got {} values for {} pending leaves",
                values.len(),
                self.pending.len()
            ));
        }
        if priors.len() != self.pending.len() * encode::ACTION_SPACE {
            return Err(format!(
                "expand got {} priors, expected {} ({} leaves x {})",
                priors.len(),
                self.pending.len() * encode::ACTION_SPACE,
                self.pending.len(),
                encode::ACTION_SPACE
            ));
        }

        let batch: Vec<Pending> = std::mem::take(&mut self.pending);
        for (i, leaf) in batch.into_iter().enumerate() {
            let row = &priors[i * encode::ACTION_SPACE..(i + 1) * encode::ACTION_SPACE];
            let start = self.edges.len() as u32;
            let mut total = 0.0f32;

            for &mv in &leaf.moves {
                let action = encode::move_to_index(mv, leaf.side)
                    .ok_or_else(|| format!("legal move {:?} has no action index", mv))?;
                let prior = row[action].max(0.0);
                total += prior;
                self.edges.push(Edge { action: action as u32, mv, prior, child: NO_NODE });
            }

            // A uniform fallback rather than a panic: a network that has put no mass on
            // any legal move is wrong, but a search that divides by zero is worse.
            let len = leaf.moves.len() as u32;
            if total > 1e-8 {
                for e in &mut self.edges[start as usize..] {
                    e.prior /= total;
                }
            } else {
                let uniform = 1.0 / len as f32;
                for e in &mut self.edges[start as usize..] {
                    e.prior = uniform;
                }
            }

            {
                let n = &mut self.nodes[leaf.node as usize];
                n.edge_start = start;
                n.edge_len = len;
                n.expanded = true;
                n.pending = false;
            }

            self.clear_virtual_loss(&leaf.path);
            self.backup(&leaf.path, values[i]);
        }
        Ok(())
    }

    /// (action index, visit count, mean value) for every root move, best first.
    pub fn root_stats(&self) -> Vec<(u32, u32, f32)> {
        let root = &self.nodes[0];
        let mut out = Vec::with_capacity(root.edge_len as usize);
        for i in root.edge_start..root.edge_start + root.edge_len {
            let e = &self.edges[i as usize];
            let (visits, q) = if e.child == NO_NODE {
                (0, 0.0)
            } else {
                let c = &self.nodes[e.child as usize];
                if c.visits == 0 {
                    (0, 0.0)
                } else {
                    (c.visits, -(c.value_sum / c.visits as f64) as f32)
                }
            };
            out.push((e.action, visits, q));
        }
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }

    pub fn root_moves(&self) -> Vec<(Move, u32)> {
        let root = &self.nodes[0];
        (root.edge_start..root.edge_start + root.edge_len)
            .map(|i| {
                let e = &self.edges[i as usize];
                let visits = if e.child == NO_NODE {
                    0
                } else {
                    self.nodes[e.child as usize].visits
                };
                (e.mv, visits)
            })
            .collect()
    }

    /// The most-visited root move. Visit count, not value: it is the statistic the whole
    /// search concentrates, and it is what AlphaZero's policy target is built from.
    pub fn best_move(&self) -> Option<Move> {
        self.root_moves().into_iter().max_by_key(|&(_, v)| v).map(|(m, _)| m)
    }

    /// The root's own value, in the frame of the side to move there.
    pub fn root_value(&self) -> f32 {
        let root = &self.nodes[0];
        if root.visits == 0 {
            0.0
        } else {
            (root.value_sum / root.visits as f64) as f32
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_position(&self) -> &GameState {
        &self.root
    }
}
