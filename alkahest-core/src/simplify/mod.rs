pub mod assumptions;
pub mod colored_egraph;
pub mod depth;
pub mod discrimination_net;
#[cfg(feature = "parallel")]
pub mod dispatch;
pub mod egraph;
pub mod engine;
#[cfg(feature = "parallel")]
pub mod parallel;
#[cfg(feature = "parallel")]
pub mod redex;
pub mod rules;
pub mod rulesets;
pub(crate) mod stack;

#[cfg(test)]
mod proptests;

pub use assumptions::{simplify_with_assumptions, AssumptionContext, AssumptionError};
pub use colored_egraph::{
    assumptions_satisfy, simplify_colored, ColorId, ColoredEgraph, CONTEXT_COLOR, ROOT_COLOR,
};
pub use depth::check_simplify_depth;
pub use discrimination_net::{expr_head, pattern_head, DiscriminationIndex, PatternHead};
pub use egraph::{
    simplify_egraph, simplify_egraph_with, DepthCost, EgraphConfig, EgraphCost, NoncommutativeCost,
    OpCost, SizeCost, StabilityCost,
};
pub use engine::{
    rules_for_config, simplify, simplify_batch, simplify_expanded, simplify_log_exp,
    simplify_trig_normal_form, simplify_with, simplify_with_pattern_rules, SimplifyConfig,
};
pub use rules::{node_kind, NodeKinds, RewriteRule};
pub use rulesets::{PatternRule, PatternRuleSet};
