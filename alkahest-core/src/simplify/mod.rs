pub mod colored_egraph;
pub mod discrimination_net;
pub mod egraph;
pub mod engine;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod rules;
pub mod rulesets;

#[cfg(test)]
mod proptests;

pub use colored_egraph::{
    assumptions_satisfy, simplify_colored, ColorId, ColoredEgraph, CONTEXT_COLOR, ROOT_COLOR,
};
pub use discrimination_net::{expr_head, pattern_head, DiscriminationIndex, PatternHead};
pub use egraph::{
    simplify_egraph, simplify_egraph_with, DepthCost, EgraphConfig, EgraphCost, NoncommutativeCost,
    OpCost, SizeCost, StabilityCost,
};
pub use engine::{
    rules_for_config, simplify, simplify_batch, simplify_expanded, simplify_with,
    simplify_with_pattern_rules, SimplifyConfig,
};
pub use rules::RewriteRule;
pub use rulesets::{PatternRule, PatternRuleSet};
