pub mod egraph;
pub mod engine;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod rules;
pub mod rulesets;

#[cfg(test)]
mod proptests;

pub use egraph::{
    simplify_egraph, simplify_egraph_with, DepthCost, EgraphConfig, EgraphCost, NoncommutativeCost,
    OpCost, SizeCost, StabilityCost,
};
pub use engine::{rules_for_config, simplify, simplify_expanded, simplify_with, SimplifyConfig};
pub use rules::RewriteRule;
pub use rulesets::PatternRule;
