//! Which host adapter runs the title.
//!
//! Translated programs need shader model 6.6 with resource binding tier 3 (descriptors are read
//! through the heap directly), and the world needs mesh shaders. An adapter that only reaches
//! feature level 12_0 starts the menus and then drops the world's draws, so the choice weighs
//! these capabilities rather than taking the first adapter that creates a device. Candidates
//! arrive in DXGI's high-performance order, which keeps a hybrid laptop on its discrete GPU
//! whenever that GPU qualifies.
//!
//! This module only decides; `host.rs` enumerates the adapters and probes them.

use std::fmt;

/// What one candidate adapter offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Caps {
    /// Shader model 6.6 or newer.
    pub shader_model_6_6: bool,
    /// Resource binding tier 3.
    pub binding_tier_3: bool,
    /// Any mesh shader tier.
    pub mesh_shaders: bool,
}

/// A capability the title needs that an adapter lacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lack {
    ShaderModel66,
    BindingTier3,
    MeshShaders,
}

impl fmt::Display for Lack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Lack::ShaderModel66 => "shader model 6.6",
            Lack::BindingTier3 => "resource binding tier 3",
            Lack::MeshShaders => "mesh shaders",
        })
    }
}

impl Caps {
    /// The capabilities missing from this adapter, in the order they matter.
    pub(crate) fn lacks(self) -> Vec<Lack> {
        let mut out = Vec::new();
        if !self.shader_model_6_6 {
            out.push(Lack::ShaderModel66);
        }
        if !self.binding_tier_3 {
            out.push(Lack::BindingTier3);
        }
        if !self.mesh_shaders {
            out.push(Lack::MeshShaders);
        }
        out
    }

    /// Ranks adapters that all lack something: the menus need shader model 6.6 and tier 3,
    /// the world also needs mesh shaders, so an adapter that runs the menus beats one that
    /// runs nothing.
    fn rank(self) -> u8 {
        let menus = self.shader_model_6_6 && self.binding_tier_3;
        u8::from(menus) * 2 + u8::from(menus && self.mesh_shaders)
    }
}

/// One adapter that created a device at feature level 12_0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Candidate {
    pub name: String,
    pub caps: Caps,
}

/// The adapter chosen and whether it can run the whole title.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Choice {
    /// Index into the candidate list.
    pub index: usize,
    /// What the chosen adapter lacks; empty when it runs the title.
    pub lacks: Vec<Lack>,
}

/// Chooses among `candidates`, given in preference order: the first that has every capability,
/// or, when none does, the best-ranked one (earliest on ties). `None` only for an empty list.
pub(crate) fn choose(candidates: &[Candidate]) -> Option<Choice> {
    let index = candidates
        .iter()
        .position(|c| c.caps.lacks().is_empty())
        .or_else(|| {
            // `max_by_key` keeps the last of equal maxima; reverse so the earliest wins.
            candidates
                .iter()
                .enumerate()
                .rev()
                .max_by_key(|(_, c)| c.caps.rank())
                .map(|(i, _)| i)
        })?;
    Some(Choice {
        index,
        lacks: candidates[index].caps.lacks(),
    })
}

/// The fields of the warning for a choice that cannot run the whole title.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Shortfall {
    /// The chosen adapter's name.
    pub chosen: String,
    /// What the chosen adapter lacks, comma-separated.
    pub lacks: String,
    /// Every candidate and what it lacks, so a report names the hardware gap instead of a
    /// dark world.
    pub adapters: String,
}

/// `None` when the chosen adapter runs the whole title.
pub(crate) fn shortfall(candidates: &[Candidate], choice: &Choice) -> Option<Shortfall> {
    if choice.lacks.is_empty() {
        return None;
    }
    let adapters = candidates
        .iter()
        .map(|c| format!("{} (lacks {})", c.name, join(&c.caps.lacks())))
        .collect::<Vec<_>>()
        .join("; ");
    Some(Shortfall {
        chosen: candidates[choice.index].name.clone(),
        lacks: join(&choice.lacks),
        adapters,
    })
}

fn join(lacks: &[Lack]) -> String {
    lacks
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: Caps = Caps {
        shader_model_6_6: true,
        binding_tier_3: true,
        mesh_shaders: true,
    };
    const MENUS: Caps = Caps {
        mesh_shaders: false,
        ..FULL
    };
    const NONE: Caps = Caps {
        shader_model_6_6: false,
        binding_tier_3: false,
        mesh_shaders: false,
    };

    fn list(caps: &[Caps]) -> Vec<Candidate> {
        caps.iter()
            .enumerate()
            .map(|(i, &caps)| Candidate {
                name: format!("gpu{i}"),
                caps,
            })
            .collect()
    }

    #[test]
    fn the_first_capable_adapter_wins_in_preference_order() {
        let c = list(&[FULL, FULL]);
        assert_eq!(
            choose(&c),
            Some(Choice {
                index: 0,
                lacks: vec![]
            })
        );
        assert_eq!(shortfall(&c, &choose(&c).unwrap()), None);
    }

    #[test]
    fn a_capable_adapter_beats_an_earlier_one_without_mesh_shaders() {
        // A preferred adapter that only runs the menus loses to a later one that runs the world.
        let c = list(&[MENUS, FULL]);
        assert_eq!(choose(&c).unwrap().index, 1);
    }

    #[test]
    fn without_a_capable_adapter_the_one_that_runs_most_is_used_and_named() {
        let c = list(&[NONE, MENUS, MENUS]);
        let choice = choose(&c).unwrap();
        assert_eq!(choice.index, 1, "earliest of the best-ranked");
        assert_eq!(choice.lacks, vec![Lack::MeshShaders]);
        let warning = shortfall(&c, &choice).unwrap();
        assert_eq!(warning.chosen, "gpu1");
        assert_eq!(warning.lacks, "mesh shaders");
        assert!(
            warning.adapters.starts_with(
                "gpu0 (lacks shader model 6.6, resource binding tier 3, mesh shaders); "
            ),
            "{}",
            warning.adapters
        );
    }

    #[test]
    fn mesh_shaders_without_the_menu_features_do_not_rank() {
        let mesh_only = Caps {
            mesh_shaders: true,
            ..NONE
        };
        assert_eq!(choose(&list(&[mesh_only, NONE])).unwrap().index, 0);
        assert_eq!(choose(&list(&[mesh_only, MENUS])).unwrap().index, 1);
    }

    #[test]
    fn an_empty_list_chooses_nothing() {
        assert_eq!(choose(&[]), None);
    }
}
