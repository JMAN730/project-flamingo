//! Indirect execution of multi-argument command signatures (#757): the
//! title's root-constant arguments before a draw or dispatch, mapped onto
//! the host root parameters that carry each stage's user data. Pure
//! planning, so it is tested everywhere; [`crate::indirect`] builds the host
//! signature.
//!
//! shortcut: only root constants are mapped; a root view argument, or a
//! constant two stages read, still drops the execution.
//!
//! From upstream PR #956 (Noname0202, 71967c10), hand-ported.

/// One argument of a host command signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostIndirectArg {
    Constant { param: u32, offset: u32, count: u32 },
    Execute(u32),
}

/// One stage's user-data root-constant parameter, for mapping title indirect
/// root constants onto our host root signature.
pub(crate) struct StageUserData<'a> {
    pub(crate) param: u32,
    pub(crate) dwords: u32,
    pub(crate) entries: &'a [victory_shader::emit::UserData],
}

/// Host command-signature layout and trailing draw/dispatch offset for an
/// indirect execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndirectPlan {
    /// Final execution kind (PC numbering: 0 draw, 1 indexed, 2 dispatch, 10 mesh).
    pub(crate) kind: u32,
    /// Byte offset of the final draw/dispatch argument inside each record.
    pub(crate) arg_offset: u32,
    /// Host arguments in record order (0 or more `Constant`s, then `Execute(kind)`).
    pub(crate) args: Vec<HostIndirectArg>,
}

/// Maps a title command signature (`stride`, `title_args`) onto `stages`'s
/// host user-data parameters. Supports single-argument draw/dispatch/mesh
/// signatures and multi-argument signatures whose prefix arguments are root
/// constants (`D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT`).
pub(crate) fn plan_indirect(
    stride: u32,
    title_args: &[[u32; 4]],
    stages: &[StageUserData<'_>],
) -> Option<IndirectPlan> {
    const DRAW: u32 = 0;
    const INDEXED: u32 = 1;
    const DISPATCH: u32 = 2;
    const CONSTANT: u32 = 5;
    const MESH: u32 = 10;
    const DISPATCH_SINGLE: u32 = 105;

    let (last, prefix) = title_args.split_last()?;
    let kind = if last[0] == DISPATCH_SINGLE {
        DISPATCH
    } else {
        last[0]
    };
    let exec_bytes = match kind {
        DRAW => 16,
        INDEXED => 20,
        DISPATCH | MESH => 12,
        _ => return None,
    };
    if prefix.is_empty() {
        return Some(IndirectPlan {
            kind,
            arg_offset: 0,
            args: vec![HostIndirectArg::Execute(kind)],
        });
    }

    let mut host_args = Vec::with_capacity(title_args.len());
    let mut arg_offset = 0u32;
    for arg in prefix {
        if arg[0] != CONSTANT {
            return None;
        }
        let title_slot = u8::try_from(arg[1]).ok()?;
        let dest_off = arg[2];
        let num = arg[3];
        if num == 0 {
            return None;
        }
        let end_dword = dest_off.checked_add(num)?;

        // Find every stage/index reading any dword in [dest_off, dest_off + num).
        let mut target_stage: Option<usize> = None;
        let mut base_index: Option<u32> = None;
        for (si, stage) in stages.iter().enumerate() {
            for u in stage.entries {
                if u.root == victory_shader::RootType::Constants
                    && u.slot == title_slot
                    && u.dword >= dest_off
                    && u.dword < end_dword
                {
                    let k = u.dword - dest_off;
                    let candidate = u.index.checked_sub(k)?;
                    match (target_stage, base_index) {
                        (None, None) => {
                            target_stage = Some(si);
                            base_index = Some(candidate);
                        }
                        (Some(s), Some(b)) if s == si && b == candidate => {}
                        _ => return None,
                    }
                }
            }
        }

        let mapped = if let (Some(si), Some(base)) = (target_stage, base_index) {
            let stage = &stages[si];
            let end_idx = base.checked_add(num)?;
            if end_idx > stage.dwords {
                return None;
            }
            // No other user-data entry may live in [base, base + num).
            let collides = stage.entries.iter().any(|u| {
                u.index >= base
                    && u.index < end_idx
                    && !(u.root == victory_shader::RootType::Constants
                        && u.slot == title_slot
                        && u.dword == dest_off + (u.index - base))
            });
            if collides {
                return None;
            }
            HostIndirectArg::Constant {
                param: stage.param,
                offset: base,
                count: num,
            }
        } else {
            // No stage reads these constants; keep record alignment by landing
            // them in an unused contiguous range of some stage's user data.
            let mut free = None;
            'find: for stage in stages {
                if num > stage.dwords {
                    continue;
                }
                for base in 0..=stage.dwords - num {
                    let end_idx = base + num;
                    let used_by_shader = stage
                        .entries
                        .iter()
                        .any(|u| u.index >= base && u.index < end_idx);
                    let used_by_prev = host_args.iter().any(|a| match *a {
                        HostIndirectArg::Constant {
                            param,
                            offset,
                            count,
                        } => param == stage.param && offset < end_idx && offset + count > base,
                        HostIndirectArg::Execute(_) => false,
                    });
                    if !used_by_shader && !used_by_prev {
                        free = Some(HostIndirectArg::Constant {
                            param: stage.param,
                            offset: base,
                            count: num,
                        });
                        break 'find;
                    }
                }
            }
            free?
        };

        host_args.push(mapped);
        arg_offset = arg_offset.checked_add(num.checked_mul(4)?)?;
    }
    if arg_offset.checked_add(exec_bytes)? > stride {
        return None;
    }
    host_args.push(HostIndirectArg::Execute(kind));
    Some(IndirectPlan {
        kind,
        arg_offset,
        args: host_args,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::RootType;
    use victory_shader::emit::UserData;

    #[test]
    fn plan_indirect_maps_root_constants_and_rejects_unsupported() {
        let entries = [
            UserData {
                index: 2,
                slot: 1,
                root: RootType::Constants,
                dword: 0,
            },
            UserData {
                index: 3,
                slot: 1,
                root: RootType::Constants,
                dword: 1,
            },
            UserData {
                index: 4,
                slot: 0,
                root: RootType::Table,
                dword: 0,
            },
        ];
        let stages = [StageUserData {
            param: 0,
            dwords: 8,
            entries: &entries,
        }];

        // Single-argument dispatch stays at offset 0.
        let single = plan_indirect(12, &[[2, 0, 0, 0]], &stages).unwrap();
        assert_eq!(single.kind, 2);
        assert_eq!(single.arg_offset, 0);
        assert_eq!(single.args, vec![HostIndirectArg::Execute(2)]);

        // Console DISPATCH_SINGLE (105) normalizes to DISPATCH (2).
        let single_console = plan_indirect(12, &[[105, 0, 0, 0]], &stages).unwrap();
        assert_eq!(single_console.kind, 2);

        // Two root constants followed by Dispatch (stride 20 = 8 + 12).
        let multi = plan_indirect(20, &[[5, 1, 0, 2], [2, 0, 0, 0]], &stages).unwrap();
        assert_eq!(multi.kind, 2);
        assert_eq!(multi.arg_offset, 8);
        assert_eq!(
            multi.args,
            vec![
                HostIndirectArg::Constant {
                    param: 0,
                    offset: 2,
                    count: 2,
                },
                HostIndirectArg::Execute(2),
            ]
        );

        // Unused constant lands in an unoccupied slot (0..1) without colliding with entries.
        let unused = plan_indirect(16, &[[5, 7, 0, 1], [2, 0, 0, 0]], &stages).unwrap();
        assert_eq!(unused.arg_offset, 4);
        assert_eq!(
            unused.args[0],
            HostIndirectArg::Constant {
                param: 0,
                offset: 0,
                count: 1,
            }
        );

        // Collision with an occupied table slot at index 4 is rejected.
        assert!(plan_indirect(24, &[[5, 1, 0, 3], [2, 0, 0, 0]], &stages).is_none());

        // Stride smaller than arg_offset + exec_bytes is rejected.
        assert!(plan_indirect(16, &[[5, 1, 0, 2], [2, 0, 0, 0]], &stages).is_none());

        // Non-constant prefix argument (e.g. CBV = 6) is rejected.
        assert!(plan_indirect(20, &[[6, 0, 0, 0], [2, 0, 0, 0]], &stages).is_none());

        // Same constant read by two different stages cannot map to one host parameter.
        let stage2 = [
            StageUserData {
                param: 0,
                dwords: 4,
                entries: &entries[..1],
            },
            StageUserData {
                param: 1,
                dwords: 4,
                entries: &entries[..1],
            },
        ];
        assert!(plan_indirect(16, &[[5, 1, 0, 1], [2, 0, 0, 0]], &stage2).is_none());
    }
}
