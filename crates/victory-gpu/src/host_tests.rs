use super::*;

#[test]
fn present_result_distinguishes_occlusion_from_success_and_failure() {
    assert!(present_result(HRESULT(0)).unwrap());
    assert!(!present_result(DXGI_STATUS_OCCLUDED).unwrap());
    assert_eq!(
        present_result(DXGI_ERROR_DEVICE_REMOVED)
            .unwrap_err()
            .code(),
        DXGI_ERROR_DEVICE_REMOVED
    );
}

#[test]
fn shadow_buffer_transitions_between_writes_and_reads() {
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    let index = D3D12_RESOURCE_STATE_INDEX_BUFFER;
    assert_eq!(shadow_transition(D3D12_RESOURCE_STATE_COMMON, uav), None);
    assert_eq!(shadow_transition(D3D12_RESOURCE_STATE_COMMON, srv), None);
    assert_eq!(shadow_transition(uav, srv), Some(srv));
    assert_eq!(shadow_transition(srv, index), Some(srv | index));
    assert_eq!(shadow_transition(srv | index, srv), None);
    assert_eq!(shadow_transition(srv | index, uav), Some(uav));
    assert_eq!(shadow_transition(uav, uav), None);
}

#[test]
fn warp_adapter_accepts_only_software() {
    // This probes DXGI without creating a D3D12 device or running a title.
    // SAFETY: plain DXGI adapter queries.
    let raw = unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return;
        };
        factory.EnumWarpAdapter::<IDXGIAdapter>()
    };
    let Ok(raw) = raw else { return };
    // SAFETY: a live adapter the factory just returned.
    let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
    eprintln!(
        "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
        desc.VendorId, desc.Flags
    );
    assert_eq!(
        warp_adapter().is_ok(),
        desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
    );
}

#[test]
fn adapter_name_stops_at_nul() {
    let raw = [b'R' as u16, b'T' as u16, b'X' as u16, 0, b'X' as u16];
    assert_eq!(adapter_name(&raw), "RTX");
    assert_eq!(adapter_name(&[b'X' as u16, b'e' as u16]), "Xe");
}

#[test]
fn adapter_filter_matches_case_insensitively() {
    assert!(adapter_matches("NVIDIA GeForce RTX 3070 Ti", "rtx 3070"));
    assert!(adapter_matches("Intel(R) Iris(R) Xe Graphics", "  INTEL "));
    assert!(!adapter_matches("Intel(R) Iris(R) Xe Graphics", "RTX"));
}

#[test]
fn unknown_adapter_filter_fails_instead_of_falling_back() {
    let err = pick_hardware_adapter(Some("__victory_no_such_adapter__")).unwrap_err();
    assert_eq!(err.code(), DXGI_ERROR_NOT_FOUND);
}

#[test]
fn default_adapter_is_hardware_when_present() {
    if let Ok(picked) = pick_hardware_adapter(None) {
        assert_eq!(picked.desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);
        // SAFETY: `GetAdapterLuid` is a plain getter on a live D3D12 device.
        let luid = unsafe { picked.device.GetAdapterLuid() };
        assert_eq!(
            (luid.LowPart, luid.HighPart),
            (
                picked.desc.AdapterLuid.LowPart,
                picked.desc.AdapterLuid.HighPart,
            ),
        );
    }
}

#[test]
fn only_textures_made_for_rendering_take_render_target_views() {
    let Some(host) = host() else { return };
    let make = |flags: D3D12_RESOURCE_FLAGS| {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: flags,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a full description and our out pointer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut texture,
                )
                .unwrap();
        }
        texture.unwrap()
    };
    assert!(renders_to(&make(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)));
    assert!(!renders_to(&make(D3D12_RESOURCE_FLAG_NONE)));
    // The view bound in such a texture's place is a legal one.
    host.rtv(None, Some(&null_target()));
    // SAFETY: a plain query.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
}

#[test]
fn retire_drops_immediately_when_no_epoch_is_active() {
    let _retire = isolated_retire_scope();
    let (counter, drops) = DropCounter::new();
    retire(counter);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn idle_list_does_not_block_retirement() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let idle = host.new_list().expect("host list");
    let (counter, drops) = DropCounter::new();
    retire(counter);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(idle);
}

#[test]
fn open_marked_list_pins_retired_objects_across_other_executions() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let open = host.new_list().expect("open list");
    open.mark_work();

    let (c1, d1) = DropCounter::new();
    retire(c1);
    assert_eq!(d1.load(Ordering::SeqCst), 0);

    // Submitting and executing a DIFFERENT list must not drop objects
    // retired while `open` is still active.
    let other = host.new_list().expect("other list");
    other.mark_work();
    other.close();
    host.execute(&[&other]);
    other.reset();
    drop(other);
    assert_eq!(
        d1.load(Ordering::SeqCst),
        0,
        "executing another list must not release an open list's epoch"
    );

    // Closing and resetting the open list releases its epoch.
    open.close();
    open.reset();
    assert_eq!(d1.load(Ordering::SeqCst), 1);

    // Marking work again and dropping the list also releases its epoch.
    open.mark_work();
    let (c2, d2) = DropCounter::new();
    retire(c2);
    assert_eq!(d2.load(Ordering::SeqCst), 0);
    drop(open);
    assert_eq!(d2.load(Ordering::SeqCst), 1);

    // A closed list pinned via begin_submit stays active across execution
    // until end_submit.
    let submitted = host.new_list().expect("submitted list");
    submitted.mark_work();
    submitted.begin_submit();
    let (c3, d3) = DropCounter::new();
    retire(c3);
    submitted.close();
    host.execute(&[&submitted]);
    assert_eq!(d3.load(Ordering::SeqCst), 0);
    submitted.end_submit();
    assert_eq!(d3.load(Ordering::SeqCst), 1);
}

#[test]
fn empty_execute_does_not_submit() {
    // D3D12 forbids ExecuteCommandLists with NumCommandLists == 0.
    let Some(host) = host() else {
        return;
    };
    let before = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.execute(&[]);
    let after = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(before, after, "empty execute must not bump the fence");
}

#[test]
fn unique_list_batches_splits_on_repeated_pointers() {
    assert!(unique_list_batches(&[]).is_empty());
    assert_eq!(unique_list_batches(&[1]), vec![0..1]);
    assert_eq!(unique_list_batches(&[1, 2, 3]), vec![0..3]);
    assert_eq!(unique_list_batches(&[1, 1]), vec![0..1, 1..2]);
    assert_eq!(unique_list_batches(&[1, 2, 1]), vec![0..2, 2..3]);
    assert_eq!(unique_list_batches(&[1, 2, 1, 2]), vec![0..2, 2..4]);
    assert_eq!(unique_list_batches(&[7, 7, 7]), vec![0..1, 1..2, 2..3]);
}

#[test]
fn duplicate_list_execute_splits_submissions() {
    // VIC-BT-001: the same HostList twice must not Reset its prelude
    // allocator while the first recording is still queued. Exact batch
    // counts are covered by `unique_list_batches` (process-wide
    // `submitted` races with parallel GPU tests).
    let Some(host) = host() else {
        return;
    };
    let list = host.new_list().expect("host list");
    list.close();
    let other = host.new_list().expect("other list");
    other.close();

    let before = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.execute(&[&list]);
    let after_one = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        after_one > before,
        "one list must wait at least one submission"
    );

    host.execute(&[&list, &other]);
    let after_unique = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        after_unique > after_one,
        "two distinct lists must submit at least once"
    );

    host.execute(&[&list, &list]);
    let after_dup = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    assert!(
        after_dup > after_unique,
        "a repeated HostList must submit again (separate waited batches)"
    );
    // SAFETY: plain device-removed query after the duplicate submission.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
}

#[test]
fn duplicate_list_execute_preserves_texture_state() {
    // Synthetic reproducer from VIC-BT-001: list L transitions a texture
    // (first != last). Submitting [L, L] must leave host state at `last`
    // and must not remove the device from a clobbered prelude barrier.
    let Some(host) = host() else {
        return;
    };
    let list = host.new_list().expect("host list");
    let null = std::ptr::null();
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4,
        Height: 4,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let iid = ID3D12Resource::IID;
    let mut tex = null;
    assert_eq!(
        crate::resource::create_committed(
            null,
            (&raw const heap).cast(),
            0,
            (&raw const desc).cast(),
            0,
            null.cast(),
            (&raw const iid).cast(),
            &mut tex,
        ),
        crate::object::S_OK
    );
    const COPY_DEST: u32 = 0x400;
    const RENDER_TARGET: u32 = 0x4;
    // Record real barriers so prelude first/last states match the GPU list.
    crate::list::ensure_state(&list, tex, COPY_DEST);
    crate::list::ensure_state(&list, tex, RENDER_TARGET);
    list.close();

    host.execute(&[&list, &list]);

    let copy = crate::resource::host_copy(tex).expect("host copy");
    assert_eq!(
        copy.state.load(Ordering::Relaxed),
        RENDER_TARGET,
        "duplicate execute must leave the texture at the list's end state"
    );
    // SAFETY: plain device-removed query after prelude barriers ran twice.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
}

#[test]
fn active_execution_epoch_pins_objects_retired_during_execute() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let list = host.new_list().expect("host list");
    let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let drops_during = drops.clone();
    // Use an unmarked deferred step without `begin_submit()` so `list`
    // itself pins no epoch; only `Host::execute`'s `exec_epoch` keeps the
    // object alive until execution finishes.
    list.defer_unmarked(move || {
        let counter = DropCounter(drops_during.clone());
        retire(counter);
        assert_eq!(drops_during.load(Ordering::SeqCst), 0);
    });
    assert!(!list.has_epoch());
    list.close();
    host.execute(&[&list]);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn stale_pinned_list_does_not_block_newer_born_retirements() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    // Simulate an early command list (e.g. oldest_epoch=2) that recorded
    // work once and stays unreset indefinitely.
    let stale = host.new_list().expect("stale list");
    stale.mark_work();

    // A resource born after `stale` last recorded work must drop
    // immediately once no newer epoch is active, without head-of-line
    // blocking behind `stale`.
    let born1 = birth_seq();
    let (c1, d1) = DropCounter::new();
    retire_born(c1, born1);
    assert_eq!(
        d1.load(Ordering::SeqCst),
        1,
        "stale pinned epoch must not block a resource created after its last work"
    );

    // A resource born before a newer active list must wait for that newer
    // list, then drop as soon as the newer list finishes -- still while
    // `stale` remains pinned.
    let born2 = birth_seq();
    let active = host.new_list().expect("active list");
    active.mark_work();
    let (c2, d2) = DropCounter::new();
    retire_born(c2, born2);
    assert_eq!(d2.load(Ordering::SeqCst), 0);
    active.close();
    active.reset();
    assert_eq!(
        d2.load(Ordering::SeqCst),
        1,
        "resource must drop once its own blocking epoch finishes"
    );

    // If the older open list subsequently records new work after `born3`,
    // its `work_seq` advances and `c3` MUST wait for `stale` to finish.
    let born3 = birth_seq();
    stale.mark_work();
    let (c3, d3) = DropCounter::new();
    retire_born(c3, born3);
    assert_eq!(
        d3.load(Ordering::SeqCst),
        0,
        "an open list that records work after birth_seq must pin that resource"
    );
    stale.close();
    stale.reset();
    assert_eq!(d3.load(Ordering::SeqCst), 1);
}

#[test]
fn resubmitted_closed_list_preserves_work_seq_until_second_completion() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let born = birth_seq();
    let list = host.new_list().expect("reused list");
    list.mark_work();
    list.close();

    // First submission completes and releases the epoch, but keeps
    // `work_seq` because the list was not reset.
    list.begin_submit();
    host.execute(&[&list]);
    list.end_submit();
    assert!(!list.has_epoch());

    // Second submission re-acquires an epoch preserving `work_seq >= born`.
    list.begin_submit();
    let (c, d) = DropCounter::new();
    retire_born(c, born);
    assert_eq!(
        d.load(Ordering::SeqCst),
        0,
        "resubmitted closed list must keep resources from its recording alive"
    );
    host.execute(&[&list]);
    assert_eq!(d.load(Ordering::SeqCst), 0);
    list.end_submit();
    assert_eq!(d.load(Ordering::SeqCst), 1);
}

#[test]
fn retire_executing_pins_only_active_execute_epochs_without_bumping_resource_seq() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let open = host.new_list().expect("open list");
    open.mark_work();

    let seq_before = RESOURCE_SEQ.load(Ordering::SeqCst);
    let (c_open, d_open) = DropCounter::new();
    retire_executing(c_open);
    assert_eq!(
        RESOURCE_SEQ.load(Ordering::SeqCst),
        seq_before,
        "retire_executing must not bump global RESOURCE_SEQ"
    );
    assert_eq!(
        d_open.load(Ordering::SeqCst),
        1,
        "newly opened list with finite work_seq must not pin retire_executing objects"
    );

    let exec_guard = crate::retire::acquire_exec_guard();
    let (c_exec, d_exec) = DropCounter::new();
    retire_executing(c_exec);
    assert_eq!(RESOURCE_SEQ.load(Ordering::SeqCst), seq_before);
    assert_eq!(
        d_exec.load(Ordering::SeqCst),
        0,
        "active Host::execute epoch (work_seq == u64::MAX) must pin retire_executing objects"
    );
    drop(exec_guard);
    assert_eq!(
        d_exec.load(Ordering::SeqCst),
        1,
        "finishing Host::execute epoch must drop pinned object"
    );

    drop(open);
}

#[test]
fn state_gen_bumps_only_on_new_or_changed_texture_states() {
    let Some(host) = host() else {
        return;
    };
    let list = host.new_list().expect("host list");
    assert_eq!(list.state_gen(), 0);
    let null = std::ptr::null();
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4,
        Height: 4,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let iid = ID3D12Resource::IID;
    let mut tex = null;
    assert_eq!(
        crate::resource::create_committed(
            null,
            (&raw const heap).cast(),
            0,
            (&raw const desc).cast(),
            0,
            null.cast(),
            (&raw const iid).cast(),
            &mut tex,
        ),
        crate::object::S_OK
    );

    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const COPY_SOURCE: u32 = 0x800;
    assert!(list.transition(tex, SHADER_RESOURCE).is_none());
    assert_eq!(list.state_gen(), 1);
    // Same state does not bump state_gen.
    assert!(list.transition(tex, SHADER_RESOURCE).is_none());
    assert_eq!(list.state_gen(), 1);
    // Changing state bumps state_gen and returns the prior state.
    assert!(list.transition(tex, COPY_SOURCE).is_some());
    assert_eq!(list.state_gen(), 2);

    list.close();
    list.reset();
    assert_eq!(list.state_gen(), 0);
    list.close();

    let release = crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(tex, 0x10);
    assert_eq!(release(tex), 0);
}

#[test]
fn host_list_reset_trims_excess_scratch_chunks() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let list = host.new_list().expect("host list");

    // Allocate 3 x 3 MiB so 3 distinct 4 MiB scratch chunks are created.
    for _ in 0..3 {
        let (gpu, ptr) = list.scratch(3 << 20).expect("scratch allocation");
        assert_ne!(gpu, 0);
        assert!(!ptr.is_null());
    }
    assert!(
        list.scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .chunks
            .len()
            >= 2,
        "multi-chunk scratch spike should allocate at least two chunks"
    );

    // Also spill past one rect counter chunk to verify counter truncation.
    for _ in 0..=RECT_COUNTER_CHUNK {
        assert!(list.rect_counter().is_some());
    }
    assert_eq!(
        list.rect_scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .counters
            .len(),
        2
    );

    let errors_before = host.d3d12_errors();
    list.close();
    list.reset();

    {
        let scratch = list.scratch.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            scratch.chunks.len(),
            1,
            "reset should keep at most one warm scratch chunk"
        );
        assert_eq!((scratch.current, scratch.used), (0, 0));
    }
    {
        let rect = list.rect_scratch.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            rect.counters.len(),
            1,
            "reset should keep at most one warm rect counter chunk"
        );
        assert_eq!((rect.counter_chunk, rect.counter_used), (0, 0));
    }

    // Subsequent allocations reuse the warm chunks and execute cleanly.
    let (gpu, ptr) = list.scratch(4096).expect("warm scratch allocation");
    assert_ne!(gpu, 0);
    assert!(!ptr.is_null());
    assert_eq!(
        list.scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .chunks
            .len(),
        1
    );
    assert!(list.rect_counter().is_some());
    assert_eq!(
        list.rect_scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .counters
            .len(),
        1
    );

    list.close();
    host.execute(&[&list]);
    assert_eq!(host.d3d12_errors(), errors_before);
}

#[test]
fn upload_scratch_aligns_placements_and_falls_back_when_oversized() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let list = host.new_list().expect("host list");
    let errors_before = host.d3d12_errors();

    // First 256-byte scratch leaves `used = 256`, which is not 512-byte aligned.
    let (gpu0, _) = list.scratch(256).expect("initial scratch");
    assert_eq!(gpu0 % 256, 0);

    let (buf1, at1, ptr1) = list
        .upload_scratch(128)
        .expect("aligned sub-allocated upload");
    assert_eq!(
        at1, 512,
        "texture placement offset must round up to 512 bytes"
    );
    assert_eq!(at1 % 512, 0);
    assert!(!ptr1.is_null());
    assert_eq!(
        list.deferred().uploads.len(),
        0,
        "sub-allocated staging must not allocate a dedicated upload buffer"
    );
    // SAFETY: `buf1` is a live resource and `ptr1` points to 256+ writable bytes.
    unsafe {
        assert_eq!(buf1.GetGPUVirtualAddress() + at1, gpu0 + at1);
        ptr1.write_bytes(0x5a, 128);
    }

    // Oversized copy falls back to a dedicated upload buffer at offset 0.
    let (buf_big, at_big, ptr_big) = list
        .upload_scratch(SCRATCH_CHUNK + 512)
        .expect("oversized fallback upload");
    assert_eq!(at_big, 0);
    assert!(!ptr_big.is_null());
    assert_ne!(buf_big.as_raw(), buf1.as_raw());
    assert_eq!(list.deferred().uploads.len(), 1);
    // SAFETY: `ptr_big` is mapped for at least `SCRATCH_CHUNK + 512` bytes.
    unsafe { ptr_big.write_bytes(0x3c, 16) };

    list.close();
    host.execute(&[&list]);
    assert_eq!(host.d3d12_errors(), errors_before);
}

#[test]
fn fill_scratch_deduplicates_until_copy_or_defer_run() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let list = host.new_list().expect("host list");
    let errors_before = host.d3d12_errors();
    let mut src = [0x11u8; 64];
    let mut dst = [0u8; 64];

    let gpu1 = list.fill_scratch(src.as_ptr(), 64).expect("first fill");
    let gpu2 = list.fill_scratch(src.as_ptr(), 64).expect("dedup fill");
    assert_eq!(gpu1, gpu2);
    assert_eq!(
        list.deferred()
            .copies
            .iter()
            .filter(|s| matches!(s, Step::Fill(..)))
            .count(),
        1
    );

    // Intervening title copy invalidates `filled` cache.
    list.defer_copy(dst.as_mut_ptr(), src.as_ptr(), 4);
    let gpu3 = list.fill_scratch(src.as_ptr(), 64).expect("post-copy fill");
    assert_ne!(gpu3, gpu1);

    // Intervening deferred run also invalidates `filled` cache.
    list.defer(|| {});
    src[0] = 0x22;
    let gpu4 = list.fill_scratch(src.as_ptr(), 64).expect("post-run fill");
    assert_ne!(gpu4, gpu3);

    list.close();
    host.execute(&[&list]);
    assert_eq!(host.d3d12_errors(), errors_before);
}

#[test]
fn defer_gpu_write_deduplicates_and_clears_on_copy() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let list = host.new_list().expect("host list");
    let errors_before = host.d3d12_errors();
    let mut backing = vec![0u8; 4096];
    let start = backing.as_mut_ptr() as u64;
    let addr = start as usize;
    crate::shadow::register(addr, start, backing.len() as u64);
    let shadow = crate::shadow::find(start, backing.len() as u64).expect("registered shadow");

    list.defer_gpu_write(&shadow, addr, 256);
    list.defer_gpu_write(&shadow, addr, 256);
    list.defer_gpu_write(&shadow, addr + 256, 256);

    let gpu_write_count = list
        .deferred()
        .copies
        .iter()
        .filter(|s| matches!(s, Step::GpuWrite(..)))
        .count();
    assert_eq!(gpu_write_count, 2);

    let (back, rebar, non_rebar) = collect_declared_writes(
        vec![
            (shadow.clone(), addr, 256),
            (shadow.clone(), addr, 256),
            (shadow.clone(), addr + 256, 256),
        ],
        false,
    );
    assert_eq!(back.len(), 1);
    assert_eq!(back[0].1.len(), 2);
    if shadow.is_rebar() {
        assert_eq!(rebar.len(), 1);
        assert_eq!(rebar[0].declared_spans.len(), 2);
        assert!(non_rebar.is_empty());
    } else if shadow.carded() {
        assert!(rebar.is_empty());
        assert!(non_rebar.is_empty());
    } else {
        assert!(rebar.is_empty());
        assert_eq!(non_rebar.len(), 3);
    }

    let mut dst = [0u8; 16];
    list.defer_copy(dst.as_mut_ptr(), backing.as_ptr(), 16);
    list.defer_gpu_write(&shadow, addr, 256);
    let gpu_write_count_after = list
        .deferred()
        .copies
        .iter()
        .filter(|s| matches!(s, Step::GpuWrite(..)))
        .count();
    assert_eq!(gpu_write_count_after, 3);

    list.close();
    list.reset();
    drop(back);
    drop(rebar);
    drop(non_rebar);
    drop(shadow);
    crate::shadow::unregister(addr, start);
    assert_eq!(host.d3d12_errors(), errors_before);
}

#[test]
fn recycle_list_releases_scratch_uploads_and_tracked_resources_immediately() {
    let _retire = isolated_retire_scope();
    let Some(host) = host() else {
        return;
    };
    let list = host.new_list().expect("host list");
    let list_raw = list.list.as_raw() as usize;

    // Track a title texture on the list, then release the title object so
    // only the list's tracked state and epoch keep HostTexture alive.
    let null = std::ptr::null();
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4,
        Height: 4,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let iid = ID3D12Resource::IID;
    let mut tex = null;
    assert_eq!(
        crate::resource::create_committed(
            null,
            (&raw const heap).cast(),
            0,
            (&raw const desc).cast(),
            0,
            null.cast(),
            (&raw const iid).cast(),
            &mut tex,
        ),
        crate::object::S_OK
    );
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    list.transition(tex, SHADER_RESOURCE);
    let weak_tex =
        std::sync::Arc::downgrade(&crate::resource::host_copy(tex).expect("tracked host texture"));
    let release = crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(tex, 0x10);
    assert_eq!(release(tex), 0);
    assert!(weak_tex.upgrade().is_some());

    // Populate scratch chunks, uploads, kept objects, shadow states,
    // rect scratch, deferred work, pipelines, and a pinned retirement epoch.
    assert!(list.scratch(3 << 20).is_some());
    assert!(list.scratch(3 << 20).is_some());
    let (upload_res, _) = list.upload(host, 4096).expect("list upload");
    list.keep(&upload_res);
    drop(upload_res);
    let shadow_buf = host.default_buffer(256).expect("shadow buffer");
    list.use_shadow(&shadow_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    drop(shadow_buf);
    assert!(list.rect_vertices(4096).is_some());
    assert!(list.rect_args(4096).is_some());
    assert!(list.rect_counter().is_some());
    list.used_pipeline("recycle_test_pipeline");
    let (deferred_counter, deferred_drops) = DropCounter::new();
    list.defer_host(move || {
        let _ = &deferred_counter;
    });
    let (retired_counter, retired_drops) = DropCounter::new();
    retire(retired_counter);
    assert!(list.has_epoch());
    assert_eq!(retired_drops.load(Ordering::SeqCst), 0);

    let errors_before = host.d3d12_errors();
    list.close();
    host.execute(&[&list]);

    assert!(weak_tex.upgrade().is_some());
    assert_eq!(deferred_drops.load(Ordering::SeqCst), 0);
    assert_eq!(retired_drops.load(Ordering::SeqCst), 0);
    assert_eq!(
        list.scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .chunks
            .len(),
        2
    );

    // Ensure room in free_lists so recycling parks the list rather than dropping it.
    {
        let mut pool = host.free_lists.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() >= LIST_POOL_CAP {
            pool.pop();
        }
    }
    let reused_before = host.reused_lists();
    host.recycle_list(list);

    assert!(
        weak_tex.upgrade().is_none(),
        "tracked HostTexture must drop immediately when list is recycled"
    );
    assert_eq!(
        deferred_drops.load(Ordering::SeqCst),
        1,
        "deferred closures must drop immediately when list is recycled"
    );
    assert_eq!(
        retired_drops.load(Ordering::SeqCst),
        1,
        "list epoch must release immediately when list is recycled"
    );

    {
        let pool = host.free_lists.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pooled) = pool.iter().find(|l| l.list.as_raw() as usize == list_raw) {
            assert!(
                pooled
                    .scratch
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .chunks
                    .is_empty(),
                "pooled list must hold 0 scratch chunks"
            );
            let rect = pooled
                .rect_scratch
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            assert!(rect.vertices.is_none());
            assert!(rect.args.is_none());
            assert!(rect.counters.is_empty());
            assert_eq!((rect.counter_chunk, rect.counter_used), (0, 0));
            assert!(
                pooled
                    .states
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty()
            );
            assert!(
                pooled
                    .shadow_states
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty()
            );
            assert!(
                pooled
                    .dirty_resources
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty()
            );
            assert!(
                pooled
                    .pipelines
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty()
            );
            assert_eq!(pooled.state_gen(), 0);
            assert!(!pooled.has_epoch());
            let deferred = pooled.deferred();
            assert!(deferred.uploads.is_empty());
            assert!(deferred.kept.is_empty());
            assert!(deferred.copies.is_empty());
        } else {
            assert!(
                host.reused_lists() > reused_before,
                "recycled list should sit in free_lists unless popped concurrently"
            );
        }
    }

    // Pop from the pool and verify D3D12 reset + recording + execution
    // succeed cleanly with zero debug-layer errors.
    let reused = host.acquire_list().expect("acquire recycled list");
    let (gpu, ptr) = reused.scratch(4096).expect("scratch after pool reuse");
    assert_ne!(gpu, 0);
    assert!(!ptr.is_null());
    assert!(reused.rect_vertices(4096).is_some());
    assert!(reused.rect_args(4096).is_some());
    assert!(reused.rect_counter().is_some());
    reused.close();
    host.execute(&[&reused]);
    assert_eq!(host.d3d12_errors(), errors_before);
}

#[cfg(test)]
mod batch {
    // Batch::texture_due lives here; `mod tests` is already taken above
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn exact_gpu_writes_are_declared_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new(0), Arc::new(0));
        assert!(batch.new_gpu_write(&a, 4096, 16));
        assert!(!batch.new_gpu_write(&a.clone(), 4096, 16));
        assert!(batch.new_gpu_write(&a, 4096, 32));
        assert!(batch.new_gpu_write(&a, 4104, 16));
        assert!(batch.new_gpu_write(&b, 4096, 16));
        // CPU copies can flush refreshes, but these GPU writes stay pending.
        batch.title_written();
        assert!(!batch.new_gpu_write(&a, 4096, 16));
        assert!(Batch::default().new_gpu_write(&a, 4096, 16));
    }

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a, 64));
        assert!(batch.texture_due(&b, 64));
        // ... later lists of the same submission with equal or smaller baseline do not ...
        assert!(!batch.texture_due(&a, 64));
        assert!(!batch.texture_due(&a.clone(), 32));
        assert!(!batch.texture_due(&b, 64));
        // ... but a later list requiring a higher mapped-byte baseline does.
        assert!(batch.texture_due(&a, 128));
        assert!(!batch.texture_due(&a, 128));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a, 0));
        assert!(!batch.texture_due(&a, 0));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a, 0));
        assert!(!batch.texture_due(&a, 0));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a, 0));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t, 0) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
         before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod eager_gbv_tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn parse_eager_gbv_respects_off_current_and_guarded_values() {
        assert_eq!(parse_eager_gbv(None), None);
        assert_eq!(parse_eager_gbv(Some(OsStr::new(""))), None);
        assert_eq!(parse_eager_gbv(Some(OsStr::new("0"))), None);
        assert_eq!(
            parse_eager_gbv(Some(OsStr::new("1"))),
            Some(EagerGbv::Current)
        );
        assert_eq!(
            parse_eager_gbv(Some(OsStr::new("guarded"))),
            Some(EagerGbv::Guarded)
        );
    }

    #[test]
    fn front_load_flag_matches_patch_mode() {
        assert_eq!(
        front_load_flag_for(
            D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_GUARDED_VALIDATION
        ),
        Ok(
            D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_GUARDED_VALIDATION_SHADERS
        )
    );
        assert_eq!(
        front_load_flag_for(
            D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_UNGUARDED_VALIDATION
        ),
        Ok(
            D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_UNGUARDED_VALIDATION_SHADERS
        )
    );
        assert_eq!(
        front_load_flag_for(
            D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_STATE_TRACKING_ONLY
        ),
        Ok(
            D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_TRACKING_ONLY_SHADERS
        )
    );
    }

    #[test]
    #[ignore = "requires an isolated process with the D3D12 debug layer and a WARP software adapter"]
    fn eager_gbv_round_trips_and_preserves_unrelated_flags_on_warp() -> windows::core::Result<()> {
        // SAFETY: runs only in an isolated test process with debug layer and WARP.
        unsafe {
            let mut layer: Option<ID3D12Debug> = None;
            D3D12GetDebugInterface(&mut layer)?;
            let layer = layer.ok_or_else(windows::core::Error::empty)?;
            layer.EnableDebugLayer();
            layer
                .cast::<ID3D12Debug1>()?
                .SetEnableGPUBasedValidation(true);
            let adapter = warp_adapter()?;
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            let device = device.ok_or_else(windows::core::Error::empty)?;

            let debug = device.cast::<ID3D12DebugDevice1>()?;
            let kind = D3D12_DEBUG_DEVICE_PARAMETER_GPU_BASED_VALIDATION_SETTINGS;
            let size = size_of::<D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS>() as u32;
            let mut settings = D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS::default();
            debug.GetDebugParameter(kind, (&raw mut settings).cast(), size)?;
            // Turn on an unrelated create flag (`FRONT_LOAD_CREATE_TRACKING_ONLY_SHADERS`)
            // and verify eager configuration preserves it alongside the added front-load bit.
            let preserved =
            D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_TRACKING_ONLY_SHADERS;
            settings.PipelineStateCreateFlags.0 |= preserved.0;
            debug.SetDebugParameter(kind, (&raw const settings).cast(), size)?;

            configure_eager_gbv(&device, EagerGbv::Current)?;
            let mut after_current = D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS::default();
            debug.GetDebugParameter(kind, (&raw mut after_current).cast(), size)?;
            assert_ne!(after_current.PipelineStateCreateFlags.0 & preserved.0, 0);

            configure_eager_gbv(&device, EagerGbv::Guarded)?;
            let mut after_guarded = D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS::default();
            debug.GetDebugParameter(kind, (&raw mut after_guarded).cast(), size)?;
            assert_eq!(
                after_guarded.DefaultShaderPatchMode,
                D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_GUARDED_VALIDATION
            );
            assert_ne!(
            after_guarded.PipelineStateCreateFlags.0
                & D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_GUARDED_VALIDATION_SHADERS
                    .0,
            0
        );
            assert_ne!(after_guarded.PipelineStateCreateFlags.0 & preserved.0, 0);
            Ok(())
        }
    }
}

#[cfg(test)]
mod wave_tests {
    use super::{
        REBAR_HASH_SOURCE, RebarWriteEntry, flagged_piece_byte_runs, host, intersect_ranges,
        isolated_retire_scope, merge_ranges, parse_rebar_mb_override, rebar_hash_pipeline,
        rebar_loose_policy, rebar_policy, resolve_rebar_budgets, spans_to_piece_runs, transition,
        wave32_supported,
    };
    use crate::watch::PAGE;

    #[test]
    fn wave32_support_matches_adapter_lane_range() {
        assert!(wave32_supported(true, 32, 32)); // NVIDIA
        assert!(wave32_supported(true, 32, 64)); // AMD RDNA
        assert!(wave32_supported(true, 16, 32)); // Intel Xe/Arc
        assert!(wave32_supported(true, 4, 128)); // WARP
        assert!(!wave32_supported(true, 64, 64)); // Wave64-only (GCN/Vega)
        assert!(!wave32_supported(false, 32, 32)); // No wave ops
    }

    #[test]
    fn rebar_policy_handles_all_hardware_tiers_and_fallbacks() {
        const MIB: u64 = 1 << 20;
        const GIB: u64 = 1 << 30;

        // Non-ReBAR, WARP, UMA/APU, or explicitly disabled always stay off.
        assert_eq!(rebar_policy(false, false, Some(32 * GIB), true), (false, 0));
        assert_eq!(rebar_policy(true, true, Some(64 * GIB), true), (false, 0));
        assert_eq!(rebar_policy(true, false, Some(32 * GIB), false), (false, 0));
        assert_eq!(rebar_policy(true, false, None, true), (false, 0));
        assert_eq!(rebar_loose_policy(false, false, Some(32 * GIB), true), 0);
        assert_eq!(rebar_loose_policy(true, true, Some(64 * GIB), true), 0);
        assert_eq!(rebar_loose_policy(true, false, Some(32 * GIB), false), 0);
        assert_eq!(rebar_loose_policy(true, false, None, true), 0);

        // Low-VRAM discrete GPUs (<= 8 GiB) stay on system-RAM shadows.
        assert_eq!(rebar_policy(true, false, Some(4 * GIB), true), (false, 0));
        assert_eq!(rebar_policy(true, false, Some(8 * GIB), true), (false, 0));
        assert_eq!(rebar_loose_policy(true, false, Some(8 * GIB), true), 0);

        // 12-14 GiB tier: conservative 512 MiB buffer budget, 0 loose budget.
        assert_eq!(
            rebar_policy(true, false, Some(12 * GIB), true),
            (true, 512 * MIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(14 * GIB), true),
            (true, 512 * MIB)
        );
        assert_eq!(rebar_loose_policy(true, false, Some(12 * GIB), true), 0);
        assert_eq!(rebar_loose_policy(true, false, Some(14 * GIB), true), 0);

        // 16 GiB tier (< 20 GiB): 1 GiB buffer budget, 2 GiB loose budget.
        assert_eq!(rebar_policy(true, false, Some(16 * GIB), true), (true, GIB));
        assert_eq!(
            rebar_loose_policy(true, false, Some(16 * GIB), true),
            2 * GIB
        );

        // 20-24 GiB tier (< 28 GiB): 2 GiB buffer budget, 4 GiB loose budget.
        assert_eq!(
            rebar_policy(true, false, Some(20 * GIB), true),
            (true, 2 * GIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(24 * GIB), true),
            (true, 2 * GIB)
        );
        assert_eq!(
            rebar_loose_policy(true, false, Some(20 * GIB), true),
            4 * GIB
        );
        assert_eq!(
            rebar_loose_policy(true, false, Some(24 * GIB), true),
            4 * GIB
        );

        // 28-32+ GiB tier (>= 28 GiB, e.g. RTX 5090): 3 GiB buffer, 6 GiB loose.
        assert_eq!(
            rebar_policy(true, false, Some(28 * GIB), true),
            (true, 3 * GIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(32 * GIB), true),
            (true, 3 * GIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(48 * GIB), true),
            (true, 3 * GIB)
        );
        assert_eq!(
            rebar_loose_policy(true, false, Some(28 * GIB), true),
            6 * GIB
        );
        assert_eq!(
            rebar_loose_policy(true, false, Some(32 * GIB), true),
            6 * GIB
        );
        assert_eq!(
            rebar_loose_policy(true, false, Some(48 * GIB), true),
            6 * GIB
        );

        // Pure env override parsing & policy composition (`VICTORY_REBAR_MB` / `VICTORY_REBAR_LOOSE_MB`).
        assert_eq!(parse_rebar_mb_override(None), None);
        assert_eq!(parse_rebar_mb_override(Some("  ")), None);
        assert_eq!(parse_rebar_mb_override(Some("abc")), None);
        assert_eq!(parse_rebar_mb_override(Some("0")), Some(0));
        assert_eq!(parse_rebar_mb_override(Some(" 1536 ")), Some(1536 * MIB));

        assert_eq!(
            resolve_rebar_budgets(true, false, Some(32 * GIB), true, Some("2048"), Some("0")),
            (true, 2 * GIB, 0)
        );
        assert_eq!(
            resolve_rebar_budgets(true, false, Some(16 * GIB), true, Some("0"), Some("1024")),
            (true, 0, GIB)
        );
        // `VICTORY_REBAR=0`, UMA, or unsupported GPU upload disables all ReBAR despite overrides.
        assert_eq!(
            resolve_rebar_budgets(
                true,
                false,
                Some(32 * GIB),
                false,
                Some("2048"),
                Some("4096")
            ),
            (false, 0, 0)
        );
        assert_eq!(
            resolve_rebar_budgets(true, true, Some(64 * GIB), true, Some("2048"), Some("4096")),
            (false, 0, 0)
        );
        assert_eq!(
            resolve_rebar_budgets(
                false,
                false,
                Some(32 * GIB),
                true,
                Some("2048"),
                Some("4096")
            ),
            (false, 0, 0)
        );
    }

    #[test]
    fn merge_ranges_coalesces_overlapping_and_adjacent_spans() {
        assert_eq!(merge_ranges(vec![]), vec![]);
        assert_eq!(merge_ranges(vec![(10, 10), (20, 15)]), vec![]);
        assert_eq!(
            merge_ranges(vec![(100, 200), (0, 50), (50, 120), (300, 400)]),
            vec![(0, 200), (300, 400)]
        );
        assert_eq!(
            merge_ranges(vec![(0, 100), (10, 40), (20, 90)]),
            vec![(0, 100)]
        );
    }

    #[test]
    fn piece_runs_and_flagged_intersection_dedupe_and_clamp() {
        // Two disjoint spans inside piece 0 plus a span in piece 1 coalesce into
        // one contiguous run `(0, 2)`, while piece 4 stays separate.
        let declared = vec![
            (0x100, 0x200),
            (0x8000, 0x9000),
            (PAGE + 16, PAGE + 64),
            (4 * PAGE, 5 * PAGE + 128),
        ];
        let runs = spans_to_piece_runs(&declared);
        assert_eq!(runs, vec![(0, 2), (4, 2)]);

        // Flag pieces 0, 4, and 5 (skip piece 1); clamp end to buffer length.
        let len = 5 * PAGE + 256;
        let (flagged, changed) = flagged_piece_byte_runs(&runs, &[1, 0, 1, 1], len);
        assert_eq!(changed, 3);
        assert_eq!(flagged, vec![(0, PAGE), (4 * PAGE, len)]);

        // Intersecting with declared spans yields only the declared bytes in
        // flagged pieces (both disjoint spans inside piece 0 are preserved).
        assert_eq!(
            intersect_ranges(&flagged, &declared),
            vec![(0x100, 0x200), (0x8000, 0x9000), (4 * PAGE, 5 * PAGE + 128)]
        );

        // Zero flagged pieces produces zero readback spans.
        let (none, changed) = flagged_piece_byte_runs(&runs, &[0, 0, 0, 0], len);
        assert_eq!(changed, 0);
        assert!(none.is_empty());
        assert!(intersect_ranges(&none, &declared).is_empty());
    }

    #[test]
    fn rebar_hash_hlsl_compiles_and_builds_on_host() {
        use crate::hlsl::{dxc_compile, system_compile};
        let probe = r#"
        RWByteAddressBuffer b : register(u0);
        [RootSignature("UAV(u0)")]
        [numthreads(1, 1, 1)]
        void main() { b.Store(0, b.Load(0)); }
    "#;
        if system_compile("probe", probe).is_ok()
            && let Err((e, msg)) = system_compile("rebar_hash", REBAR_HASH_SOURCE)
        {
            panic!("rebar_hash HLSL failed ({e}): {msg}");
        }
        if dxc_compile("probe", probe).is_ok()
            && let Err(e) = dxc_compile("rebar_hash", REBAR_HASH_SOURCE)
        {
            panic!("rebar_hash HLSL failed with DXC ({e})");
        }
        if let Some(h) = host() {
            assert!(rebar_hash_pipeline(&h.device).is_some());
        }
    }

    #[test]
    fn rebar_hash_shader_detects_single_byte_change_and_swap_on_gpu() {
        use windows::Win32::Graphics::Direct3D12::*;

        let Some(h) = host() else {
            return; // no host GPU here
        };
        let Some(pipe) = rebar_hash_pipeline(&h.device) else {
            panic!("rebar_hash compute pipeline failed to build on a live host");
        };
        let errors_before = h.d3d12_errors();
        let total_bytes = (2 * PAGE) as u64;
        let (src_buf, src_ptr) = h.shared_buffer(total_bytes).expect("shared src");
        let hashes_buf = h
            .default_buffer_in_state(32, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .expect("hashes UAV");
        let flags_uav = h
            .default_buffer_in_state(16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .expect("flags UAV");
        let (flags_rb, flags_ptr) = h.readback_buffer(16).expect("flags readback");

        // Seed two 64 KiB pieces with distinct non-repeating synthetic bytes.
        // SAFETY: `src_ptr` is mapped for `2 * PAGE` bytes.
        let src_slice = unsafe { std::slice::from_raw_parts_mut(src_ptr, 2 * PAGE) };
        for (i, b) in src_slice.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(37) ^ (i >> 8)) & 0xFF) as u8;
        }

        // SAFETY: live D3D12 resources.
        let (src_gpu, hashes_gpu, flags_gpu) = unsafe {
            (
                src_buf.GetGPUVirtualAddress(),
                hashes_buf.GetGPUVirtualAddress(),
                flags_uav.GetGPUVirtualAddress(),
            )
        };

        let run_pass = |mode: u32| -> [u32; 2] {
            let list = h.new_list().expect("host list");
            let l = &list.list;
            let constants = [0u32, total_bytes as u32, 0u32, mode, 2u32];
            // SAFETY: open list with live buffers and valid state transitions.
            unsafe {
                l.ResourceBarrier(&[transition(
                    &src_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                l.SetComputeRootSignature(&pipe.root);
                l.SetPipelineState(&pipe.pso);
                l.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
                l.SetComputeRootShaderResourceView(1, src_gpu);
                l.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                l.SetComputeRootUnorderedAccessView(3, flags_gpu);
                l.Dispatch(2, 1, 1);
                if mode != 0 {
                    l.ResourceBarrier(&[
                        transition(
                            &flags_uav,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        ),
                        transition(
                            &src_buf,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COMMON,
                        ),
                    ]);
                    l.CopyBufferRegion(&flags_rb, 0, &flags_uav, 0, 8);
                    l.ResourceBarrier(&[transition(
                        &flags_uav,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                } else {
                    l.ResourceBarrier(&[transition(
                        &src_buf,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
            }
            list.close();
            h.execute(&[&list]);
            list.reset();
            // SAFETY: fence completed and `flags_ptr` covers 16 bytes of READBACK.
            let out = unsafe { std::slice::from_raw_parts(flags_ptr.cast::<u32>(), 2) };
            [out[0], out[1]]
        };

        // 1. Compute initial hashes.
        let _ = run_pass(0);
        // 2. Compare unchanged -> both flags must be 0.
        assert_eq!(run_pass(1), [0, 0]);
        // 3. Flip a single byte in piece 1 -> only piece 1 flagged (and hash updated).
        src_slice[PAGE + 12345] ^= 0x01;
        assert_eq!(run_pass(1), [0, 1]);
        // 4. Re-compare without further writes -> stored hash was updated, so [0, 0].
        assert_eq!(run_pass(1), [0, 0]);
        // 5. Swap two 16-byte chunks at different lanes in piece 0 -> non-commutative
        //    reduction must detect the swap on piece 0 only.
        for i in 0..16 {
            src_slice.swap(i, 16 + i);
        }
        assert_eq!(run_pass(1), [1, 0]);
        assert_eq!(run_pass(1), [0, 0]);
        assert_eq!(
            h.d3d12_errors(),
            errors_before,
            "unexpected D3D12 debug layer error during rebar_hash test"
        );
    }

    #[test]
    fn rebar_finish_copy_back_failure_keeps_pending_and_preserves_vram_on_refresh() {
        let _rebar = crate::shadow::lock_rebar_gpu_test();
        let _retire = isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };

        let title = vec![0x11u8; 4096];
        let start = title.as_ptr() as u64;
        let len = title.len() as u64;
        let (buf_a, ptr_a) = h.shared_buffer(len).expect("shared buffer a");
        let (buf_b, ptr_b) = h.shared_buffer(len).expect("shared buffer b");
        // SAFETY: live D3D12 buffers.
        let (gpu_a, gpu_b) =
            unsafe { (buf_a.GetGPUVirtualAddress(), buf_b.GetGPUVirtualAddress()) };
        let shadow_owed =
            crate::shadow::Shadow::new_rebar_from_parts_for_test(buf_a, ptr_a, gpu_a, start, len);
        let shadow_empty =
            crate::shadow::Shadow::new_rebar_from_parts_for_test(buf_b, ptr_b, gpu_b, start, len);

        assert!(
            shadow_owed.writing(start as usize + 64, 64, |_, n| Some(vec![
                0x11u8;
                n as usize
            ]))
        );
        assert!(shadow_empty.writing(start as usize, 64, |_, n| Some(vec![0x11u8; n as usize])));
        // Initialize hash state so we can verify dirty-span restoration on readback failure.
        let _ = shadow_owed
            .prepare_rebar_hashes(|n| h.default_buffer(n))
            .expect("initial rebar hashes");
        assert!(
            shadow_owed
                .prepare_rebar_hashes(|n| h.default_buffer(n))
                .expect("clean rebar hashes")
                .pre_runs
                .is_empty()
        );
        // Simulate GPU-written bytes in `[64..128)`.
        // SAFETY: `ptr_a` is mapped for `len` bytes of shared RAM in this test.
        unsafe {
            std::ptr::write_bytes(ptr_a, 0x22, len as usize);
            std::ptr::write_bytes(ptr_a.add(64), 0xAA, 64);
        }

        let entries = vec![
            RebarWriteEntry {
                shadow: shadow_owed.clone(),
                declared_spans: vec![(64, 128)],
                piece_runs: vec![(0, 1)],
                compare_out_base: None,
                compare_pieces: 0,
            },
            RebarWriteEntry {
                shadow: shadow_empty.clone(),
                declared_spans: Vec::new(),
                piece_runs: Vec::new(),
                compare_out_base: None,
                compare_pieces: 0,
            },
        ];

        // Simulate `copy_rebar_readback` returning `false`.
        let ok = h.finish_rebar_copy_back_with(entries, 0, |_, _| false);
        assert!(!ok, "failed ReBAR readback must report failure");
        assert!(
            shadow_owed.has_pending(),
            "undelivered GPU span must remain pending on readback failure"
        );
        assert!(
            !shadow_empty.has_pending(),
            "entry with no owed spans may clear pending cleanly"
        );
        assert_eq!(
            shadow_owed
                .prepare_rebar_hashes(|n| h.default_buffer(n))
                .expect("restored dirty hashes")
                .pre_runs,
            vec![(0, 1)]
        );

        // A subsequent refresh must not overwrite the pending GPU bytes `[64..128)`.
        shadow_owed.update(start as usize, len as usize);
        let bytes = shadow_owed.read_test_bytes(start, len as usize);
        assert!(bytes[..64].iter().all(|&b| b == 0x11));
        assert!(
            bytes[64..128].iter().all(|&b| b == 0xAA),
            "refresh must keep undelivered GPU result intact while copy-back is pending"
        );
        assert!(bytes[128..].iter().all(|&b| b == 0x11));
    }
}

#[cfg(test)]
mod plane_shaders {
    /// Build the actual pipelines so a blob without a usable root signature
    /// cannot pass on the strength of a successfully compiled vertex shader.
    #[cfg(victory_bundled_hlsl)]
    #[test]
    fn bundled_depth_helpers_build_both_pipelines() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        };
        if super::host().is_none() {
            return;
        }
        for (steps, format) in [
            (true, DXGI_FORMAT_D24_UNORM_S8_UINT),
            (false, DXGI_FORMAT_D32_FLOAT_S8X24_UINT),
        ] {
            super::PlanePass::make(steps, format).expect("depth plane pipeline");
        }
    }

    /// The pass's shaders compile with one of our two compilers, whichever
    /// this machine has (the system's `d3dcompiler` first, DXC when it
    /// refuses them: Wine's builtin refuses other helpers' source), and the
    /// bytecode is a DXBC container with the root signature inside.
    #[test]
    fn depth_plane_pass_shaders_compile() {
        use victory_shader::{Container, FourCC};
        let stages = [
            ("corner", super::PLANE_CORNERS, "vs_5_1"),
            ("plane", super::PLANE_STEPS, "ps_5_1"),
            ("plane", super::PLANE_FLOATS, "ps_5_1"),
        ];
        for (entry, source, target) in stages {
            let bytes = crate::hlsl::compile_stage("depth plane probe", source, entry, target)
                .unwrap_or_else(|e| panic!("{entry} ({target}): {e}"));
            let container = Container::parse(&bytes).unwrap();
            if entry == "plane" {
                assert!(
                    [FourCC::RTS0, FourCC(*b"RDAT")]
                        .iter()
                        .any(|&part| container.part(part).is_some()),
                    "{entry}: no root signature in the shader"
                );
            }
        }
    }

    #[test]
    fn depth_plane_draw_preserves_first_and_last_array_slices() {
        use super::*;
        let Some(host) = host() else {
            eprintln!("Skipping depth plane array readback: no D3D12 device");
            return;
        };
        const SLICES: u16 = 257;
        let list = host.new_list().unwrap();
        let texture = |format, flags, state| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 1,
                Height: 1,
                DepthOrArraySize: SLICES,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            // SAFETY: the descriptors live through the call and the output slot is ours.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut texture,
                    )
                    .unwrap()
            };
            texture.unwrap()
        };
        let plane = texture(
            DXGI_FORMAT_R32_TYPELESS,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let depth = texture(
            DXGI_FORMAT_R32G8X24_TYPELESS,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );
        let (upload, input) = host.upload_buffer(1024).unwrap();
        let (readback, output) = host.readback_buffer(1024).unwrap();
        let subresource = |resource: &ID3D12Resource, index| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: the resources remain live through submission; these descriptors borrow them.
            pResource: unsafe { std::mem::transmute_copy(resource) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: index,
            },
        };
        let location = |resource: &ID3D12Resource, texture: &ID3D12Resource, index, offset| {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            // SAFETY: the texture is live; index is in range, and one footprint is requested.
            unsafe {
                host.device.GetCopyableFootprints(
                    &texture.GetDesc(),
                    index,
                    1,
                    offset,
                    Some(&mut footprint),
                    None,
                    None,
                    None,
                )
            };
            D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as for the subresource locations above.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            }
        };
        let samples = [(0, 0, 0.25_f32), (u32::from(SLICES) - 1, 512, 0.75_f32)];
        for (index, offset, expected) in samples {
            // SAFETY: an aligned f32 fits inside the 1024-byte upload buffer at both offsets.
            unsafe { input.add(offset).cast::<f32>().write(expected) };
            let src = location(&upload, &plane, index, offset as u64);
            // SAFETY: both resources are in copy states and the locations cover one full texel.
            unsafe {
                list.list
                    .CopyTextureRegion(&subresource(&plane, index), 0, 0, 0, &src, None)
            };
        }
        // SAFETY: the plane was created in COPY_DEST and the list has not changed that state.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &plane,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )])
        };
        let plane_views = OnceLock::new();
        record_plane_pass(&list.list, &depth, &plane, &plane_views);
        // SAFETY: depth was created in DEPTH_WRITE and the pass leaves it in that state.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &depth,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )])
        };
        for (index, offset, _) in samples {
            let dst = location(&readback, &depth, index, offset as u64);
            // SAFETY: both locations are live, in their copy states and cover one full depth texel.
            unsafe {
                list.list
                    .CopyTextureRegion(&dst, 0, 0, 0, &subresource(&depth, index), None)
            };
        }
        list.mark_work();
        list.close();
        host.execute(&[&list]);
        for (index, offset, expected) in samples {
            // SAFETY: execute waited for the readback and both offsets are aligned within the mapping.
            let actual = unsafe { output.add(offset).cast::<f32>().read() };
            assert_eq!(actual, expected, "depth slice {index}");
        }
    }

    /// `defer_host_read` runs after preceding deferred copies and holds a
    /// byte-range mapping guard over its source range while executing.
    #[test]
    fn defer_host_read_guards_its_range_against_overlapping_remaps() {
        let _retire = super::isolated_retire_scope();
        let Some(host) = super::host() else { return };
        let list = host.new_list().expect("host list");
        let src = [0x5Au8; 64];
        let mut dst = [0u8; 64];
        let dst_ptr = dst.as_mut_ptr();
        let dst_addr = dst_ptr as usize;
        list.defer_copy(dst_ptr, src.as_ptr(), src.len());

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let observed = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
        let observed_read = observed.clone();
        list.defer_host_read(dst_ptr, 64, move || {
            // SAFETY: `dst` lives until after `run_deferred` finishes.
            let first = unsafe { *(dst_addr as *const u8) };
            observed_read.store(first, std::sync::atomic::Ordering::SeqCst);
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });

        let remapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let remapped_worker = remapped.clone();
        let waiter = std::thread::spawn(move || {
            entered_rx.recv().unwrap();
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let remap = std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                let _guard = crate::watch::title_memory_remapping(dst_addr, 64);
                remapped_worker.store(true, std::sync::atomic::Ordering::SeqCst);
            });
            started_rx.recv().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(
                !remapped.load(std::sync::atomic::Ordering::SeqCst),
                "overlapping remap proceeded during defer_host_read"
            );
            release_tx.send(()).unwrap();
            remap.join().unwrap();
            assert!(remapped.load(std::sync::atomic::Ordering::SeqCst));
        });

        let mut batch = super::Batch::default();
        let mut submitted = 0u64;
        list.run_deferred(&mut batch, host, &mut submitted);
        waiter.join().unwrap();
        assert_eq!(observed.load(std::sync::atomic::Ordering::SeqCst), 0x5A);
        list.close();
        list.reset();
    }
}

#[cfg(test)]
mod title_memory_io {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const PAGE: usize = 4096;

    /// Two committed pages followed by one reserved (not committed) page.
    fn region() -> *mut u8 {
        // SAFETY: a fresh reservation of ours, of which two pages are committed.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE);
            assert!(!base.is_null());
            let committed = VirtualAlloc(base, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!committed.is_null());
            base.cast()
        }
    }

    fn release(base: *mut u8) {
        // SAFETY: the reservation made by `region`.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn write_mapped_writes_mapped_pages_and_skips_the_rest() {
        let base = region();
        let data = vec![0xabu8; 3 * PAGE];
        // The third page is not committed: its piece is skipped, the others land.
        assert_eq!(write_mapped(base, data.as_ptr(), 3 * PAGE), Some(2 * PAGE));
        // SAFETY: the two committed pages of `region`.
        let written = unsafe { std::slice::from_raw_parts(base, 2 * PAGE) };
        assert!(written.iter().all(|&b| b == 0xab));
        release(base);
    }

    #[test]
    fn copy_into_title_memory_that_is_not_all_mapped_is_left_alone() {
        let base = region();
        let from = vec![0x5au8; 3 * PAGE];
        copy_mapped(base, from.as_ptr(), 3 * PAGE, true);
        // A whole copy into mapped memory works without asking the system first.
        copy_mapped(base, from.as_ptr(), 2 * PAGE, true);
        // SAFETY: the two committed pages of `region`.
        let now = unsafe { std::slice::from_raw_parts(base, 2 * PAGE) };
        assert!(now.iter().all(|&b| b == 0x5a));
        release(base);
    }

    #[test]
    fn victory_copy_try_variants_preserve_bytes_and_recover_from_faults() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualProtect,
        };
        super::install_fast_copy_handler();

        let variants: Vec<(&str, unsafe extern "C" fn(*mut u8, *const u8, usize) -> u32)> = {
            let mut v: Vec<(&str, unsafe extern "C" fn(*mut u8, *const u8, usize) -> u32)> =
                vec![("sse2", super::victory_copy_try_sse2)];
            if std::is_x86_feature_detected!("avx2") {
                v.push(("avx2", super::victory_copy_try_avx2));
            }
            v
        };

        // 1. Verify exact copies across lengths 0..=4097 and unaligned src/dst
        // offsets without clobbering sentinel guard bytes.
        let src: Vec<u8> = (0..8192usize)
            .map(|i| ((i.wrapping_mul(53) ^ (i >> 5) ^ 0x3c) & 0xff) as u8)
            .collect();
        let mut dst = vec![0xEEu8; 8192];
        for &(name, copy_fn) in &variants {
            for dst_off in [0usize, 1, 7, 15, 16, 17, 31, 32, 33] {
                for src_off in [0usize, 3, 15, 16, 31] {
                    for len in [
                        0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95,
                        127, 128, 129, 159, 160, 161, 255, 256, 257, 4097,
                    ] {
                        dst.fill(0xEE);
                        // SAFETY: disjoint buffers with `dst_off + len < 8192`.
                        let ok = unsafe {
                            copy_fn(
                                dst.as_mut_ptr().add(dst_off),
                                src.as_ptr().add(src_off),
                                len,
                            )
                        };
                        assert_eq!(ok, 1, "{name} copy failed for len={len}");
                        assert_eq!(
                            &dst[dst_off..dst_off + len],
                            &src[src_off..src_off + len],
                            "{name} mismatch dst_off={dst_off} src_off={src_off} len={len}"
                        );
                        assert!(
                            dst[..dst_off].iter().all(|&b| b == 0xEE)
                                && dst[dst_off + len..].iter().all(|&b| b == 0xEE),
                            "{name} overwrote guard bytes dst_off={dst_off} len={len}"
                        );
                    }
                }
            }
        }

        // 2. Verify fault recovery on an uncommitted page at multiple distances
        // (hits small tail, alignment prologue, and 128B/64B main vector loops)
        // and verify `is_fast_copy_rip` covers both routines.
        assert!(super::is_fast_copy_rip(
            &raw const super::victory_copy_try_avx2_start as u64
        ));
        assert!(super::is_fast_copy_rip(
            &raw const super::victory_copy_try_sse2_start as u64
        ));
        assert!(!super::is_fast_copy_rip(
            &raw const super::victory_copy_try_avx2_end as u64
        ));
        assert!(!super::is_fast_copy_rip(
            &raw const super::victory_copy_try_sse2_end as u64
        ));

        const MEM_TOP_DOWN: u32 = 0x10_0000;
        // SAFETY: reserve 2 pages, commit only the first.
        unsafe {
            let base = VirtualAlloc(
                std::ptr::null(),
                2 * 4096,
                MEM_RESERVE | MEM_TOP_DOWN,
                PAGE_NOACCESS,
            )
            .cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(
                VirtualAlloc(base.cast(), 4096, MEM_COMMIT, PAGE_READWRITE),
                base.cast()
            );
            base.write_bytes(0x42, 4096);
            let mut out = vec![0u8; 2 * 4096];
            for &(name, copy_fn) in &variants {
                for &(start_off, len) in &[
                    (4096usize, 16usize),
                    (4096 - 3, 16),
                    (4096 - 32, 64),
                    (4096 - 100, 256),
                    (0, 2 * 4096),
                ] {
                    let ok = copy_fn(out.as_mut_ptr().add(7), base.add(start_off), len);
                    assert_eq!(
                        ok, 0,
                        "{name} must return 0 on unmapped page (start_off={start_off}, len={len})"
                    );
                }
            }

            // 3. Verify `PAGE_GUARD` source fault also recovers cleanly to 0.
            let mut old = 0u32;
            assert_ne!(
                VirtualProtect(base.cast(), 4096, PAGE_READWRITE | PAGE_GUARD, &mut old),
                0
            );
            let ok = super::victory_copy_try(out.as_mut_ptr(), base, 256);
            assert_eq!(ok, 0, "PAGE_GUARD read must recover cleanly");

            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn read_into_raw_probe_and_sparse_copy_handle_holes_and_noaccess() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualProtect,
        };

        // Allocate 192 KiB: [0..64 KiB) committed, [64 KiB..128 KiB) reserved hole,
        // [128 KiB..192 KiB) committed, with its last 4 KiB set to PAGE_NOACCESS.
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const BLOCK: usize = 64 << 10;
        const TOTAL: usize = 3 * BLOCK;
        // SAFETY: isolated reservation freed before returning.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                TOTAL,
                MEM_RESERVE | MEM_TOP_DOWN,
                PAGE_NOACCESS,
            )
            .cast::<u8>()
        };
        assert!(!base.is_null());
        // SAFETY: commit blocks 0 and 2, protect the final 4 KiB page as NOACCESS.
        unsafe {
            assert_eq!(
                VirtualAlloc(base.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE),
                base.cast()
            );
            assert_eq!(
                VirtualAlloc(
                    base.add(2 * BLOCK).cast(),
                    BLOCK,
                    MEM_COMMIT,
                    PAGE_READWRITE
                ),
                base.add(2 * BLOCK).cast()
            );
            base.write_bytes(0x31, BLOCK);
            base.add(2 * BLOCK).write_bytes(0x74, BLOCK);
            let mut old = 0u32;
            assert_ne!(
                VirtualProtect(base.add(TOTAL - PAGE).cast(), PAGE, PAGE_NOACCESS, &mut old),
                0
            );
        }

        // `read_into_raw` and `read_into_probe` agree on readable, unmapped, and NOACCESS spans.
        let mut buf = [0u8; 64];
        assert!(super::read_into_raw(base, &mut buf));
        assert!(buf.iter().all(|&b| b == 0x31));
        buf.fill(0);
        assert!(super::read_into_probe(base, &mut buf));
        assert!(buf.iter().all(|&b| b == 0x31));
        assert!(!super::read_into_raw(base.wrapping_add(BLOCK), &mut buf));
        assert!(!super::read_into_probe(base.wrapping_add(BLOCK), &mut buf));
        assert!(!super::read_into_raw(
            base.wrapping_add(TOTAL - PAGE),
            &mut buf
        ));
        assert!(!super::read_into_probe(
            base.wrapping_add(TOTAL - PAGE),
            &mut buf
        ));

        // `copy_mapped` (len >= 64 KiB -> `read_sparse` fallback) copies committed spans
        // and zeroes both reserved holes and `PAGE_NOACCESS` pages.
        let mut dst = vec![0xAAu8; TOTAL];
        copy_mapped(dst.as_mut_ptr(), base, TOTAL, false);
        assert!(dst[..BLOCK].iter().all(|&b| b == 0x31));
        assert!(dst[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(dst[2 * BLOCK..TOTAL - PAGE].iter().all(|&b| b == 0x74));
        assert!(dst[TOTAL - PAGE..].iter().all(|&b| b == 0));

        // SAFETY: free test reservation.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }
}

#[cfg(test)]
mod present_retry {
    use super::{PresentStep, Retry, retry_delay};
    use std::time::{Duration, Instant};

    #[test]
    fn the_backoff_starts_short_and_stops_at_a_second() {
        let schedule: Vec<u64> = (1..=7).map(|n| retry_delay(n).as_millis() as u64).collect();
        assert_eq!(schedule, [100, 200, 400, 800, 1_000, 1_000, 1_000]);
    }

    #[test]
    fn a_failed_attempt_waits_before_the_next_one() {
        let start = Instant::now();
        let mut retry = Retry::new();
        // Nothing has failed, so an attempt is due.
        assert!(retry.due(start));
        retry.failed(start);
        assert_eq!(retry.attempts, 1);
        assert!(!retry.due(start));
        assert!(!retry.due(start + Duration::from_millis(99)));
        assert!(retry.due(start + Duration::from_millis(100)));
        // A second failure doubles the wait and counts.
        let second = start + Duration::from_millis(100);
        retry.failed(second);
        assert_eq!(retry.attempts, 2);
        assert!(!retry.due(second + Duration::from_millis(199)));
        assert!(retry.due(second + Duration::from_millis(200)));
    }

    #[test]
    fn a_working_attempt_clears_the_state_and_reports_the_failures() {
        let start = Instant::now();
        let mut retry = Retry::new();
        for _ in 0..3 {
            retry.failed(start);
        }
        assert_eq!(retry.attempts, 3);
        assert_eq!(retry.reset(), 3);
        assert_eq!(retry.attempts, 0);
        assert!(retry.due(start));
        // A later failure starts the schedule over.
        retry.failed(start);
        assert!(!retry.due(start + Duration::from_millis(99)));
    }

    /// A minute behind a desktop that never appears: one attempt per backoff
    /// instead of one per presented frame.
    #[test]
    fn a_minute_without_a_desktop_asks_a_bounded_number_of_times() {
        let mut retry = Retry::new();
        let mut now = Instant::now();
        let mut attempts = 0u32;
        for _ in 0..1_200 {
            // 60 s at 20 fps.
            if retry.due(now) {
                attempts += 1;
                retry.failed(now);
            }
            now += Duration::from_millis(50);
        }
        assert!(
            (59..=64).contains(&attempts),
            "{attempts} attempts in a minute, one per backoff expected"
        );
    }

    #[test]
    fn every_step_of_a_presenter_has_a_name() {
        for step in [
            PresentStep::Window,
            PresentStep::Factory,
            PresentStep::SwapChain,
            PresentStep::List,
        ] {
            assert!(!step.name().is_empty());
        }
    }
}

#[cfg(test)]
mod relay_tests {
    use super::*;

    #[test]
    fn eval_late_met_handles_exact_partial_timestamp_and_shader_overlap() {
        use crate::queue::Written;
        let addr = 0x1000usize;
        // Exact value match satisfies wait.
        assert_eq!(
            relay::eval_late_met(&[], [(addr, Written::Value(7), 8)], addr, 8, |v| v >= 7,),
            Some(true)
        );
        // Later overwrite with smaller value makes it false.
        assert_eq!(
            relay::eval_late_met(
                &[],
                [(addr, Written::Value(7), 8), (addr, Written::Value(3), 8)],
                addr,
                8,
                |v| v >= 7,
            ),
            Some(false)
        );
        // Overlapping shader range requires settling (`None`).
        assert_eq!(
            relay::eval_late_met(
                &[(addr - 16, addr + 16)],
                [(addr, Written::Value(7), 8)],
                addr,
                8,
                |v| v >= 7,
            ),
            None
        );
        // Timestamp write requires settling (`None`).
        assert_eq!(
            relay::eval_late_met(&[], [(addr, Written::GpuTimestamp, 8)], addr, 8, |v| v >= 1),
            None
        );
        // Partial width write requires settling (`None`).
        assert_eq!(
            relay::eval_late_met(&[], [(addr, Written::Value(7), 4)], addr, 8, |v| v >= 7,),
            None
        );
    }

    #[test]
    fn landable_stops_at_first_incomplete_flight() {
        assert_eq!(
            relay::landable(&[Some(1), None, Some(2), None, Some(3)], 2),
            4
        );
        assert_eq!(relay::landable(&[Some(2), None], 1), 0);
        assert_eq!(relay::landable(&[None, Some(1), None], 0), 1);
        assert_eq!(relay::landable(&[Some(1), Some(2)], 5), 2);
    }

    #[test]
    fn flagged_pieces_unions_duplicate_piece_flags() {
        let set =
            relay::flagged_pieces(&[(1, 0, true), (1, 0, false), (1, 1, false), (2, 3, true)]);
        assert!(set.contains(&(1, 0)));
        assert!(!set.contains(&(1, 1)));
        assert!(set.contains(&(2, 3)));
    }

    #[test]
    fn should_record_brought_and_fallback_fetched_mark() {
        assert!(!relay::should_record_brought(0, None));
        assert!(!relay::should_record_brought(1, None));
        assert!(relay::should_record_brought(2, None));
        assert!(!relay::should_record_brought(5, Some(4)));
        assert!(relay::should_record_brought(6, Some(4)));

        assert_eq!(fallback_fetched_mark(false, 10, Some(6)), 10);
        assert_eq!(fallback_fetched_mark(true, 10, Some(6)), 6);
        assert_eq!(fallback_fetched_mark(true, 4, Some(6)), 4);
        assert_eq!(fallback_fetched_mark(true, 10, None), 0);
    }

    #[test]
    fn card_compare_and_fetch_shaders_compile_cleanly() {
        let cmp = crate::hlsl::compile_compute("card_compare", CARD_COMPARE);
        assert!(cmp.is_ok(), "CARD_COMPARE failed to compile: {cmp:?}");
        let fetch = crate::hlsl::compile_compute("card_fetch", CARD_FETCH);
        assert!(fetch.is_ok(), "CARD_FETCH failed to compile: {fetch:?}");
    }

    #[test]
    fn relay_rebar_back_copies_via_copy_queue_and_preserves_cpu_writes() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else { return };
        let mut title = vec![0x11u8; 64 << 10];
        let Some(shadow) =
            crate::shadow::Shadow::new_rebar_for_test(title.as_mut_ptr() as u64, 64 << 10, 0x11)
        else {
            return;
        };
        // Write GPU-side pattern into the mapped ReBAR buffer at [4096..8192).
        shadow.write_test_bytes(4096, &[0xcd; 4096]);
        // Concurrent CPU write outside the GPU-written range.
        title[0..256].fill(0x7a);

        let mut fetch = None;
        let start = title.as_ptr() as usize;
        host.relay_rebar_back(
            &[(shadow.clone(), vec![(start, start + (64 << 10))])],
            &mut fetch,
        );

        assert!(title[0..256].iter().all(|&b| b == 0x7a));
        assert!(title[4096..8192].iter().all(|&b| b == 0xcd));
        assert!(title[256..4096].iter().all(|&b| b == 0x11));
    }

    #[test]
    fn sync_copy_queue_tracks_signalled_fence_and_order() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else { return };
        let Some(copy_q) = host.copy_queue() else {
            return;
        };
        let list = host.new_list().expect("host list");
        list.close();
        host.execute(&[&list]);
        let wm1 = host.sync_copy_queue(copy_q).expect("sync_copy_queue");
        let wm2 = host
            .sync_copy_queue_for(copy_q, false)
            .expect("sync_copy_queue_for(false)");
        assert!(wm1 >= wm2);
    }

    #[test]
    fn shadow_relay_matches_host_relay() {
        assert_eq!(crate::shadow::relay(), crate::host::relay());
    }

    #[test]
    fn sync_execute_clears_pending_for_card_shadow() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else { return };
        if !crate::shadow::cards() {
            return;
        }
        let mut backing = vec![0x33u8; 2 << 20];
        let start = backing.as_mut_ptr() as u64;
        let len = backing.len() as u64;
        let resource = start as usize;
        crate::shadow::register(resource, start, len);
        let shadow = crate::shadow::find(start, len);
        if let Some(shadow) = shadow
            && shadow.carded()
        {
            let list = host.new_list().expect("host list");
            list.defer_gpu_write(&shadow, start as usize, 64 << 10);
            list.close();
            host.execute(&[&list]);
            assert!(
                !shadow.writes_pending(),
                "synchronous execute must clear Shadow::pending for card shadows"
            );
            list.reset();
        }
        crate::shadow::unregister(resource, start);
    }

    #[test]
    fn range_settle_is_default_on_and_respects_disable() {
        assert!(relay::range_settle_enabled(None));
        assert!(relay::range_settle_enabled(Some("1")));
        assert!(!relay::range_settle_enabled(Some("0")));
        assert!(relay::range_settle_enabled(Some("auto")));
    }

    #[test]
    fn line_overlaps_stale_distinguishes_disjoint_and_overlapping_flights() {
        use crate::retire::ExecEpochGuard;
        use crate::shadow::Shadow;
        use std::sync::Arc;

        let mut side = vec![0u8; 4 * crate::watch::PAGE];
        let Some(shadow_a) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            0x1000_0000,
            (4 * crate::watch::PAGE) as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return;
        };
        let Some(shadow_b) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            0x1000_0000,
            (4 * crate::watch::PAGE) as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return;
        };
        let ptr_a = Arc::as_ptr(&shadow_a) as usize;
        let ptr_b = Arc::as_ptr(&shadow_b) as usize;

        let make_flight = |number: u64, back: Vec<relay::Declared>| relay::Flight {
            number,
            fence: number,
            lists: Vec::new(),
            card: None,
            back,
            readback_ranges: Vec::new(),
            _exec_epoch: ExecEpochGuard {
                scope: None,
                epoch: 0,
            },
        };

        let mut line = relay::Line {
            line: vec![
                relay::Late::Run(make_flight(
                    1,
                    vec![(shadow_a.clone(), vec![(0x1000_0000, 0x1001_0000)])],
                )),
                relay::Late::Run(make_flight(
                    2,
                    vec![(shadow_b.clone(), vec![(0x1002_0000, 0x1003_0000)])],
                )),
            ],
            written: Vec::new(),
            lists: 2,
            target: 2,
            settling: false,
            poke: false,
            landed: Vec::new(),
            in_hand: Vec::new(),
            in_hand_back: Vec::new(),
            pending_woken: Vec::new(),
        };

        // Disjoint ranges on the same or other shadow must not match.
        assert!(!relay::line_overlaps_stale(
            &line,
            &[(ptr_a, 0x1001_0000, 0x1002_0000)]
        ));
        assert!(!relay::line_overlaps_stale(
            &line,
            &[(ptr_b, 0x1000_0000, 0x1001_0000)]
        ));
        assert!(!relay::line_overlaps_stale(&line, &[]));

        // Overlapping range on shadow_a matches flight 1.
        assert!(relay::line_overlaps_stale(
            &line,
            &[(ptr_a, 0x1000_8000, 0x1001_8000)]
        ));
        // Overlapping range on shadow_b matches flight 2.
        assert!(relay::line_overlaps_stale(
            &line,
            &[(ptr_b, 0x1002_0000, 0x1002_1000)]
        ));

        // When flight 1 moves into `in_hand_back` (`settling == true`), it still overlaps
        // until landing completes and clears `in_hand_back`.
        line.line.remove(0);
        line.settling = true;
        line.in_hand_back.push((ptr_a, 0x1000_0000, 0x1001_0000));
        assert!(relay::line_overlaps_stale(
            &line,
            &[(ptr_a, 0x1000_0000, 0x1000_1000)]
        ));
        assert!(!relay::line_overlaps_stale(
            &line,
            &[(ptr_a, 0x1001_0000, 0x1002_0000)]
        ));

        line.in_hand_back.clear();
        line.settling = false;
        assert!(!relay::line_overlaps_stale(
            &line,
            &[(ptr_a, 0x1000_0000, 0x1000_1000)]
        ));
        assert!(relay::line_overlaps_stale(
            &line,
            &[(ptr_b, 0x1002_0000, 0x1003_0000)]
        ));
    }

    #[test]
    fn range_scoped_wait_unblocks_without_draining_unrelated_flights() {
        use crate::retire::ExecEpochGuard;
        use crate::shadow::Shadow;
        use std::sync::{Arc, Condvar, Mutex};

        let mut side = vec![0u8; 4 * crate::watch::PAGE];
        let Some(shadow) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            0x2000_0000,
            (4 * crate::watch::PAGE) as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return;
        };
        let s_ptr = Arc::as_ptr(&shadow) as usize;
        let stale_hit = [(s_ptr, 0x2000_0000usize, 0x2001_0000usize)];
        let stale_disjoint = [(s_ptr, 0x2003_0000usize, 0x2004_0000usize)];

        let make_flight = |number: u64, range: (usize, usize)| relay::Flight {
            number,
            fence: number,
            lists: Vec::new(),
            card: None,
            back: vec![(shadow.clone(), vec![range])],
            readback_ranges: Vec::new(),
            _exec_epoch: ExecEpochGuard {
                scope: None,
                epoch: 0,
            },
        };

        let line = Arc::new(Mutex::new(relay::Line {
            line: vec![
                relay::Late::Run(make_flight(1, (0x2000_0000, 0x2001_0000))),
                relay::Late::Run(make_flight(2, (0x2002_0000, 0x2003_0000))),
            ],
            written: vec![(0x2000_0000, 0x2001_0000), (0x2002_0000, 0x2003_0000)],
            lists: 2,
            target: 2,
            settling: false,
            poke: false,
            landed: Vec::new(),
            in_hand: Vec::new(),
            in_hand_back: Vec::new(),
            pending_woken: Vec::new(),
        }));
        let cv = Arc::new(Condvar::new());

        // Disjoint stale range does not block at all even with both flights in flight.
        assert_eq!(
            relay::wait_on_line(&line, &cv, relay::WHY_STALE, false, |l| {
                !relay::line_overlaps_stale(l, &stale_disjoint)
            }),
            None
        );

        // Overlapping stale range blocks until flight 1 lands, even while flight 2 stays in line.
        let stage = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let line_w = line.clone();
        let cv_w = cv.clone();
        let stage_w = stage.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            {
                let mut l = line_w.lock().unwrap();
                l.line.remove(0);
                l.settling = true;
                l.in_hand_back.push((s_ptr, 0x2000_0000, 0x2001_0000));
                stage_w.store(1, std::sync::atomic::Ordering::Release);
            }
            cv_w.notify_all();
            std::thread::sleep(std::time::Duration::from_millis(20));
            {
                let mut l = line_w.lock().unwrap();
                l.in_hand_back.clear();
                l.settling = false;
                l.written = vec![(0x2002_0000, 0x2003_0000)];
                stage_w.store(2, std::sync::atomic::Ordering::Release);
            }
            cv_w.notify_all();
        });

        let waited = relay::wait_on_line(&line, &cv, relay::WHY_STALE, false, |l| {
            !relay::line_overlaps_stale(l, &stale_hit)
        });
        assert!(waited.is_some());
        assert_eq!(stage.load(std::sync::atomic::Ordering::Acquire), 2);
        assert_eq!(
            line.lock().unwrap().line.len(),
            1,
            "unrelated flight 2 remains in LINE"
        );
        handle.join().unwrap();
    }

    #[test]
    fn range_scoped_why_read_unblocks_without_draining_unrelated_flights() {
        use crate::retire::ExecEpochGuard;
        use crate::shadow::Shadow;
        use std::sync::{Arc, Condvar, Mutex};

        let mut side = vec![0u8; 4 * crate::watch::PAGE];
        let Some(shadow_a) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            0x3000_0000,
            (4 * crate::watch::PAGE) as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return;
        };
        let Some(shadow_b) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            0x3000_0000,
            (4 * crate::watch::PAGE) as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return;
        };
        let ptr_a = Arc::as_ptr(&shadow_a) as usize;
        let make_flight =
            |number: u64, back: Vec<relay::Declared>, readbacks: Vec<(usize, usize)>| {
                relay::Flight {
                    number,
                    fence: number,
                    lists: Vec::new(),
                    card: None,
                    back,
                    readback_ranges: readbacks,
                    _exec_epoch: ExecEpochGuard {
                        scope: None,
                        epoch: 0,
                    },
                }
            };

        let line = Arc::new(Mutex::new(relay::Line {
            line: vec![
                relay::Late::Run(make_flight(
                    1,
                    vec![(shadow_a.clone(), vec![(0x3000_0000, 0x3001_0000)])],
                    vec![(0x3100_0000, 0x3100_4000)],
                )),
                relay::Late::Run(make_flight(
                    2,
                    vec![(shadow_b.clone(), vec![(0x3002_0000, 0x3003_0000)])],
                    Vec::new(),
                )),
            ],
            written: vec![
                (0x3000_0000, 0x3001_0000),
                (0x3002_0000, 0x3003_0000),
                (0x3100_0000, 0x3100_4000),
            ],
            lists: 2,
            target: 2,
            settling: false,
            poke: false,
            landed: Vec::new(),
            in_hand: Vec::new(),
            in_hand_back: Vec::new(),
            pending_woken: Vec::new(),
        }));
        let cv = Arc::new(Condvar::new());

        // Disjoint copy/fill/texture ranges never block.
        assert_eq!(
            relay::wait_on_line(&line, &cv, relay::WHY_READ, false, |l| {
                !relay::line_overlaps_read(l, &[(0x3003_0000, 0x3004_0000)])
            }),
            None
        );
        // Readback range on flight 1 overlaps `line_overlaps_read`.
        assert!(relay::line_overlaps_read(
            &line.lock().unwrap(),
            &[(0x3100_1000, 0x3100_2000)]
        ));

        // Copy overlapping flight 1 waits until flight 1 lands (including its
        // `in_hand_back` phase), but unblocks while unrelated flight 2 remains.
        let stage = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let line_w = line.clone();
        let cv_w = cv.clone();
        let stage_w = stage.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            {
                let mut l = line_w.lock().unwrap();
                l.line.remove(0);
                l.settling = true;
                l.in_hand_back.push((ptr_a, 0x3000_0000, 0x3001_0000));
                l.in_hand_back.push((0, 0x3100_0000, 0x3100_4000));
                // Non-shadow `s_ptr == 0` entry must not match `line_overlaps_stale`.
                assert!(!relay::line_overlaps_stale(
                    &l,
                    &[(ptr_a, 0x3100_0000, 0x3100_4000)]
                ));
                assert!(relay::line_overlaps_read(&l, &[(0x3100_0000, 0x3100_1000)]));
                stage_w.store(1, std::sync::atomic::Ordering::Release);
            }
            cv_w.notify_all();
            std::thread::sleep(std::time::Duration::from_millis(20));
            {
                let mut l = line_w.lock().unwrap();
                l.in_hand_back.clear();
                l.settling = false;
                l.written = vec![(0x3002_0000, 0x3003_0000)];
                stage_w.store(2, std::sync::atomic::Ordering::Release);
            }
            cv_w.notify_all();
        });

        let waited = relay::wait_on_line(&line, &cv, relay::WHY_READ, false, |l| {
            !relay::line_overlaps_read(l, &[(0x3000_0400, 0x3000_0800)])
        });
        assert!(waited.is_some());
        assert_eq!(stage.load(std::sync::atomic::Ordering::Acquire), 2);
        assert_eq!(
            line.lock().unwrap().line.len(),
            1,
            "unrelated flight 2 remains in LINE"
        );
        handle.join().unwrap();
    }

    #[test]
    fn blocked_ranges_scopes_copy_dst_and_prioritizes_run() {
        let Some(h) = host() else {
            return;
        };
        let written = vec![(0x4000_0000usize, 0x4001_0000usize)];

        // Copy reading from `written` into non-written dst is allowed ahead (`None`).
        let list_read = h.new_list().expect("list_read");
        list_read.defer_copy(0x5000_0000 as *mut u8, 0x4000_1000 as *const u8, 256);
        assert!(relay::blocked_ranges(&written, &[&list_read], false).is_none());

        // Copy writing INTO `written` returns only the conflicting dst range.
        let list_dst = h.new_list().expect("list_dst");
        list_dst.defer_copy(0x4000_2000 as *mut u8, 0x5000_0000 as *const u8, 512);
        match relay::blocked_ranges(&written, &[&list_dst], false) {
            Some(relay::LateBlock::Read(ranges)) => {
                assert_eq!(ranges, vec![(0x4000_2000, 0x4000_2200)]);
            }
            other => panic!("expected Read block, got {:?}", other.is_some()),
        }

        // `Step::Run` after a blocking copy returns `LateBlock::Run`.
        list_dst.defer(|| {});
        assert!(matches!(
            relay::blocked_ranges(&written, &[&list_dst], false),
            Some(relay::LateBlock::Run)
        ));
    }

    #[test]
    fn stale_under_flights_ignores_gaps_between_disjoint_refreshes() {
        use crate::shadow::Shadow;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(h) = host() else {
            return;
        };
        let page = crate::watch::PAGE;
        let len = 4 * page;
        // SAFETY: isolated write-watched test buffer freed at the end of the test.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
            .cast::<u8>()
        };
        assert!(!title.is_null());
        let va0 = title as usize;
        let mut side = vec![0u8; len];
        let Some(shadow) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            va0 as u64,
            len as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            // SAFETY: cleanup.
            unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
            return;
        };
        // Mark shadow unwatched so any pending overlap would report stale immediately.
        shadow.mark_unwatched_for_test();
        // Pending shader write ONLY in page 1..2 (middle gap).
        shadow.writing(va0 + page, page, |_, _| None);

        // Submission refreshes ONLY page 0 and page 3 (disjoint around page 1..2).
        let list = h.new_list().expect("list");
        list.defer_refresh(&shadow, va0, page);
        list.defer_refresh(&shadow, va0 + 3 * page, page);
        assert!(
            relay::stale_under_flights(&[&list]).is_empty(),
            "disjoint refreshes must not pull in unrelated pending gap"
        );

        // Adding a refresh overlapping page 1 reports ONLY page 1.
        list.defer_refresh(&shadow, va0 + page, 4096);
        let stale = relay::stale_under_flights(&[&list]);
        assert_eq!(
            stale,
            vec![(
                std::sync::Arc::as_ptr(&shadow) as usize,
                va0 + page,
                va0 + page + 4096
            )]
        );
        // SAFETY: free test allocation.
        unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn ask_once_void_forces_fresh_get_write_watch_after_settle_stale() {
        use std::sync::atomic::Ordering;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        struct CleanupGuard {
            base: usize,
        }
        impl Drop for CleanupGuard {
            fn drop(&mut self) {
                crate::watch::swap_current_stamp(0);
                crate::watch::ask_once_void();
                // SAFETY: `self.base` is a private allocation owned by this test.
                unsafe { VirtualFree(self.base as *mut _, 0, MEM_RELEASE) };
            }
        }

        // SAFETY: private write-watched page simulating title RAM.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        let _cleanup = CleanupGuard { base };

        // Baseline reset so kernel write-watch bit starts clean.
        assert!(crate::watch::writes(base, base + crate::watch::PAGE, false).is_some());

        // Mint a title submission stamp and set this thread's stamp to it so
        // after the first `ask_once`, `current_stamp() <= ASK_STAMP` holds.
        let stamp = crate::watch::TITLE_SUBMITS.fetch_add(1, Ordering::AcqRel) + 1;
        crate::watch::swap_current_stamp(stamp);

        let ranges = [(base, base + crate::watch::PAGE)];
        crate::watch::ask_once(&ranges);
        let (epoch1, _) = crate::watch::asked(base, base + crate::watch::PAGE)
            .expect("first ask_once covers private range");

        // Simulate `settle_stale` landing GPU writes directly into private
        // write-watched title memory without updating `DIR`.
        // SAFETY: committed private `PAGE_READWRITE` memory owned by this test.
        unsafe { (base as *mut u8).write_volatile(0x5A) };

        // Voiding the standing asking clears `ASK_STAMP` and `AHEAD`.
        crate::watch::ask_once_void();
        assert!(
            crate::watch::asked(base, base + crate::watch::PAGE).is_none(),
            "ask_once_void must clear standing ask_once state"
        );

        // Even with `CURRENT_STAMP` still at `stamp` (`<=` the prior `ASK_STAMP`),
        // re-asking must perform a fresh kernel `GetWriteWatch` query and observe
        // the newly dirtied page.
        crate::watch::ask_once(&ranges);
        let (epoch2, written) = crate::watch::asked(base, base + crate::watch::PAGE)
            .expect("second ask_once re-queries private range");
        assert!(epoch2 > epoch1);
        assert!(
            written
                .iter()
                .any(|&(p, w)| p == base / crate::watch::PAGE && w == epoch2),
            "re-ask after ask_once_void must observe newly dirtied write-watch page"
        );
    }

    #[test]
    fn rehash_runs_defaults_on_and_honors_zero_override() {
        assert!(relay::rehash_runs_enabled(None));
        assert!(relay::rehash_runs_enabled(Some("1")));
        assert!(!relay::rehash_runs_enabled(Some("0")));
    }

    #[test]
    fn wide_merge_is_default_on_multicore_and_respects_disable() {
        assert!(relay::wide_merge_enabled(None, Some(4)));
        assert!(relay::wide_merge_enabled(None, Some(16)));
        assert!(!relay::wide_merge_enabled(None, Some(2)));
        assert!(!relay::wide_merge_enabled(None, None));
        assert!(relay::wide_merge_enabled(Some("1"), Some(1)));
        assert!(!relay::wide_merge_enabled(Some("0"), Some(16)));
        assert!(relay::wide_merge_enabled(Some("auto"), Some(4)));
    }

    #[test]
    fn wide_relay_card_merge_matches_sequential_across_pieces_and_shadows() {
        use crate::shadow::Shadow;
        const PIECE: usize = 64 << 10;

        type MergeCaseOut = (usize, Vec<Vec<u8>>, Vec<Vec<u8>>);
        type ShadowSpec = (usize, Vec<(usize, usize)>, Vec<(usize, usize, u64)>);

        let run_case = |parallel: bool| -> Option<MergeCaseOut> {
            // 3 card shadows:
            // - Shadow 0: 24 pieces (1.5 MiB) -> tests partitioning one large shadow across workers (>= 16 pieces).
            // - Shadow 1: 4 pieces (256 KiB) with dirty & sent ranges + overlapping plan-ordered pieces.
            // - Shadow 2: 2 pieces (128 KiB) with unchanged and sparsely modified pieces.
            let specs: &[ShadowSpec] = &[
                (
                    24 * PIECE,
                    vec![],
                    vec![(3 * PIECE + 128, 3 * PIECE + 256, 11)],
                ),
                (
                    4 * PIECE,
                    vec![(PIECE + 64, PIECE + 192)],
                    vec![(16, 48, 8), (2 * PIECE + 32, 2 * PIECE + 96, 6)],
                ),
                (2 * PIECE, vec![], vec![]),
            ];

            let mut titles: Vec<Vec<u8>> = Vec::new();
            let mut sides: Vec<Vec<u8>> = Vec::new();
            let mut shadows: Vec<std::sync::Arc<Shadow>> = Vec::new();
            for (s_idx, (len, dirty, sent)) in specs.iter().enumerate() {
                let base: Vec<u8> = (0..*len)
                    .map(|i| (((s_idx + 1) * 31 + i * 13 + (i >> 8)) & 0xff) as u8)
                    .collect();
                let mut title = base.clone();
                let mut side = base;
                // Simulate newer title CPU writes already staged into `side` for dirty/later-sent ranges.
                for &(a, b) in dirty {
                    for byte in &mut title[a..b] {
                        *byte ^= 0x55;
                    }
                    side[a..b].copy_from_slice(&title[a..b]);
                }
                for &(a, b, send_no) in sent {
                    if send_no > 6 {
                        for byte in &mut title[a..b] {
                            *byte ^= 0x33;
                        }
                        side[a..b].copy_from_slice(&title[a..b]);
                    }
                }
                let shadow = Shadow::new_card_for_test(
                    side.as_mut_ptr(),
                    title.as_ptr() as u64,
                    *len as u64,
                    dirty.clone(),
                    sent.clone(),
                )?;
                titles.push(title);
                sides.push(side);
                shadows.push(shadow);
            }

            // Build fetched GPU buffers (owned here until `relay_merge_card_pieces` returns).
            let mut gpu_bufs: Vec<Vec<u8>> = Vec::new();
            let mut meta: Vec<(usize, usize, usize, u64)> = Vec::new();

            // Shadow 0: 24 disjoint pieces (even pieces modified by GPU, odd pieces unchanged except piece 3).
            for p in 0..24usize {
                let lo = p * PIECE;
                let hi = lo + PIECE;
                let mut buf = sides[0][lo..hi].to_vec();
                if p % 2 == 0 {
                    let start = 17 + (p * 19) % 512;
                    let len = 1024 + p * 37;
                    for b in &mut buf[start..start + len] {
                        *b ^= 0xA7;
                    }
                }
                if p == 3 {
                    // Card holds pre-send-11 bytes in [128..256), plus a real GPU write at [400..500).
                    for byte in &mut buf[128..256] {
                        *byte ^= 0x33;
                    }
                    for b in &mut buf[400..500] {
                        *b ^= 0xC9;
                    }
                }
                gpu_bufs.push(buf);
                meta.push((0, lo, hi, 10));
            }

            // Shadow 1:
            // - Piece 0: two overlapping plans in order (plan 1 after=7, plan 2 after=9) to verify order preservation.
            // - Piece 1: overlaps `dirty` [64..192) + GPU write at [300..420).
            // - Piece 2: sent at 6 <= after (10), GPU writes at [10..120).
            {
                let mut p0_plan1 = sides[1][0..PIECE].to_vec();
                // Pre-send-8 bytes at [16..48), GPU write from plan 1 at [100..200).
                for byte in &mut p0_plan1[16..48] {
                    *byte ^= 0x33;
                }
                for b in &mut p0_plan1[100..200] {
                    *b = 0x11;
                }
                gpu_bufs.push(p0_plan1);
                meta.push((1, 0, PIECE, 7));

                let mut p0_plan2 = sides[1][0..PIECE].to_vec();
                // Plan 2 ran after send 8 (`after=9`), and overwrote [120..240) with 0x22.
                for b in &mut p0_plan2[120..240] {
                    *b = 0x22;
                }
                gpu_bufs.push(p0_plan2);
                meta.push((1, 0, PIECE, 9));

                let mut p1 = sides[1][PIECE..2 * PIECE].to_vec();
                for byte in &mut p1[64..192] {
                    *byte ^= 0x55; // stale card bytes before dirty send
                }
                for b in &mut p1[300..420] {
                    *b ^= 0x6D;
                }
                gpu_bufs.push(p1);
                meta.push((1, PIECE, 2 * PIECE, 10));

                let mut p2 = sides[1][2 * PIECE..3 * PIECE].to_vec();
                for b in &mut p2[10..120] {
                    *b ^= 0x8E;
                }
                gpu_bufs.push(p2);
                meta.push((1, 2 * PIECE, 3 * PIECE, 10));
            }

            // Shadow 2: 1 unchanged piece + 1 sparse-diff piece.
            {
                gpu_bufs.push(sides[2][0..PIECE].to_vec());
                meta.push((2, 0, PIECE, 10));
                let mut p1 = sides[2][PIECE..2 * PIECE].to_vec();
                p1[511] ^= 0xF0;
                p1[PIECE - 7] ^= 0x0F;
                gpu_bufs.push(p1);
                meta.push((2, PIECE, 2 * PIECE, 10));
            }

            let pieces: Vec<relay::RelayCardPiece> = meta
                .into_iter()
                .zip(gpu_bufs.iter())
                .map(|((s_idx, lo, hi, after), buf)| relay::RelayCardPiece {
                    shadow: shadows[s_idx].clone(),
                    lo,
                    hi,
                    gpu_ptr: buf.as_ptr() as usize,
                    after,
                })
                .collect();

            let written = relay::relay_merge_card_pieces(pieces, parallel);
            drop(shadows);
            Some((written, titles, sides))
        };

        let Some((seq_written, seq_titles, seq_sides)) =
            run_case(relay::wide_merge_enabled(Some("0"), Some(16)))
        else {
            return;
        };
        let Some((par_written, par_titles, par_sides)) =
            run_case(relay::wide_merge_enabled(None, Some(16)))
        else {
            return;
        };

        assert!(seq_written > 0, "expected non-zero merged bytes");
        assert_eq!(
            par_written, seq_written,
            "parallel written byte count must match sequential"
        );
        assert_eq!(
            par_titles, seq_titles,
            "parallel merged title memory must match sequential"
        );
        assert_eq!(
            par_sides, seq_sides,
            "parallel merged CPU staging must match sequential"
        );
        // Verify plan 2's newer overlapping write won over plan 1's older write on Shadow 1 piece 0.
        assert!(par_titles[1][120..240].iter().all(|&b| b == 0x22));
    }

    /// Multi-threaded lock-order stress test verifying that:
    /// 1. `HostList::run_deferred` releases `list.deferred()` before running steps
    ///    (so a step that waits on `LINE_LANDED` — such as `batch.flush()` ->
    ///    `Shadow::update_busy()` -> `settle_card_busy()` — never deadlocks with
    ///    `Host::relay_land`'s `read_backs(list)` acquiring `list.deferred()`),
    ///    and restores `deferred.copies` after completion.
    /// 2. Parallel `relay_merge_card_pieces(..., true)` across `victory-wide`
    ///    workers runs concurrently with `copy_back_done` -> `LINE` -> `LINE_LANDED`
    ///    without deadlocking or leaving stale pending ranges visible to waiters.
    #[test]
    fn relay_merge_wide_and_run_deferred_lock_order_stress() {
        use std::sync::{Arc, Condvar};
        const PIECE: usize = 64 << 10;
        const PIECES: usize = 16;
        const LEN: usize = PIECES * PIECE;

        let _retire = isolated_retire_scope();
        let Some(host) = host() else { return };

        let title = vec![0x10u8; LEN];
        let mut side = vec![0x10u8; LEN];
        let title_addr = title.as_ptr() as usize;
        let Some(shadow) = crate::shadow::Shadow::new_card_for_test(
            side.as_mut_ptr(),
            title_addr as u64,
            LEN as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return;
        };

        let list = Arc::new(host.new_list().expect("host list"));
        let line = Arc::new(Mutex::new(relay::Line::new_for_test()));
        let landed = Arc::new(Condvar::new());
        let step_entered = Arc::new((Mutex::new(false), Condvar::new()));

        let line_step = line.clone();
        let landed_step = landed.clone();
        let entered_step = step_entered.clone();
        list.defer(move || {
            {
                let mut e = entered_step.0.lock().unwrap();
                *e = true;
                entered_step.1.notify_all();
            }
            // Mirrors `settle_card_busy` waiting on `LINE_LANDED` from inside `run_deferred`.
            let waited =
                relay::wait_on_line(&line_step, &landed_step, relay::WHY_STALE, false, |l| {
                    !l.settling
                });
            assert!(waited.is_some());
        });

        let gpu_bufs: Vec<Vec<u8>> = (0..PIECES)
            .map(|p| {
                let mut b = vec![0x10u8; PIECE];
                b[p * 17] = (0x80 + p) as u8;
                b
            })
            .collect();

        for round in 0..8usize {
            shadow.writing(title_addr, LEN, |_, _| None);
            {
                line.lock().unwrap().settling = true;
                *step_entered.0.lock().unwrap() = false;
            }
            let pieces: Vec<relay::RelayCardPiece> = (0..PIECES)
                .map(|p| relay::RelayCardPiece {
                    shadow: shadow.clone(),
                    lo: p * PIECE,
                    hi: (p + 1) * PIECE,
                    gpu_ptr: gpu_bufs[p].as_ptr() as usize,
                    after: round as u64 + 1,
                })
                .collect();

            let list_relay = list.clone();
            let shadow_relay = shadow.clone();
            let line_relay = line.clone();
            let landed_relay = landed.clone();
            let entered_relay = step_entered.clone();
            let relay_thread = std::thread::spawn(move || {
                // Wait until `run_deferred` is actively inside its step (blocked on `LINE_LANDED`).
                let mut e = entered_relay.0.lock().unwrap();
                while !*e {
                    e = entered_relay.1.wait(e).unwrap();
                }
                drop(e);
                // 1. Parallel `relay_merge_card_pieces` across `victory-wide`.
                let _ = relay::relay_merge_card_pieces(pieces, true);
                // 2. `read_backs` acquires `list.deferred()` while `run_deferred` is waiting.
                read_backs(&list_relay);
                // 3. `copy_back_done` strictly BEFORE clearing `settling` and notifying `LINE_LANDED`.
                shadow_relay.copy_back_done(&[(title_addr, title_addr + LEN)]);
                {
                    line_relay.lock().unwrap().settling = false;
                }
                landed_relay.notify_all();
            });

            let mut batch = Batch::default();
            let mut submitted = 0u64;
            list.run_deferred(&mut batch, host, &mut submitted);
            relay_thread.join().expect("relay thread panicked");

            // `copy_back_done` ran strictly before `LINE_LANDED` woke `run_deferred`.
            assert!(!shadow.writes_pending());
            // `RestoreCopies` put `copies` back so the closed list stays resubmittable.
            assert_eq!(list.deferred().copies.len(), 1);
        }

        // Verify all 16 GPU piece writes landed cleanly via parallel `relay_merge_card_pieces`.
        for p in 0..PIECES {
            assert_eq!(title[p * PIECE + p * 17], (0x80 + p) as u8);
        }
        drop(shadow);
        let _ = side;
    }

    #[test]
    fn read_sparse_and_pieces_checked_retry_when_lazy_block_materializes_mid_query() {
        use crate::watch::PAGE;
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(h) = host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: anonymous 1-page section so `lazy_back` can defer and merge via own view.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 1 page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va = ptr as usize;
        let block = va / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(block);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: `ptr` is mapped RW for `PAGE` bytes.
        unsafe { std::ptr::write_bytes(ptr, 0x11, PAGE) };

        let mut side = vec![0x11u8; PAGE];
        let gpu_write = |shadow: &crate::shadow::Shadow, fill: u8| {
            let (up, up_ptr) = h.upload_buffer(PAGE as u64).expect("upload");
            // SAFETY: `up_ptr` covers `PAGE` mapped upload bytes.
            unsafe {
                std::ptr::write_bytes(up_ptr, 0x11, PAGE);
                std::ptr::write_bytes(up_ptr, fill, 64);
            }
            let list = h.new_list().expect("list");
            // SAFETY: live upload and shadow GPU buffers.
            unsafe {
                list.list
                    .CopyBufferRegion(&shadow.buffer, 0, &up, 0, PAGE as u64);
            }
            list.close();
            h.execute(&[&list]);
            list.reset();
        };

        if let Some(shadow) = crate::shadow::Shadow::new_card_sized_for_test(
            side.as_mut_ptr(),
            va as u64,
            PAGE as u64,
        ) {
            // 1. Force materialization strictly between `VirtualQuery` (seeing `PAGE_NOACCESS`)
            //    and `range_has_lazy_block` in `read_sparse_checked`. Without the
            //    `remove_generation()` check, `read_sparse_checked` would zero-fill `dst`.
            gpu_write(&shadow, 0xA5);
            assert!(crate::lazy_back::try_defer_page_for_test(
                &shadow, 0, PAGE, 0
            ));
            assert!(crate::lazy_back::has_block(block));

            AFTER_VIRTUAL_QUERY_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    assert!(crate::lazy_back::materialize_range(va, PAGE));
                }));
            });
            let mut dst = vec![0xEEu8; PAGE];
            let ok = read_sparse_checked(ptr, dst.as_mut_ptr(), PAGE);
            AFTER_VIRTUAL_QUERY_HOOK.with(|hook| *hook.borrow_mut() = None);
            assert!(ok, "read_sparse_checked must retry and succeed");
            assert_eq!(
                &dst[..64],
                &[0xA5; 64],
                "read_sparse_checked must return materialized GPU bytes instead of zero-filling"
            );
            assert!(!crate::lazy_back::has_block(block));

            // 2. Force materialization strictly between the failing `read` and
            //    `range_has_lazy_block` in `read_pieces_checked`.
            crate::lazy_back::clear_hot_for_test_pub(block);
            gpu_write(&shadow, 0x5B);
            assert!(crate::lazy_back::try_defer_page_for_test(
                &shadow, 0, PAGE, 0
            ));
            assert!(crate::lazy_back::has_block(block));

            AFTER_READ_PIECES_FAIL_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    assert!(crate::lazy_back::materialize_range(va, PAGE));
                }));
            });
            dst.fill(0xEE);
            let ok = read_pieces_checked(ptr, dst.as_mut_ptr(), PAGE);
            AFTER_READ_PIECES_FAIL_HOOK.with(|hook| *hook.borrow_mut() = None);
            assert!(ok, "read_pieces_checked must retry and succeed");
            assert_eq!(
                &dst[..64],
                &[0x5B; 64],
                "read_pieces_checked must return materialized GPU bytes instead of zero-filling"
            );
            assert!(!crate::lazy_back::has_block(block));

            drop(shadow);
        }

        crate::lazy_back::clear_hot_for_test_pub(block);
        crate::watch::title_memory_remapped(va, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn submission_aborts_before_execute_when_dirty_lazy_refresh_fails() {
        use crate::watch::PAGE;
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(h) = host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: anonymous 1-page section so `lazy_back` can defer and merge via own view.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 1 page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va = ptr as usize;
        let block = va / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(block);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed committed view memory.
        unsafe { std::ptr::write_bytes(ptr, 0x11, PAGE) };

        let mut side = vec![0x11u8; PAGE];
        let shadow = crate::shadow::Shadow::new_card_sized_for_test(
            side.as_mut_ptr(),
            va as u64,
            PAGE as u64,
        )
        .expect("card shadow");
        // Establish watched baseline epoch on page 0.
        let _ = shadow.update(va, PAGE);
        shadow.clear_card_dirty_for_test();

        // Seed GPU bytes `0xBB` at offset 0..16, then write CPU byte `0x77` at offset 8192
        // before arming page 0 so page 0 is a dirty lazy-owned page.
        let (up, up_ptr) = h.upload_buffer(PAGE as u64).expect("upload");
        // SAFETY: mapped upload buffer.
        unsafe {
            std::ptr::write_bytes(up_ptr, 0x11, PAGE);
            std::ptr::write_bytes(up_ptr, 0xBB, 16);
        }
        let seed_list = h.new_list().expect("seed list");
        // SAFETY: live upload and shadow buffers.
        unsafe {
            seed_list
                .list
                .CopyBufferRegion(&shadow.buffer, 0, &up, 0, PAGE as u64);
        }
        seed_list.close();
        h.execute(&[&seed_list]);
        seed_list.reset();

        // SAFETY: dirty CPU write in another 4 KiB sub-part of page 0.
        unsafe { ptr.add(8192).write(0x77) };
        crate::lazy_back::clear_hot_for_test_pub(block);
        assert!(crate::lazy_back::try_defer_page_for_test(
            &shadow, 0, PAGE, 0
        ));
        assert!(crate::lazy_back::has_block(block));

        let (rb, rb_ptr) = h.shared_buffer(PAGE as u64).expect("readback");
        // SAFETY: mapped host-visible buffer initialized to sentinel `0xEE`.
        unsafe { std::ptr::write_bytes(rb_ptr, 0xEE, PAGE) };

        // 1. `Batch::flush` path (`Step::Refresh`): injected materialization failure
        //    must abort `Host::execute` BEFORE `ExecuteCommandLists` so the GPU list
        //    never runs with stale bytes.
        let list_flush = h.new_list().expect("list_flush");
        list_flush.defer_refresh(&shadow, va, PAGE);
        // SAFETY: live shadow and readback buffers.
        unsafe {
            list_flush
                .list
                .CopyBufferRegion(&rb, 0, &shadow.buffer, 0, PAGE as u64);
        }
        list_flush.close();
        {
            let _fail_panics = fail_panics_for_test();
            crate::lazy_back::set_fail_next_fetches_for_test(2);
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                h.execute(&[&list_flush]);
            }));
            crate::lazy_back::set_fail_next_fetches_for_test(0);
            assert!(
                res.is_err(),
                "Host::execute must abort when Batch::flush fails to materialize dirty lazy page"
            );
        }
        // SAFETY: mapped readback buffer must remain untouched (`0xEE`), proving the list never ran.
        let rb_after_flush = unsafe { std::slice::from_raw_parts(rb_ptr, 64) };
        assert!(
            rb_after_flush.iter().all(|&b| b == 0xEE),
            "command list must NOT execute after failed dirty lazy refresh in Batch::flush"
        );
        assert!(
            crate::lazy_back::has_block(block),
            "dirty lazy page must remain armed after aborted Batch::flush submission"
        );
        list_flush.reset();

        // 2. `Batch::refresh_ahead` path (`Step::Refresh` + `Step::GpuWrite`): injected
        //    materialization failure must ALSO abort `Host::execute` BEFORE `ExecuteCommandLists`.
        let list_ahead = h.new_list().expect("list_ahead");
        list_ahead.defer_refresh(&shadow, va, PAGE);
        list_ahead.defer_gpu_write(&shadow, va, PAGE);
        // SAFETY: live shadow and readback buffers.
        unsafe {
            list_ahead
                .list
                .CopyBufferRegion(&rb, 0, &shadow.buffer, 0, PAGE as u64);
        }
        list_ahead.close();
        {
            let _fail_panics = fail_panics_for_test();
            crate::lazy_back::set_fail_next_fetches_for_test(2);
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                h.execute(&[&list_ahead]);
            }));
            crate::lazy_back::set_fail_next_fetches_for_test(0);
            assert!(
                res.is_err(),
                "Host::execute must abort when Batch::refresh_ahead fails to materialize dirty lazy page"
            );
        }
        // SAFETY: mapped readback buffer must remain untouched (`0xEE`), proving the list never ran.
        let rb_after_ahead = unsafe { std::slice::from_raw_parts(rb_ptr, 64) };
        assert!(
            rb_after_ahead.iter().all(|&b| b == 0xEE),
            "command list must NOT execute after failed dirty lazy refresh in Batch::refresh_ahead"
        );
        assert!(
            crate::lazy_back::has_block(block),
            "dirty lazy page must remain armed after aborted Batch::refresh_ahead submission"
        );
        list_ahead.reset();

        // 3. Recovery: once DMA succeeds, a normal submission refreshes and runs cleanly.
        assert!(crate::lazy_back::materialize_range(va, PAGE));
        assert!(!crate::lazy_back::has_block(block));
        drop(shadow);

        crate::lazy_back::clear_hot_for_test_pub(block);
        crate::watch::title_memory_remapped(va, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }
}

/// A closed list can be submitted more than once: each submission's
/// breadcrumb report gets the list's noted commands, and the list keeps
/// them until it resets.
#[cfg(feature = "diagnostics")]
#[test]
fn every_submission_of_a_list_gets_its_noted_commands() {
    let notes: Mutex<Vec<Note>> = Mutex::new(vec![
        ("a draw".to_owned(), None),
        ("a copy".to_owned(), Some((0x1000, 16))),
    ]);
    let first = submitted_notes(&notes);
    let second = submitted_notes(&notes);
    assert_eq!(first.len(), 2);
    assert_eq!(first, second);
    assert_eq!(notes.lock().unwrap().len(), 2);
}
