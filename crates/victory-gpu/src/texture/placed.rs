//! Guest backing for a placed colour UAV. A host shader can change some texels
//! while the CPU changes others between submissions. Compare consecutive guest
//! images and upload only the guest texels that changed.
use std::sync::{Mutex, OnceLock};
use victory_formats::texture::Layout;
use windows::Win32::Graphics::Direct3D12::*;

pub(crate) struct Placed {
    address: usize,
    layout: Layout,
    footprints: OnceLock<(Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, usize)>,
    last: Mutex<Snapshot>,
}

#[derive(Default)]
struct Snapshot {
    image: Vec<u8>,
    hash: Option<blake3::Hash>,
    asked: u64,
}

pub(crate) struct Upload {
    buffer: ID3D12Resource,
    regions: Vec<(u32, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_BOX)>,
}

impl Placed {
    pub(crate) fn new(address: usize, layout: Layout) -> Self {
        Self {
            address,
            layout,
            footprints: OnceLock::new(),
            last: Mutex::default(),
        }
    }

    pub(crate) fn changes(&self, texture: &ID3D12Resource) -> Option<Upload> {
        let size = usize::try_from(self.layout.size).ok()?;
        let end = self.address.checked_add(size)?;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let watched = crate::watch::writes(self.address, end, false);
        if let Some((_, pages)) = &watched
            && last.asked != 0
            && pages.iter().all(|&(_, stamp)| stamp <= last.asked)
        {
            return None;
        }
        super::SCRATCH.with_borrow_mut(|scratch| {
            if scratch.len() < size {
                scratch.resize(size, 0);
            }
            let tiled = &mut scratch[..size];
            if !crate::host::read_into(self.address as *const u8, tiled) {
                last.asked = 0;
                return None;
            }
            let hash = blake3::hash(tiled);
            let asked = watched.map_or(0, |(stamp, _)| stamp);
            if last.hash == Some(hash) {
                last.asked = asked;
                return None;
            }
            let host = crate::host::host()?;
            // SAFETY: a live host texture and enough footprints for its subresources.
            let (footprints, bytes) = self.footprints.get_or_init(|| unsafe {
                let d = texture.GetDesc();
                let count = u32::from(d.MipLevels)
                    * if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                let mut footprints = vec![Default::default(); count as usize];
                let mut size = 0;
                host.device.GetCopyableFootprints(
                    &d,
                    0,
                    count,
                    0,
                    Some(footprints.as_mut_ptr()),
                    None,
                    None,
                    Some(&mut size),
                );
                (footprints, size as usize)
            });
            let mut image = vec![0; *bytes];
            let mut regions = Vec::new();
            let mips = self.layout.desc.mips;
            let element = self.layout.desc.element_bytes as usize;
            for (sub, fp) in footprints.iter().enumerate() {
                let sub = sub as u32;
                let f = fp.Footprint;
                let pitch = f.RowPitch as usize;
                let offset = fp.Offset as usize;
                self.layout
                    .detile(
                        tiled,
                        sub % mips,
                        sub / mips,
                        &mut image[offset..],
                        pitch,
                        pitch * f.Height as usize,
                    )
                    .ok()?;
                if last.image.is_empty() {
                    regions.push((
                        sub,
                        *fp,
                        D3D12_BOX {
                            left: 0,
                            top: 0,
                            front: 0,
                            right: f.Width,
                            bottom: f.Height,
                            back: f.Depth,
                        },
                    ));
                    continue;
                }
                for z in 0..f.Depth {
                    for y in 0..f.Height {
                        let start = offset + (z as usize * f.Height as usize + y as usize) * pitch;
                        let row = start..start + f.Width as usize * element;
                        let old = &last.image[row.clone()];
                        let new = &image[row];
                        if old == new {
                            continue;
                        }
                        let mut run = None;
                        for (x, (a, b)) in old
                            .chunks_exact(element)
                            .zip(new.chunks_exact(element))
                            .enumerate()
                        {
                            if a != b {
                                run.get_or_insert(x as u32);
                            } else if let Some(left) = run.take() {
                                regions.push((
                                    sub,
                                    *fp,
                                    D3D12_BOX {
                                        left,
                                        top: y,
                                        front: z,
                                        right: x as u32,
                                        bottom: y + 1,
                                        back: z + 1,
                                    },
                                ));
                            }
                        }
                        if let Some(left) = run {
                            regions.push((
                                sub,
                                *fp,
                                D3D12_BOX {
                                    left,
                                    top: y,
                                    front: z,
                                    right: f.Width,
                                    bottom: y + 1,
                                    back: z + 1,
                                },
                            ));
                        }
                    }
                }
            }
            let upload = if regions.is_empty() {
                None
            } else {
                let (buffer, data) = host.upload_buffer(*bytes as u64)?;
                // SAFETY: the fresh upload buffer has the same footprint size as image.
                unsafe { std::ptr::copy_nonoverlapping(image.as_ptr(), data, *bytes) };
                Some(Upload { buffer, regions })
            };
            *last = Snapshot {
                image,
                hash: Some(hash),
                asked,
            };
            upload
        })
    }
}

impl Upload {
    pub(crate) fn record(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) {
        for &(sub, fp, region) in &self.regions {
            // SAFETY: the locations borrow resources kept alive until the submission finishes.
            unsafe {
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: sub,
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&self.buffer),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: fp,
                    },
                };
                list.CopyTextureRegion(
                    &to,
                    region.left,
                    region.top,
                    region.front,
                    &from,
                    Some(&region),
                );
            }
        }
    }
}

/// GPU writes to guest-visible backing become visible before guest fence signals.
pub(crate) struct Feedback {
    _allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12CommandList,
    copies: Vec<Readback>,
}

struct Readback {
    copy: std::sync::Arc<crate::resource::HostTexture>,
    buffer: ID3D12Resource,
    data: *const u8,
}

impl Feedback {
    pub(crate) fn record<'a>(
        host: &crate::host::Host,
        written: impl Iterator<Item = &'a std::sync::Arc<crate::resource::HostTexture>>,
    ) -> Option<Self> {
        use windows::core::Interface;
        let copies: Vec<_> = written
            .filter_map(|copy| {
                let (_, bytes) = copy.placed.get()?.footprints.get()?;
                let (buffer, data) = host.readback_buffer(*bytes as u64)?;
                Some(Readback {
                    copy: copy.clone(),
                    buffer,
                    data,
                })
            })
            .collect();
        if copies.is_empty() {
            return None;
        }
        // SAFETY: the allocator and resources are held until the host fence completes.
        unsafe {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap_or_else(|e| {
                    crate::host::fail(format_args!("placed texture readback allocator: {e}"))
                });
            let list: ID3D12GraphicsCommandList = host
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .unwrap_or_else(|e| {
                    crate::host::fail(format_args!("placed texture readback list: {e}"))
                });
            for rb in &copies {
                let texture = &rb.copy.texture;
                let state = D3D12_RESOURCE_STATES(
                    rb.copy.state.load(std::sync::atomic::Ordering::Relaxed) as i32,
                );
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    list.ResourceBarrier(&[crate::host::transition(
                        texture,
                        state,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                }
                let placed = rb.copy.placed.get().unwrap();
                for (sub, fp) in placed.footprints.get().unwrap().0.iter().enumerate() {
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&rb.buffer),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: *fp,
                        },
                    };
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(texture),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: sub as u32,
                        },
                    };
                    list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                }
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    list.ResourceBarrier(&[crate::host::transition(
                        texture,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        state,
                    )]);
                }
            }
            list.Close().unwrap_or_else(|e| {
                crate::host::fail(format_args!("placed texture readback did not close: {e}"))
            });
            Some(Self {
                _allocator: allocator,
                list: list.cast().expect("graphics command list"),
                copies,
            })
        }
    }

    pub(crate) fn finish(self) {
        for rb in &self.copies {
            let placed = rb.copy.placed.get().unwrap();
            let (footprints, bytes) = placed.footprints.get().unwrap();
            // SAFETY: the CPU-readable buffer contains this many bytes, and the GPU fence has completed.
            let gpu = unsafe { std::slice::from_raw_parts(rb.data, *bytes) };
            placed.finish(footprints, gpu);
        }
    }
}

impl Placed {
    fn finish(&self, footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT], gpu: &[u8]) {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if last.image.len() != gpu.len() || last.image == gpu {
            return;
        }
        let size = self.layout.size as usize;
        super::SCRATCH.with_borrow_mut(|scratch| {
            scratch.resize(size, 0);
            if !crate::host::read_into(self.address as *const u8, scratch) {
                return;
            }
            let mut before = scratch.clone();
            let mut after = scratch.clone();
            for (sub, fp) in footprints.iter().enumerate() {
                let mip = sub as u32 % self.layout.desc.mips;
                let slice = sub as u32 / self.layout.desc.mips;
                let pitch = fp.Footprint.RowPitch as usize;
                let depth_pitch = pitch * fp.Footprint.Height as usize;
                let start = fp.Offset as usize;
                if self
                    .layout
                    .tile(
                        &mut before,
                        mip,
                        slice,
                        &last.image[start..],
                        pitch,
                        depth_pitch,
                    )
                    .is_err()
                    || self
                        .layout
                        .tile(&mut after, mip, slice, &gpu[start..], pitch, depth_pitch)
                        .is_err()
                {
                    return;
                }
            }
            // Preserve padding and CPU edits made while the submission was running.
            let mut concurrent = false;
            for ((out, old), &cpu) in after.iter_mut().zip(&before).zip(scratch.iter()) {
                if cpu != *old {
                    concurrent |= *out != cpu;
                    *out = cpu;
                }
            }
            for (page, (src, dst)) in after.chunks(4096).zip(scratch.chunks(4096)).enumerate() {
                if src == dst {
                    continue;
                }
                let first = src.iter().zip(dst).position(|(a, b)| a != b).unwrap();
                let end = src.iter().zip(dst).rposition(|(a, b)| a != b).unwrap() + 1;
                if !crate::host::write_title_memory(
                    self.address + page * 4096 + first,
                    &src[first..end],
                ) {
                    return;
                }
            }
            last.image.copy_from_slice(gpu);
            last.hash = (!concurrent).then(|| blake3::hash(&after));
            last.asked = 0;
        });
    }
}
