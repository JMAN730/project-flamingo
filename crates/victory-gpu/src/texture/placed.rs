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
    tiled: Vec<u8>,
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
            let asked = watched.map_or(0, |(stamp, _)| stamp);
            if last.tiled == tiled {
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
            let mut regions = Vec::new();
            let mut image = vec![0; *bytes];
            let mips = self.layout.desc.mips;
            let element = self.layout.desc.element_bytes as usize;
            for (sub, &fp) in footprints.iter().enumerate() {
                let sub = sub as u32;
                let f = fp.Footprint;
                let pitch = f.RowPitch as usize;
                let depth_pitch = pitch * f.Height as usize;
                let output = &mut image[fp.Offset as usize..];
                self.layout
                    .copy_spans(
                        size,
                        sub % mips,
                        sub / mips,
                        output.len(),
                        pitch,
                        depth_pitch,
                        |src, dst, len| {
                            let new = &tiled[src..src + len];
                            output[dst..dst + len].copy_from_slice(new);
                            if last.tiled.is_empty() {
                                return;
                            }
                            let old = &last.tiled[src..src + len];
                            if old == new {
                                return;
                            }
                            let x = (dst % pitch / element) as u32;
                            let y = ((dst % depth_pitch) / pitch) as u32;
                            let z = (dst / depth_pitch) as u32;
                            let mut run = None;
                            let mut emit = |left, right| {
                                regions.push((
                                    sub,
                                    fp,
                                    D3D12_BOX {
                                        left: x + left,
                                        top: y,
                                        front: z,
                                        right: x + right,
                                        bottom: y + 1,
                                        back: z + 1,
                                    },
                                ))
                            };
                            for (i, (a, b)) in old
                                .chunks_exact(element)
                                .zip(new.chunks_exact(element))
                                .enumerate()
                            {
                                if a != b {
                                    run.get_or_insert(i as u32);
                                } else if let Some(left) = run.take() {
                                    emit(left, i as u32);
                                }
                            }
                            if let Some(left) = run {
                                emit(left, (len / element) as u32);
                            }
                        },
                    )
                    .ok()?;
                if last.tiled.is_empty() {
                    regions.push((
                        sub,
                        fp,
                        D3D12_BOX {
                            left: 0,
                            top: 0,
                            front: 0,
                            right: f.Width,
                            bottom: f.Height,
                            back: f.Depth,
                        },
                    ));
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
            last.tiled.clear();
            last.tiled.extend_from_slice(tiled);
            last.asked = asked;
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

/// Buffers from the previous submission, keyed by their exact footprint size.
#[derive(Default)]
pub(crate) struct ReadbackCache(std::collections::BTreeMap<usize, Vec<MappedReadback>>);

struct MappedReadback {
    buffer: ID3D12Resource,
    data: *const u8,
}

struct Readback {
    copy: std::sync::Arc<crate::resource::HostTexture>,
    mapped: MappedReadback,
}

impl Feedback {
    pub(crate) fn record<'a>(
        host: &crate::host::Host,
        written: impl Iterator<Item = &'a std::sync::Arc<crate::resource::HostTexture>>,
        cache: &mut ReadbackCache,
    ) -> Option<Self> {
        use windows::core::Interface;
        let requests: Vec<_> = written
            .filter_map(|copy| {
                let (_, bytes) = copy.placed.get()?.footprints.get()?;
                let cached = cache.0.get_mut(bytes).and_then(Vec::pop);
                Some((copy, *bytes, cached))
            })
            .collect();
        // Release unused buffers before allocating any replacements.
        cache.0.clear();
        let copies: Vec<_> = requests
            .into_iter()
            .filter_map(|(copy, bytes, cached)| {
                let mapped = cached.or_else(|| {
                    let (buffer, data) = host.readback_buffer(bytes as u64)?;
                    Some(MappedReadback { buffer, data })
                })?;
                Some(Readback {
                    copy: copy.clone(),
                    mapped,
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
                        pResource: std::mem::transmute_copy(&rb.mapped.buffer),
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

    pub(crate) fn finish(self, cache: &mut ReadbackCache) {
        for rb in self.copies {
            let placed = rb.copy.placed.get().unwrap();
            let (footprints, bytes) = placed.footprints.get().unwrap();
            // SAFETY: the CPU-readable buffer contains this many bytes, and the GPU fence has completed.
            let gpu = unsafe { std::slice::from_raw_parts(rb.mapped.data, *bytes) };
            placed.finish(footprints, gpu);
            cache.0.entry(*bytes).or_default().push(rb.mapped);
        }
    }
}

impl Placed {
    fn finish(&self, footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT], gpu: &[u8]) {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if last.tiled.is_empty() {
            return;
        }
        const PAGE: usize = 4096;
        let mut pages = std::collections::BTreeMap::new();
        for (sub, fp) in footprints.iter().enumerate() {
            let sub = sub as u32;
            let mips = self.layout.desc.mips;
            let pitch = fp.Footprint.RowPitch as usize;
            let input = &gpu[fp.Offset as usize..];
            let visited = self.layout.copy_spans(
                last.tiled.len(),
                sub % mips,
                sub / mips,
                input.len(),
                pitch,
                pitch * fp.Footprint.Height as usize,
                |mut at, linear, len| {
                    let mut src = &input[linear..linear + len];
                    if src == &last.tiled[at..at + len] {
                        return;
                    }
                    while !src.is_empty() {
                        let offset = at % PAGE;
                        let n = src.len().min(PAGE - offset);
                        if src[..n] != last.tiled[at..at + n] {
                            let start = at - offset;
                            let page = pages.entry(start).or_insert_with(|| {
                                last.tiled[start..(start + PAGE).min(last.tiled.len())].to_vec()
                            });
                            page[offset..offset + n].copy_from_slice(&src[..n]);
                        }
                        at += n;
                        src = &src[n..];
                    }
                },
            );
            if visited.is_err() {
                return;
            }
        }
        // Only pages containing GPU changes need another read of guest memory.
        // Keep newer CPU edits. Leave them different from the GPU baseline so
        // the next submission uploads them.
        let mut cpu = [0u8; PAGE];
        let mut merged = [0u8; PAGE];
        for (start, page) in pages {
            let cpu = &mut cpu[..page.len()];
            let merged = &mut merged[..page.len()];
            let address = self.address + start;
            if !crate::host::read_into(address as *const u8, cpu) {
                continue;
            }
            let baseline = &mut last.tiled[start..start + page.len()];
            for i in 0..page.len() {
                merged[i] = if cpu[i] == baseline[i] {
                    page[i]
                } else {
                    cpu[i]
                };
            }
            if let Some(first) = merged.iter().zip(cpu.iter()).position(|(a, b)| a != b) {
                let end = merged
                    .iter()
                    .zip(cpu.iter())
                    .rposition(|(a, b)| a != b)
                    .unwrap()
                    + 1;
                if !crate::host::write_title_memory(address + first, &merged[first..end]) {
                    continue;
                }
            }
            baseline.copy_from_slice(&page);
            last.asked = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Dimension, Swizzle};

    #[test]
    fn feedback_preserves_cpu_edits_padding_and_other_array_levels() {
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S] {
            let layout = Layout::new(&Desc {
                dimension: Dimension::Tex2d,
                width: 65,
                height: 35,
                depth: 2,
                mips: 2,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle,
            })
            .unwrap();
            let mut backing = vec![0x33; layout.size as usize];
            let placed = Placed::new(backing.as_mut_ptr() as usize, layout);
            placed.last.lock().unwrap().tiled = backing.clone();
            let mut expected = backing.clone();
            let mut footprints = Vec::new();
            let mut gpu = Vec::new();
            for sub in 0..4 {
                let mip = sub % 2;
                let slice = sub / 2;
                let m = &placed.layout.mips[mip as usize];
                let pitch = (m.width * 4).next_multiple_of(256);
                let offset = gpu.len().next_multiple_of(512);
                let size = (pitch * (m.height - 1) + m.width * 4) as usize;
                gpu.resize(offset + size, 0x77); // host row padding differs from guest memory
                for y in 0..m.height {
                    let start = offset + (y * pitch) as usize;
                    gpu[start..start + (m.width * 4) as usize].fill(0x33);
                }
                footprints.push(D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: offset as u64,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Width: m.width,
                        Height: m.height,
                        Depth: 1,
                        RowPitch: pitch,
                        ..Default::default()
                    },
                });
                if sub == 3 {
                    let (x, y) = (m.width - 1, m.height - 1);
                    let at = placed.layout.offset(mip, x, y, slice).unwrap() as usize;
                    let start = offset + (y * pitch + x * 4) as usize;
                    gpu[start..start + 4].copy_from_slice(&[1, 2, 3, 4]);
                    backing[at + 1] = 0x99;
                    expected[at..at + 4].copy_from_slice(&[1, 0x99, 3, 4]);
                }
            }
            placed.finish(&footprints, &gpu);
            assert_eq!(backing, expected, "{swizzle:?}");
        }
    }
}
