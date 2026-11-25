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
