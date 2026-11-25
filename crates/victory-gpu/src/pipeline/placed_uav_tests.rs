use super::{PIPELINE_IIDS, Shader, Translated, hand_out, host_pipeline};
use crate::object::{S_OK, This, guid, slot};
use std::sync::{Arc, OnceLock};
use victory_shader::native::{ComputeIo, StageIo};
use victory_shader::{
    Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, emit, io,
    ir, resource, structure,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
use windows::core::Interface;

/// Synthetic compute shader storing four root constants at texel (0, 0).
fn store_shader() -> Arc<Shader> {
    let words: [u32; 12] = [
        0xBE85_0380, // s_mov_b32 s5, 0: high word of the table address
        0xF40C_0202,
        0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0
        0x7E02_0200,
        0x7E04_0201,
        0x7E06_0202,
        0x7E08_0203, // v_mov_b32 v[1:4], s[0:3]
        0x7E0A_0280,
        0x7E0C_0280, // v_mov_b32 v5/v6, 0: texel coordinates
        0xF020_1F08,
        0x0002_0105, // image_store v[1:4], v[5:6], s[8:15], dim:2D, dmask:0xf
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    // Compute has no input/output signatures. Supply just the DXBC header.
    let mut container = vec![0; 32];
    container[..4].copy_from_slice(b"DXBC");
    container[20..24].copy_from_slice(&1u32.to_le_bytes());
    container[24..28].copy_from_slice(&32u32.to_le_bytes());
    let c = Container::parse(&container).unwrap();
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let host = emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
    Arc::new(Shader {
        stage: "compute",
        hash: blake3::hash(&native.code).to_hex()[..16].to_owned(),
        container,
        translated: OnceLock::from(Some(Translated {
            host,
            used: native.used,
        })),
        wave64: OnceLock::new(),
    })
}

struct Guest(This);

impl Drop for Guest {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

/// A separately recorded guest CopyTextureRegion from mip zero into a CPU-visible buffer.
struct TexelReadback {
    list: Guest,
    destination: Guest,
    _allocator: Guest,
}

impl TexelReadback {
    fn new(texture: This) -> Self {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            crate::list::create_allocator(
                null,
                0,
                std::ptr::from_ref(&ID3D12CommandAllocator::IID).cast(),
                &mut allocator
            ),
            S_OK
        );
        let allocator = Guest(allocator);
        let mut list = null;
        assert_eq!(
            crate::list::create_list(
                null,
                0,
                0,
                allocator.0,
                null,
                std::ptr::from_ref(&ID3D12GraphicsCommandList::IID).cast(),
                &mut list
            ),
            S_OK
        );
        let list = Guest(list);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut destination = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                D3D12_RESOURCE_STATE_COPY_DEST.0 as u32,
                null.cast(),
                std::ptr::from_ref(&ID3D12Resource::IID).cast(),
                &mut destination
            ),
            S_OK
        );
        let destination = Guest(destination);
        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        }
        const _: () = assert!(size_of::<Location>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        let from = Location {
            resource: texture,
            kind: 0,
            footprint: Default::default(),
        };
        let to = Location {
            resource: destination.0,
            kind: 1,
            footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    Width: 16,
                    Height: 1,
                    Depth: 1,
                    RowPitch: 256,
                },
            },
        };
        let copy = slot::<
            extern "system" fn(This, &Location, u32, u32, u32, &Location, *const [u32; 6]),
        >(list.0, crate::list::test_slot(list.0, 16));
        copy(list.0, &to, 0, 0, 0, &from, null.cast());
        let close =
            slot::<extern "system" fn(This) -> i32>(list.0, crate::list::test_slot(list.0, 9));
        assert_eq!(close(list.0), S_OK);
        Self {
            list,
            destination,
            _allocator: allocator,
        }
    }

    fn read(&self, host: &crate::host::Host) -> [[f32; 4]; 2] {
        let segments = crate::list::segments(self.list.0);
        let [crate::list::Segment::List(list)] = segments.as_slice() else {
            panic!("expected a single copy list");
        };
        host.execute(&[list]);
        let (data, len) = crate::resource::buffer_bytes(self.destination.0).unwrap();
        assert!(len >= 32);
        // SAFETY: the completed guest readback filled this buffer with two RGBA32_FLOAT texels.
        unsafe { data.cast::<[[f32; 4]; 2]>().read_unaligned() }
    }
}

// The placement stays live for the process. A nonzero initial texel distinguishes
// preserved title data from a freshly allocated host resource.
const TEXTURE_ALIGNMENT: usize = 64 << 10;

struct Fixture {
    pipeline: Guest,
    views: Guest,
    handle: u64,
    address: u64,
    placed: This,
}

impl Fixture {
    fn new(host: &crate::host::Host, width: u32, height: u32) -> Option<Self> {
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        // SAFETY: an output value of the size required for this feature query.
        let shader_model = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
        };
        if shader_model.is_err() || sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
            eprintln!("placed UAV dispatch needs SM 6.6");
            return None;
        }

        let mut pipeline = std::ptr::null();
        assert_eq!(
            hand_out(vec![store_shader()], None, &PIPELINE_IIDS[0], &mut pipeline),
            S_OK
        );
        let pipeline = Guest(pipeline);
        assert!(
            host_pipeline(pipeline.0).is_some(),
            "synthetic host pipeline"
        );

        #[repr(C)]
        struct HeapDescWithPlacement(D3D12_DESCRIPTOR_HEAP_DESC, [u64; 2]);
        let desc = HeapDescWithPlacement(
            D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            [0; 2],
        );
        let iid = ID3D12DescriptorHeap::IID;
        let iid = guid(iid.data1, iid.data2, iid.data3, iid.data4);
        let mut views = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                std::ptr::from_ref(&desc).cast(),
                &iid,
                &mut views
            ),
            S_OK
        );
        let views = Guest(views);
        let handle = crate::descriptor::heap(views.0).start + 32;

        let count = (width * height * 4) as usize + TEXTURE_ALIGNMENT / 2;
        let backing = Box::leak(vec![3.0f32; count].into_boxed_slice());
        let address = (backing.as_mut_ptr() as usize).next_multiple_of(TEXTURE_ALIGNMENT) as u64;
        let format = DXGI_FORMAT_R32G32B32A32_FLOAT.0 as u32;
        let mut resource = [0; 14];
        resource[0] = D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32;
        resource[4] = width;
        resource[6] = height;
        resource[7] = 1 | (1 << 16); // one slice and one mip
        resource[8] = format;
        resource[9] = 1; // sample count
        resource[11] = 1; // linear layout
        resource[12] = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32;
        let mut view = [0; 16];
        view[0] = format;
        view[1] = D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32;
        view[10] = address as u32;
        view[11] = (address >> 32) as u32;
        let placed = crate::resource::texture_at(address, &resource).expect("placed texture");
        crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &view, handle);

        Some(Self {
            pipeline,
            views,
            handle,
            address,
            placed,
        })
    }

    fn dispatch(&self, host: &crate::host::Host, expected: [f32; 4]) -> crate::host::HostList {
        let list = host.new_list().unwrap();
        let mut state = crate::draw::State::default();
        state.set_pipeline(self.pipeline.0);
        state.set_heaps(&[self.views.0]);
        state.set_constants(true, 0, &expected.map(f32::to_bits), 0);
        state.set_table(true, 1, self.handle);
        let dispatched = crate::draw::dispatched();
        state.dispatch(&list, 1, 1, 1);
        assert!(
            crate::draw::dispatched() > dispatched,
            "dispatch did not record"
        );
        list.close();
        list
    }
}

#[test]
fn placed_uav_dispatch_writes_its_texture() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let Some(fixture) = Fixture::new(host, 16, 1) else {
        return;
    };
    let expected = [1.0f32, 0.5, -0.25, 2.0];
    let list = fixture.dispatch(host, expected);
    let consumer = TexelReadback::new(fixture.placed);
    host.execute(&[&list]);
    assert_eq!(consumer.read(host), [expected, [3.0; 4]]);
    let other = [-0.5f32, 3.0, 4.0, 5.0];
    // SAFETY: the adjacent texel is inside the aligned guest backing and the dispatch has completed.
    unsafe { (fixture.address as *mut [f32; 4]).add(1).write(other) };
    assert_eq!(consumer.read(host), [expected, other]);
    // A direct CPU read after the GPU fence observes the stored texel. Reset one
    // component to its original value; the remaining GPU-written channels survive.
    let first = fixture.address as *mut [f32; 4];
    // SAFETY: the mapped, aligned texel is live and both submissions completed.
    unsafe {
        assert_eq!(first.read(), expected, "fenced guest backing");
        first.cast::<f32>().write(3.0);
    }
    assert_eq!(consumer.read(host), [[3.0, 0.5, -0.25, 2.0], other]);
    // SAFETY: the host device is live after both completed submissions.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    eprintln!("placed UAV readback, fenced backing and CPU channel reset verified");
}

#[test]
fn placed_uav_readbacks_stay_distinct_across_submissions() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let Some(a) = Fixture::new(host, 16, 1) else {
        return;
    };
    let b = Fixture::new(host, 16, 1).unwrap();
    let wider = Fixture::new(host, 128, 1).unwrap();
    for (step, second) in [&b, &b, &wider].into_iter().enumerate() {
        let left = [step as f32 + 1.0, 4.0, 5.0, 6.0];
        let right = [-left[0], 7.0, 8.0, 9.0];
        let first_list = a.dispatch(host, left);
        let second_list = second.dispatch(host, right);
        host.execute(&[&first_list, &second_list]);
        // SAFETY: both fixtures own distinct aligned backing, and execute waited for the fence.
        unsafe {
            assert_eq!((a.address as *const [f32; 4]).read(), left);
            assert_eq!((second.address as *const [f32; 4]).read(), right);
        }
    }
    eprintln!("distinct placed UAV writebacks verified across footprint changes");
}

#[test]
#[ignore = "times fenced submissions on the selected adapter"]
fn time_single_texel_writes_to_a_16_mib_placed_uav() {
    let host = crate::host::host().expect("host D3D12 device");
    let fixture = Fixture::new(host, 1024, 1024).expect("SM 6.6 adapter");
    let mut times = [0; 10];
    for (i, sample) in times.iter_mut().enumerate() {
        let list = fixture.dispatch(host, [i as f32, 0.5, -0.25, 2.0]);
        let start = std::time::Instant::now();
        host.execute(&[&list]);
        *sample = start.elapsed().as_micros();
    }
    eprintln!("placed UAV submission times in microseconds (first is cold): {times:?}");
}
