//! Synthetic image stores of some channels (`dmask` not 0xF) to an RGBA32F
//! texture UAV. The channels `dmask` leaves out are written as zero, as
//! RDNA2 MIMG stores do. And integer stores to UINT texture UAVs, which
//! take the integer typed store by the class the host lists for the view.

use super::{HostShader, emit};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, io, ir, resource, structure};

const LANES: usize = 32;

/// `v_mov_b32 vdst, src0`.
const fn v_mov_b32(vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | 0x01 << 9 | src0
}

/// `image_store` of `dmask`'s channels, 2D: the data from `vdata`, the
/// coordinates from v2 and v3, the descriptor s[12:19].
const fn image_store(dmask: u32, vdata: u32) -> [u32; 2] {
    [
        0xF000_0000 | 0x08 << 18 | dmask << 8 | 2 << 2,
        0x0003_0002 | vdata << 8,
    ]
}

/// Lane `i` stores to texel (i, 0): all four channels (1, 2, 3, 4), then
/// y alone (9), then x and z (7, 8), which are packed from one VGPR pair.
fn words() -> Vec<u32> {
    let mut w = vec![
        0xBE81_0380, // s_mov_b32 s1, 0 (high half of the table address)
        0xF40C_0300,
        0xFA00_0000,         // s_load_dwordx8 s[12:19], s[0:1], 0
        v_mov_b32(2, 0x100), // v2 = lane
        v_mov_b32(3, 0x80),  // v3 = 0
        v_mov_b32(4, 0xF2),  // v4 = 1.0
        v_mov_b32(5, 0xF4),  // v5 = 2.0
        v_mov_b32(6, 0xFF),
        3.0f32.to_bits(),   // v6 = 3.0
        v_mov_b32(7, 0xF6), // v7 = 4.0
        v_mov_b32(8, 0xFF),
        9.0f32.to_bits(), // v8 = 9.0
        v_mov_b32(10, 0xFF),
        7.0f32.to_bits(), // v10 = 7.0
        v_mov_b32(11, 0xFF),
        8.0f32.to_bits(), // v11 = 8.0
        0xBF8C_C07F,      // s_waitcnt lgkmcnt(0)
    ];
    w.extend(image_store(0xF, 4));
    w.extend(image_store(0x2, 8));
    w.extend(image_store(0x5, 10));
    w.push(0xBF81_0000); // s_endpgm
    w
}

fn shader(code: &[u32]) -> Result<HostShader, crate::ShaderError> {
    shader_of(code, 1)
}

/// `code` as a compute program whose table (s[0:1]) holds `views` image
/// UAV descriptors, eight dwords apart.
fn shader_of(code: &[u32], views: u32) -> Result<HostShader, crate::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: (0..views)
            .map(|k| UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8 * k,
                lower_bound: k,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            })
            .collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1, // one user SGPR
            threads: [LANES as u32, 1, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    let bytes = crate::dxbc::build(&[]);
    let container = Container::parse(&bytes).unwrap();
    emit(&container, &native, &p, &st, &io, &res)
}

fn validate(shader: &HostShader) {
    if let Ok(mut validator) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        validator
            .validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
}

#[test]
fn image_stores_of_some_channels_translate() {
    validate(&shader(&words()).expect("translate"));
}

/// A channel mask no run of leading channels covers (y alone) is stored too.
#[test]
fn a_store_of_one_inner_channel_translates() {
    let mut w = words();
    w.truncate(w.len() - 1);
    w.extend(image_store(0x4, 8));
    w.push(0xBF81_0000);
    validate(&shader(&w).expect("translate"));
}

#[test]
fn image_stores_of_some_channels_zero_the_others() {
    use super::host::{Dispatch, Host, Root, View};
    let shader = shader(&words()).expect("translate");
    validate(&shader);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let mut ran = 0;
    for &(name, warp) in super::valu_tests::backends(force_warp) {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let output = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]),
                    )],
                    views: &[View::TextureUav(LANES as u32, 1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        // (1, 2, 3, 4), then y = 9 alone, then x = 7 and z = 8 alone: the
        // last store's channels, the others zero.
        let want: Vec<u32> = (0..LANES)
            .flat_map(|_| [7.0f32, 0.0, 8.0, 0.0].map(f32::to_bits))
            .collect();
        assert_eq!(output[0], want, "{name}");
        eprintln!("{name}: verified image stores of some channels");
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

/// `image_store` of `dmask`'s channels, 2D: the data from `vdata`, the
/// coordinates from v2 and v3, the descriptor at `s[rsrc:rsrc + 7]`.
const fn image_store_from(dmask: u32, vdata: u32, rsrc: u32) -> [u32; 2] {
    [
        0xF000_0000 | 0x08 << 18 | dmask << 8 | 2 << 2,
        0x0000_0002 | (rsrc / 4) << 16 | vdata << 8,
    ]
}

/// Lane `i` stores integers to texel (i, 0) of two integer views: `i + 5`
/// through the first descriptor (s[12:19]), `i + 1` through the second
/// (s[20:27]). Both are denormals as floats, which a float store into an
/// integer view would turn into 0.
fn integer_words() -> Vec<u32> {
    let mut w = vec![
        0xBE81_0380, // s_mov_b32 s1, 0 (high half of the table address)
        0xF40C_0300,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[0:1], 0
        0xF40C_0500,
        0xFA00_0020,         // s_load_dwordx8 s[20:27], s[0:1], 32
        v_mov_b32(2, 0x100), // v2 = lane
        v_mov_b32(3, 0x80),  // v3 = 0
        0x4A08_0085,         // v_add_nc_u32 v4, 5, v0
        0x4A0A_0081,         // v_add_nc_u32 v5, 1, v0
        0xBF8C_C07F,         // s_waitcnt lgkmcnt(0)
    ];
    w.extend(image_store_from(0x1, 4, 12));
    w.extend(image_store_from(0x1, 5, 20));
    w.push(0xBF81_0000); // s_endpgm
    w
}

/// The integer stores read their views' classes from the host's list and
/// take the integer typed store for an integer view, which validates.
#[test]
fn integer_image_stores_translate() {
    let shader = shader_of(&integer_words(), 2).expect("translate");
    assert!(
        shader
            .user_data
            .iter()
            .any(|u| u.root == RootType::Other(super::USER_VIEW_CLASSES)),
        "{:?}",
        shader.user_data
    );
    validate(&shader);
}

/// On a device: integers stored into an R32_UINT and an R8_UINT texture UAV
/// read back as those integers, with the host's view class list saying both
/// views are UINT. (Before stores took the view's class, each was a float
/// store of the integer's bits, a denormal, which may be stored as 0.)
#[test]
fn integer_image_stores_reach_uint_views_on_hardware() {
    use super::host::{Dispatch, Host, Root, Stride, View};
    let shader = shader_of(&integer_words(), 2).expect("translate");
    validate(&shader);
    // Heap: the two UAVs at 0 and 1, the view class list at 2 (one dword
    // per slot).
    let classes = [super::VIEW_CLASS_UINT, super::VIEW_CLASS_UINT, 0];
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            RootType::Other(super::USER_VIEW_CLASSES) => 2,
            _ => 0,
        };
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let mut ran = 0;
    for &(name, warp) in super::valu_tests::backends(force_warp) {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let output = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud.clone()))],
                    views: &[
                        View::UintTextureUav(LANES as u32, 1),
                        View::ByteTextureUav(LANES as u32, 1),
                        View::Buffer {
                            data: &classes,
                            stride: Stride::Raw,
                        },
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        let want32: Vec<u32> = (0..LANES as u32).map(|i| i + 5).collect();
        let want8: Vec<u32> = (0..LANES as u32).map(|i| i + 1).collect();
        assert_eq!(output[0], want32, "{name}: R32_UINT");
        assert_eq!(output[1], want8, "{name}: R8_UINT");
        eprintln!("{name}: verified integer image stores into UINT views");
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}
