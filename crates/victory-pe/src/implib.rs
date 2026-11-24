//! COFF import libraries (`.lib`).
//!
//! An umbrella import library tells us which DLL provides each API that a
//! console title may link against. That is the map the runtime needs to
//! decide which of our implementations answers which import.

use object::Object;
use object::ObjectSymbol;
use object::read::archive::ArchiveFile;
use object::read::coff::{CoffFile, ImportFile, ImportName, ImportType};

use crate::{ImportSymbol, PeError};

/// One `dll!export` pair described by an import library.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LibImport {
    pub dll: String,
    pub export: ImportSymbol,
    pub kind: LibImportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LibImportKind {
    Code,
    Data,
    Const,
}

/// Result of reading an import library.
#[derive(Debug, Default)]
pub struct ImportLibrary {
    pub imports: Vec<LibImport>,
    /// Members that are not short-format imports (regular objects, long
    /// import descriptors and their null thunks). Reported so nothing is
    /// silently lost.
    pub other_members: usize,
}

// Values from the COFF import-object header. They are kept as plain integers
// here because the `object` crate exposes them as newtypes whose constants
// cannot be compared with bytes read from an archive.
const SIG1: u16 = 0; // IMAGE_FILE_MACHINE_UNKNOWN
const SIG2: u16 = 0xffff; // IMPORT_OBJECT_HDR_SIG2
#[cfg(test)]
const TYPE_CODE: u16 = 0;
const TYPE_DATA: u16 = 1;
const TYPE_CONST: u16 = 2;
const NAME_ORDINAL: u16 = 0;
const NAME: u16 = 1;
const NAME_NO_PREFIX: u16 = 2;
const NAME_UNDECORATE: u16 = 3;
const NAME_EXPORTAS: u16 = 4;

/// A long import descriptor has the same 20-byte prefix as a short import
/// object but carries no inline string data. The names live in the COFF
/// object that follows the null descriptor and null thunk members.
fn import_header(bytes: &[u8]) -> Option<(u16, u32, u16, u16)> {
    if bytes.len() < 20 {
        return None;
    }
    let sig1 = u16::from_le_bytes(bytes[0..2].try_into().ok()?);
    let sig2 = u16::from_le_bytes(bytes[2..4].try_into().ok()?);
    if sig1 != SIG1 || sig2 != SIG2 {
        return None;
    }
    Some((
        u16::from_le_bytes(bytes[8..10].try_into().ok()?), // machine
        u32::from_le_bytes(bytes[12..16].try_into().ok()?), // size_of_data
        u16::from_le_bytes(bytes[16..18].try_into().ok()?), // ordinal/hint
        u16::from_le_bytes(bytes[18..20].try_into().ok()?), // type/name type
    ))
}

fn import_kind(flags: u16) -> LibImportKind {
    match flags & 0b11 {
        TYPE_DATA => LibImportKind::Data,
        TYPE_CONST => LibImportKind::Const,
        _ => LibImportKind::Code,
    }
}

/// Names defined by the COFF object that belongs to a long descriptor.
fn coff_symbols(bytes: &[u8]) -> Vec<String> {
    let Ok(file) = CoffFile::<_, object::pe::ImageFileHeader>::parse(bytes) else {
        return Vec::new();
    };
    file.symbols()
        .filter_map(|symbol| symbol.name().ok().map(str::to_owned))
        .collect()
}

fn strip_decoration(name: &str, name_type: u16) -> String {
    // `IMPORT_OBJECT_NAME` is passed through as-is; only the no-prefix and
    // undecorate forms ask us to remove a leading decoration.
    if name_type == NAME {
        return name.to_owned();
    }
    let stripped = if matches!(name_type, NAME_NO_PREFIX | NAME_UNDECORATE) {
        name.strip_prefix(['?', '@', '_']).unwrap_or(name)
    } else {
        name
    };
    if name_type == NAME_UNDECORATE {
        stripped.split('@').next().unwrap_or(stripped).to_owned()
    } else {
        stripped.to_owned()
    }
}

impl ImportLibrary {
    pub fn parse(data: &[u8]) -> Result<Self, PeError> {
        let archive = ArchiveFile::parse(data)?;
        let mut lib = ImportLibrary::default();
        let members: Vec<_> = archive.members().collect::<Result<Vec<_>, _>>()?;
        let contents: Vec<Vec<u8>> = members
            .iter()
            .map(|member| member.data(data).map(<[u8]>::to_vec))
            .collect::<Result<_, _>>()?;
        let mut consumed = vec![false; contents.len()];

        for (index, member) in contents.iter().enumerate() {
            if consumed[index] {
                continue;
            }
            if let Some((_, size, ordinal, flags)) = import_header(member) {
                if size == 0 {
                    // Long-format descriptor. Its DLL and export names live
                    // in the COFF object(s) that follow the null members.
                    consumed[index] = true;
                    lib.other_members += 1;
                    let kind = import_kind(flags);
                    let name_type = flags >> 2 & 0b111;
                    let mut dll = None;
                    let mut names = Vec::new();
                    for (next, object) in contents.iter().enumerate().skip(index + 1) {
                        if consumed[next] || import_header(object).is_some() {
                            break;
                        }
                        let symbols = coff_symbols(object);
                        if symbols.is_empty() {
                            continue;
                        }
                        consumed[next] = true;
                        lib.other_members += 1;
                        for symbol in symbols {
                            if let Some(d) = symbol.strip_prefix("__IMPORT_DESCRIPTOR_") {
                                dll = Some(d.to_owned());
                            } else if let Some(export) = symbol.strip_prefix("__imp_") {
                                names.push(export.to_owned());
                            }
                        }
                    }
                    let Some(dll) = dll else {
                        continue;
                    };
                    for name in names {
                        let export = if name_type == NAME_ORDINAL {
                            ImportSymbol::Ordinal(ordinal)
                        } else if name_type == NAME_EXPORTAS {
                            ImportSymbol::Name { hint: 0, name }
                        } else {
                            ImportSymbol::Name {
                                hint: 0,
                                name: strip_decoration(&name, name_type),
                            }
                        };
                        lib.imports.push(LibImport {
                            dll: dll.clone(),
                            export,
                            kind,
                        });
                    }
                    continue;
                }
            }
            match ImportFile::parse(&member[..]) {
                Ok(import) => {
                    lib.imports.push(LibImport {
                        dll: String::from_utf8_lossy(import.dll()).into_owned(),
                        export: match import.import() {
                            ImportName::Name(name) => ImportSymbol::Name {
                                hint: 0,
                                name: String::from_utf8_lossy(name).into_owned(),
                            },
                            ImportName::Ordinal(o) => ImportSymbol::Ordinal(o),
                        },
                        kind: match import.import_type() {
                            ImportType::Code => LibImportKind::Code,
                            ImportType::Data => LibImportKind::Data,
                            ImportType::Const => LibImportKind::Const,
                        },
                    });
                }
                Err(_) => lib.other_members += 1,
            }
        }
        lib.imports.sort();
        lib.imports.dedup();
        Ok(lib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short-format import object: `IMPORT_OBJECT_HEADER`, then the
    /// symbol and the DLL name, each NUL-terminated.
    fn short_import(
        dll: &str,
        symbol: &str,
        ordinal_or_hint: u16,
        ty: u16,
        name_ty: u16,
    ) -> Vec<u8> {
        let data_len = symbol.len() + 1 + dll.len() + 1;
        let mut out = Vec::new();
        out.extend_from_slice(&0u16.to_le_bytes()); // Sig1
        out.extend_from_slice(&0xFFFFu16.to_le_bytes()); // Sig2
        out.extend_from_slice(&0u16.to_le_bytes()); // Version
        out.extend_from_slice(&crate::MACHINE_AMD64.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        out.extend_from_slice(&ordinal_or_hint.to_le_bytes());
        out.extend_from_slice(&(ty | name_ty << 2).to_le_bytes());
        for s in [symbol, dll] {
            out.extend_from_slice(s.as_bytes());
            out.push(0);
        }
        out
    }

    /// A `!<arch>` archive of `members`, each under a short GNU-style name.
    fn archive(members: &[Vec<u8>]) -> Vec<u8> {
        let names: Vec<String> = (0..members.len()).map(|i| format!("m{i}.obj/")).collect();
        let named: Vec<(&str, &[u8])> = names
            .iter()
            .zip(members)
            .map(|(name, data)| (name.as_str(), data.as_slice()))
            .collect();
        archive_named(&named)
    }

    /// A `!<arch>` archive whose members carry the raw header names given.
    fn archive_named(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = b"!<arch>\n".to_vec();
        for (name, data) in members {
            let header = format!(
                "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
                name,
                0,
                0,
                0,
                644,
                data.len()
            );
            assert_eq!(header.len(), 60);
            out.extend_from_slice(header.as_bytes());
            out.extend_from_slice(data);
            if data.len() % 2 == 1 {
                out.push(b'\n');
            }
        }
        out
    }

    const CODE: u16 = 0;
    const DATA: u16 = 1;
    const CONST: u16 = 2;
    const BY_ORDINAL: u16 = 0;
    const BY_NAME: u16 = 1;
    const NO_PREFIX: u16 = 2;
    const UNDECORATE: u16 = 3;

    fn name(dll: &str, name: &str, kind: LibImportKind) -> LibImport {
        LibImport {
            dll: dll.into(),
            export: ImportSymbol::Name {
                hint: 0,
                name: name.into(),
            },
            kind,
        }
    }

    #[test]
    fn reads_short_imports_sorted_and_deduplicated() {
        let lib = ImportLibrary::parse(&archive(&[
            short_import("b.dll", "_Stdcall@8", 7, CODE, UNDECORATE),
            short_import("a.dll", "Plain", 3, CODE, BY_NAME),
            short_import("a.dll", "_Prefixed", 0, DATA, NO_PREFIX),
            short_import("a.dll", "Ignored", 42, CONST, BY_ORDINAL),
            // The same export again, under another hint: one entry.
            short_import("a.dll", "Plain", 9, CODE, BY_NAME),
            // Not a short import object: counted, not dropped silently.
            b"not an import".to_vec(),
        ]))
        .unwrap();

        assert_eq!(
            lib.imports,
            [
                name("a.dll", "Plain", LibImportKind::Code),
                name("a.dll", "Prefixed", LibImportKind::Data),
                LibImport {
                    dll: "a.dll".into(),
                    export: ImportSymbol::Ordinal(42),
                    kind: LibImportKind::Const,
                },
                name("b.dll", "Stdcall", LibImportKind::Code),
            ]
        );
        assert_eq!(lib.other_members, 1);
    }

    #[test]
    fn rejects_what_is_not_an_archive() {
        assert!(ImportLibrary::parse(b"MZ not an archive").is_err());
        assert!(
            ImportLibrary::parse(&archive(&[]))
                .unwrap()
                .imports
                .is_empty()
        );
    }

    /// The layout `lib.exe` writes: a first and a second linker member
    /// (both named `/`), a `//` long-names member, then the import objects
    /// under `/<offset>` names. The bookkeeping members are not imports and
    /// not "other" members either.
    #[test]
    fn reads_msvc_layout_with_linker_and_long_name_members() {
        // First linker member: big-endian symbol count, no symbols.
        let first_linker = 0u32.to_be_bytes();
        // Second linker member: member count and symbol count, both 0.
        let second_linker = [0u8; 8];
        let long_names =
            b"a_member_name_too_long_for_the_header.obj\0another_long_member_name.obj\0";
        let first = short_import(
            "xboxkrnl.exe",
            "KeQueryPerformanceCounter",
            1,
            CODE,
            BY_NAME,
        );
        let second = short_import("kernelx.dll", "GetTickCount64", 2, CODE, BY_NAME);

        let lib = ImportLibrary::parse(&archive_named(&[
            ("/", &first_linker),
            ("/", &second_linker),
            ("//", long_names),
            ("/0", &first),
            ("/42", &second),
        ]))
        .unwrap();

        assert_eq!(
            lib.imports,
            [
                name("kernelx.dll", "GetTickCount64", LibImportKind::Code),
                name(
                    "xboxkrnl.exe",
                    "KeQueryPerformanceCounter",
                    LibImportKind::Code
                ),
            ]
        );
        assert_eq!(lib.other_members, 0);
    }

    /// A damaged archive is an error, never a panic; a damaged member inside
    /// an intact archive is counted, not read as an import.
    #[test]
    fn damaged_archives_and_members_do_not_panic() {
        let good = short_import("a.dll", "Plain", 0, CODE, BY_NAME);
        let whole = archive(std::slice::from_ref(&good));

        // Every truncation of the archive either parses or errors.
        for len in 0..whole.len() {
            let _ = ImportLibrary::parse(&whole[..len]);
        }
        // A member header that claims more bytes than the file holds.
        let mut oversized = whole.clone();
        let size_field = 8 + 48;
        oversized[size_field..size_field + 10].copy_from_slice(b"999999    ");
        assert!(ImportLibrary::parse(&oversized).is_err());
        // A long name that points past the names table.
        assert!(
            ImportLibrary::parse(&archive_named(&[("//", b"x.obj\0"), ("/99", &good)])).is_err()
        );

        // An import object whose SizeOfData runs past the member, and one cut
        // short inside its fixed header: neither is an import.
        let mut overlong = good.clone();
        overlong[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        let lib =
            ImportLibrary::parse(&archive(&[overlong, good[..10].to_vec(), good.clone()])).unwrap();
        assert_eq!(lib.imports, [name("a.dll", "Plain", LibImportKind::Code)]);
        assert_eq!(lib.other_members, 2);
    }

    /// One export name in two DLLs, or once as code and once as data, is
    /// two different imports: deduplication only drops exact repeats.
    #[test]
    fn keeps_the_same_name_from_other_dlls_and_of_other_kinds() {
        let lib = ImportLibrary::parse(&archive(&[
            short_import("b.dll", "Shared", 0, CODE, BY_NAME),
            short_import("a.dll", "Shared", 0, DATA, BY_NAME),
            short_import("a.dll", "Shared", 0, CODE, BY_NAME),
            short_import("a.dll", "Shared", 0, CODE, BY_NAME),
        ]))
        .unwrap();
        assert_eq!(
            lib.imports,
            [
                name("a.dll", "Shared", LibImportKind::Code),
                name("a.dll", "Shared", LibImportKind::Data),
                name("b.dll", "Shared", LibImportKind::Code),
            ]
        );
        assert_eq!(lib.other_members, 0);
    }

    /// Names that are not UTF-8 are read lossily rather than rejected.
    #[test]
    fn reads_names_that_are_not_utf8() {
        let mut member = short_import("x.dll", "Ab", 0, CODE, BY_NAME);
        // After the 20-byte header: `Ab\0x.dll\0`. Damage one byte of the
        // symbol and one of the DLL name.
        member[20 + 1] = 0xFF;
        member[20 + 3] = 0xFE;
        let lib = ImportLibrary::parse(&archive(&[member])).unwrap();
        assert_eq!(
            lib.imports,
            [name("\u{FFFD}.dll", "A\u{FFFD}", LibImportKind::Code)]
        );
    }

    /// A member that looks like an import object but lacks its `0xFFFF`
    /// signature is a regular member, not an import.
    #[test]
    fn a_member_with_a_bad_signature_is_another_member() {
        let mut member = short_import("a.dll", "Plain", 0, CODE, BY_NAME);
        member[2] = 0xFE; // Sig2 is no longer 0xFFFF
        let lib = ImportLibrary::parse(&archive(&[member])).unwrap();
        assert!(lib.imports.is_empty());
        assert_eq!(lib.other_members, 1);
    }

    const EXPORT_AS: u16 = 4;

    /// A short-format import object with the extra export name that
    /// `IMPORT_OBJECT_NAME_EXPORTAS` places after the DLL name.
    fn export_as_import(dll: &str, symbol: &str, export: &str) -> Vec<u8> {
        let mut out = short_import(dll, symbol, 0, CODE, EXPORT_AS);
        out.extend_from_slice(export.as_bytes());
        out.push(0);
        let data_len = (out.len() - 20) as u32;
        out[12..16].copy_from_slice(&data_len.to_le_bytes());
        out
    }

    /// `EXPORTAS` names the export with its third string, not the symbol;
    /// one that lacks that string is not an import.
    #[test]
    fn reads_the_export_name_of_an_export_as_import() {
        let lib = ImportLibrary::parse(&archive(&[
            export_as_import("a.dll", "__imp_Decorated@4", "Exported"),
            // Same header, but SizeOfData stops after the DLL name.
            short_import("a.dll", "Missing", 0, CODE, EXPORT_AS),
        ]))
        .unwrap();
        assert_eq!(
            lib.imports,
            [name("a.dll", "Exported", LibImportKind::Code)]
        );
        assert_eq!(lib.other_members, 1);
    }

    /// Import type 3 and name types 5 to 7 are not defined. Such members
    /// are counted, and the members around them are still read.
    #[test]
    fn undefined_import_and_name_types_are_other_members() {
        let lib = ImportLibrary::parse(&archive(&[
            short_import("a.dll", "BadType", 0, 3, BY_NAME),
            short_import("a.dll", "Name5", 0, CODE, 5),
            short_import("a.dll", "Kept", 0, CODE, BY_NAME),
            short_import("a.dll", "Name6", 0, DATA, 6),
            short_import("a.dll", "Name7", 0, CONST, 7),
        ]))
        .unwrap();
        assert_eq!(lib.imports, [name("a.dll", "Kept", LibImportKind::Code)]);
        assert_eq!(lib.other_members, 4);
    }

    /// `NO_PREFIX` drops one leading `?`, `@` or `_`; `UNDECORATE` also
    /// drops everything from the first `@` after that. Names without a
    /// prefix are kept as written.
    #[test]
    fn strips_one_prefix_and_the_stdcall_suffix() {
        let cases = [
            ("?Question", NO_PREFIX, "Question"),
            ("@At", NO_PREFIX, "At"),
            ("__Twice", NO_PREFIX, "_Twice"),
            ("Bare@4", NO_PREFIX, "Bare@4"),
            ("_std@12", UNDECORATE, "std"),
            ("@fast@8", UNDECORATE, "fast"),
            ("?cpp@@YAXXZ", UNDECORATE, "cpp"),
            ("plain", UNDECORATE, "plain"),
        ];
        for (symbol, name_ty, expected) in cases {
            let lib =
                ImportLibrary::parse(&archive(&[short_import("a.dll", symbol, 0, CODE, name_ty)]))
                    .unwrap();
            assert_eq!(
                lib.imports,
                [name("a.dll", expected, LibImportKind::Code)],
                "{symbol} with name type {name_ty}"
            );
        }
    }

    #[test]
    fn ordinal_imports_keep_their_dll_and_kind() {
        // By-ordinal objects still carry a symbol name; it is not the export,
        // so two objects with one ordinal and different names are one entry.
        let lib = ImportLibrary::parse(&archive(&[
            short_import("b.dll", "x", 1, DATA, BY_ORDINAL),
            short_import("a.dll", "y", 1, CODE, BY_ORDINAL),
            short_import("a.dll", "z", 1, CODE, BY_ORDINAL),
            short_import("a.dll", "w", 2, CODE, BY_ORDINAL),
            short_import("a.dll", "Named", 1, CODE, BY_NAME),
        ]))
        .unwrap();
        let ordinal = |dll: &str, o: u16, kind| LibImport {
            dll: dll.into(),
            export: ImportSymbol::Ordinal(o),
            kind,
        };
        assert_eq!(
            lib.imports,
            [
                name("a.dll", "Named", LibImportKind::Code),
                ordinal("a.dll", 1, LibImportKind::Code),
                ordinal("a.dll", 2, LibImportKind::Code),
                ordinal("b.dll", 1, LibImportKind::Data),
            ]
        );
        assert_eq!(lib.other_members, 0);
    }

    /// BSD archives keep a long member name in front of the member data
    /// (`#1/<len>`); the import object starts after that name.
    #[test]
    fn reads_members_with_bsd_long_names() {
        let member_name = b"a_long_bsd_member_name.obj\0\0";
        let mut member = member_name.to_vec();
        member.extend_from_slice(&short_import("a.dll", "Bsd", 0, CODE, BY_NAME));
        let header = format!("#1/{}", member_name.len());
        let lib = ImportLibrary::parse(&archive_named(&[(&header, &member)])).unwrap();
        assert_eq!(lib.imports, [name("a.dll", "Bsd", LibImportKind::Code)]);
        assert_eq!(lib.other_members, 0);

        // A name length past the member is an error, not a panic.
        let header = format!("#1/{}", member.len() + 1);
        assert!(ImportLibrary::parse(&archive_named(&[(&header, &member)])).is_err());
    }

    /// The members of a thin archive live in other files, so the archive
    /// holds no import objects; each member is counted, none is read.
    #[test]
    fn thin_archive_members_are_counted_not_read() {
        // A thin archive's member headers are not followed by their data;
        // each header carries the size of the outside file.
        let mut out = b"!<thin>\n".to_vec();
        for i in 0..2 {
            let header = format!(
                "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
                format!("m{i}.obj/"),
                0,
                0,
                0,
                644,
                100
            );
            out.extend_from_slice(header.as_bytes());
        }
        let lib = ImportLibrary::parse(&out).unwrap();
        assert!(lib.imports.is_empty());
        assert_eq!(lib.other_members, 2);
    }

    /// A long-format import member is an ordinary COFF object: it starts
    /// with its machine, not the `0, 0xFFFF` signature, so it is counted.
    #[test]
    fn a_coff_object_member_is_another_member() {
        let mut coff = Vec::new();
        coff.extend_from_slice(&crate::MACHINE_AMD64.to_le_bytes());
        coff.extend_from_slice(&0u16.to_le_bytes()); // NumberOfSections
        coff.extend_from_slice(&[0u8; 16]); // the rest of IMAGE_FILE_HEADER
        let lib = ImportLibrary::parse(&archive(&[
            coff,
            short_import("a.dll", "Kept", 0, CODE, BY_NAME),
        ]))
        .unwrap();
        assert_eq!(lib.imports, [name("a.dll", "Kept", LibImportKind::Code)]);
        assert_eq!(lib.other_members, 1);
    }

    /// GNU `ar` writes a `/SYM64/` symbol table for large archives. Like
    /// the `/` linker members it is bookkeeping: neither an import nor an
    /// other member.
    #[test]
    fn the_gnu_64_bit_symbol_table_is_skipped() {
        let sym64 = 0u64.to_be_bytes();
        let import = short_import("a.dll", "After", 0, CODE, BY_NAME);
        let lib = ImportLibrary::parse(&archive_named(&[("/SYM64/", &sym64), ("a.obj/", &import)]))
            .unwrap();
        assert_eq!(lib.imports, [name("a.dll", "After", LibImportKind::Code)]);
        assert_eq!(lib.other_members, 0);
    }

    /// The symbol and the DLL name must both end inside `SizeOfData`. An
    /// object whose DLL name is cut before its NUL is not an import.
    #[test]
    fn a_dll_name_without_its_terminator_is_not_an_import() {
        let mut member = short_import("a.dll", "Cut", 0, CODE, BY_NAME);
        member.pop(); // drop the DLL name's NUL
        let data_len = (member.len() - 20) as u32;
        member[12..16].copy_from_slice(&data_len.to_le_bytes());
        let lib = ImportLibrary::parse(&archive(&[member])).unwrap();
        assert!(lib.imports.is_empty());
        assert_eq!(lib.other_members, 1);
    }

    /// `ar` pads a member of odd length with one `\n` so the next header
    /// starts on an even offset. The member after the pad is read too.
    #[test]
    fn a_member_of_odd_length_is_followed_by_its_padding() {
        let odd = short_import("a.dll", "Od", 0, CODE, BY_NAME);
        assert_eq!(odd.len() % 2, 1, "the fixture needs an odd member");
        let whole = archive(&[odd, short_import("b.dll", "Even", 0, DATA, BY_NAME)]);
        let lib = ImportLibrary::parse(&whole).unwrap();
        assert_eq!(
            lib.imports,
            [
                name("a.dll", "Od", LibImportKind::Code),
                name("b.dll", "Even", LibImportKind::Data),
            ]
        );
        assert_eq!(lib.other_members, 0);
    }

    /// A member with no bytes at all is not an import object: it is
    /// counted, and the members after it are still read.
    #[test]
    fn an_empty_member_is_another_member() {
        let lib = ImportLibrary::parse(&archive(&[
            Vec::new(),
            short_import("a.dll", "After", 0, CODE, BY_NAME),
        ]))
        .unwrap();
        assert_eq!(lib.imports, [name("a.dll", "After", LibImportKind::Code)]);
        assert_eq!(lib.other_members, 1);
    }

    /// The machine field is not filtered: an umbrella library is read for
    /// its DLL map, and an import object for another machine names the same
    /// `dll!export` pair.
    #[test]
    fn an_import_object_for_another_machine_is_still_read() {
        let mut member = short_import("a.dll", "Arm", 0, CODE, BY_NAME);
        member[6..8].copy_from_slice(&0xAA64u16.to_le_bytes()); // ARM64
        let lib = ImportLibrary::parse(&archive(&[member])).unwrap();
        assert_eq!(lib.imports, [name("a.dll", "Arm", LibImportKind::Code)]);
        assert_eq!(lib.other_members, 0);
    }

    /// A symbol that is nothing but the prefix `NO_PREFIX` strips, or an
    /// empty symbol read by name, names the empty export rather than
    /// failing the library.
    #[test]
    fn a_symbol_that_is_only_a_prefix_names_the_empty_export() {
        let lib = ImportLibrary::parse(&archive(&[
            short_import("a.dll", "_", 0, CODE, NO_PREFIX),
            short_import("a.dll", "", 0, CODE, BY_NAME),
            short_import("a.dll", "_@4", 0, CODE, UNDECORATE),
        ]))
        .unwrap();
        assert_eq!(lib.imports, [name("a.dll", "", LibImportKind::Code)]);
        assert_eq!(lib.other_members, 0);
    }

    fn long_descriptor(ordinal: u16, flags: u16) -> Vec<u8> {
        let mut out = vec![0u8; 20];
        out[2..4].copy_from_slice(&SIG2.to_le_bytes());
        out[8..10].copy_from_slice(&0x8664u16.to_le_bytes());
        out[16..18].copy_from_slice(&ordinal.to_le_bytes());
        out[18..20].copy_from_slice(&flags.to_le_bytes());
        out
    }

    /// A minimal COFF object with undefined external symbols.
    fn coff_object(names: &[&str]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut offsets = Vec::new();
        for name in names {
            offsets.push(strings.len() as u32 + 4);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        let mut out = Vec::new();
        out.extend_from_slice(&0x8664u16.to_le_bytes()); // machine
        out.extend_from_slice(&0u16.to_le_bytes()); // sections
        out.extend_from_slice(&0u32.to_le_bytes()); // timestamp
        out.extend_from_slice(&20u32.to_le_bytes()); // symbol table offset
        out.extend_from_slice(&(names.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // optional header
        out.extend_from_slice(&0u16.to_le_bytes()); // characteristics
        for offset in offsets {
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes()); // value
            out.extend_from_slice(&0i16.to_le_bytes()); // undefined section
            out.extend_from_slice(&0u16.to_le_bytes()); // type
            out.push(2); // external
            out.push(0); // no aux records
        }
        out.extend_from_slice(&(strings.len() as u32 + 4).to_le_bytes());
        out.extend_from_slice(&strings);
        out
    }

    #[test]
    fn short_imports_are_read() {
        let bytes = archive(&[short_import(
            "xsystem.dll",
            "XSystemGetDeviceType",
            0,
            TYPE_CODE,
            NAME,
        )]);
        let lib = ImportLibrary::parse(&bytes).unwrap();
        assert_eq!(lib.imports.len(), 1);
        assert_eq!(lib.imports[0].dll, "xsystem.dll");
        assert_eq!(
            lib.imports[0].export,
            ImportSymbol::Name {
                hint: 0,
                name: "XSystemGetDeviceType".into()
            }
        );
        assert_eq!(lib.other_members, 0);
    }

    #[test]
    fn long_imports_take_their_names_from_the_coff_object() {
        let object = coff_object(&[
            "__IMPORT_DESCRIPTOR_xsystem.dll",
            "__imp_XSystemGetDeviceType",
            "__imp_XSystemSetDeviceType",
        ]);
        let bytes = archive(&[
            long_descriptor(0, TYPE_CODE | NAME << 2),
            vec![0u8; 20],
            vec![0u8; 8],
            object,
        ]);
        let lib = ImportLibrary::parse(&bytes).unwrap();
        let exports: Vec<_> = lib
            .imports
            .iter()
            .filter(|i| i.dll == "xsystem.dll")
            .map(|i| i.export.to_string())
            .collect();
        assert_eq!(exports, ["XSystemGetDeviceType", "XSystemSetDeviceType"]);
        // Descriptor, null descriptor, null thunk and the symbols object.
        assert_eq!(lib.other_members, 4);
    }

    #[test]
    fn long_import_by_ordinal_uses_the_header_value() {
        let object = coff_object(&[
            "__IMPORT_DESCRIPTOR_xsystem.dll",
            "__imp_XSystemGetDeviceType",
        ]);
        let bytes = archive(&[long_descriptor(42, NAME_ORDINAL), object]);
        let lib = ImportLibrary::parse(&bytes).unwrap();
        assert_eq!(lib.imports.len(), 1);
        assert_eq!(lib.imports[0].export, ImportSymbol::Ordinal(42));
    }

    #[test]
    fn unknown_members_are_reported_not_silently_dropped() {
        let bytes = archive(&[vec![1, 2, 3]]);
        let lib = ImportLibrary::parse(&bytes).unwrap();
        assert!(lib.imports.is_empty());
        assert_eq!(lib.other_members, 1);
    }
}
