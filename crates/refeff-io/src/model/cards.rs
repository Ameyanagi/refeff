use crate::error::{IoError, Result};
use crate::input::{FeffInput, FeffLine, LineKind};

pub(super) fn parse_active_cards(input: &FeffInput) -> Vec<String> {
    let mut cards = input
        .cards()
        .filter_map(|line| match &line.kind {
            LineKind::Card { keyword, .. } => feff_card_token(keyword),
            LineKind::SectionData { .. } => None,
        })
        .collect::<Vec<_>>();
    cards.sort_by_key(|(token, _)| *token);
    cards.dedup_by_key(|(token, _)| *token);
    cards
        .into_iter()
        .map(|(_, display)| display.to_string())
        .collect()
}

pub(super) fn parse_input_cards(input: &FeffInput) -> Vec<String> {
    input
        .cards()
        .filter_map(|line| match &line.kind {
            LineKind::Card { keyword, .. } => {
                feff_card_token(keyword).map(|(_, display)| display.to_string())
            }
            LineKind::SectionData { .. } => None,
        })
        .collect()
}

pub(super) fn active_card(active_cards: &[String], canonical: &str) -> bool {
    active_cards.iter().any(|card| card == canonical)
}

pub(super) fn required_card_line<'a>(
    input: &'a FeffInput,
    canonical: &str,
) -> Result<&'a FeffLine> {
    card_by_feff_name(input, canonical).ok_or_else(|| IoError::Parse {
        path: input.source.clone(),
        line: 0,
        message: format!("{canonical} card not found"),
    })
}

pub(super) fn card_by_feff_name<'a>(input: &'a FeffInput, canonical: &str) -> Option<&'a FeffLine> {
    input.cards().find(|line| {
        if let LineKind::Card { keyword, .. } = &line.kind {
            return keyword == canonical
                || feff_card_token(keyword)
                    .map(|(_, display)| display == canonical)
                    .unwrap_or(false);
        }
        false
    })
}

pub(super) fn feff_card_token(keyword: &str) -> Option<(i32, &'static str)> {
    let upper = keyword.to_ascii_uppercase();
    let w = upper.get(..upper.len().min(4)).unwrap_or("");
    match w {
        "ATOM" => Some((1, "ATOMS")),
        "HOLE" => Some((2, "HOLE")),
        "OVER" => Some((3, "OVERLAP")),
        "CONT" => Some((4, "CONTROL")),
        "EXCH" => Some((5, "EXCHANGE")),
        "ION" => Some((6, "ION")),
        "TITL" => Some((7, "TITLE")),
        "FOLP" => Some((8, "FOLP")),
        "RPAT" | "RMAX" => Some((9, "RPATH")),
        "DEBY" => Some((10, "DEBYE")),
        "RMUL" => Some((11, "RMULT")),
        "SS" => Some((12, "SS")),
        "PRIN" => Some((13, "PRINT")),
        "POTE" => Some((14, "POTENTIALS")),
        "NLEG" => Some((15, "NLEG")),
        "CRIT" => Some((16, "CRITERIA")),
        "NOGE" => Some((17, "NOGEOM")),
        "IORD" => Some((18, "IORD")),
        "PCRI" => Some((19, "PCRITERIA")),
        "SIG2" => Some((20, "SIG2")),
        "XANE" => Some((21, "XANES")),
        "CORR" => Some((22, "CORRECTIONS")),
        "AFOL" => Some((23, "AFOLP")),
        "EXAF" => Some((24, "EXAFS")),
        "POLA" => Some((25, "POLARIZATION")),
        "ELLI" => Some((26, "ELLIPTICITY")),
        "RGRI" => Some((27, "RGRID")),
        "RPHA" => Some((28, "RPHASES")),
        "NSTA" => Some((29, "NSTAR")),
        "NOHO" => Some((30, "NOHOLE")),
        "SIG3" => Some((31, "SIG3")),
        "JUMP" => Some((32, "JUMPRM")),
        "MBCO" => Some((33, "MBCONV")),
        "SPIN" => Some((34, "SPIN")),
        "EDGE" => Some((35, "EDGE")),
        "SCF" => Some((36, "SCF")),
        "FMS" => Some((37, "FMS")),
        "LDOS" => Some((38, "LDOS")),
        "INTE" => Some((39, "INTERSTITIAL")),
        "CFAV" => Some((40, "CFAVERAGE")),
        "S02" => Some((41, "S02")),
        "XES" => Some((42, "XES")),
        "DANE" => Some((43, "DANES")),
        "FPRI" => Some((44, "FPRIME")),
        "RSIG" => Some((45, "RSIGMA")),
        "XNCD" | "XMCD" => Some((46, "XMCD")),
        "MULT" => Some((47, "MULT")),
        "UNFR" => Some((48, "UNFREEZEF")),
        "TDLD" => Some((49, "TDLDA")),
        "PMBS" => Some((50, "PMBSE")),
        "PLAS" | "MPSE" => Some((51, "MPSE")),
        "SO2C" | "SFCO" => Some((52, "SFCONV")),
        "SELF" => Some((53, "SELF")),
        "SFSE" => Some((54, "SFSE")),
        "RCON" => Some((55, "RCONV")),
        "ELNE" => Some((56, "ELNES")),
        "EXEL" => Some((57, "EXELFS")),
        "MAGI" => Some((58, "MAGIC")),
        "ABSO" => Some((59, "ABSOLUTE")),
        "SYMM" => Some((60, "SYMMETRY")),
        "REAL" => Some((61, "REAL")),
        "RECI" => Some((62, "RECIPROCAL")),
        "SGRO" => Some((63, "SGROUP")),
        "LATT" => Some((64, "LATTICE")),
        "KMES" => Some((65, "KMESH")),
        "STRF" => Some((66, "STRFAC")),
        "BAND" => Some((67, "BAND")),
        "CORE" => Some((68, "COREHOLE")),
        "MARK" | "TARG" => Some((71, "TARGET")),
        "EGRI" => Some((72, "EGRID")),
        "COOR" => Some((73, "COORDINATES")),
        "EXTP" => Some((74, "EXTPOT")),
        "CHBR" => Some((75, "CHBROADENING")),
        "CHSH" => Some((76, "CHSHIFT")),
        "DIMS" => Some((77, "DIMS")),
        "NRIX" => Some((78, "NRIXS")),
        "LJMA" => Some((79, "LJMAX")),
        "LDEC" => Some((80, "LDECMX")),
        "SETE" => Some((81, "SETE")),
        "EPS0" => Some((82, "EPS0")),
        "OPCO" => Some((83, "OPCONS")),
        "NUMD" => Some((84, "NUMD")),
        "PREP" => Some((85, "PREP")),
        "EGAP" => Some((86, "EGAP")),
        "CHWI" => Some((87, "CHWIDTH")),
        "MDFF" => Some((88, "MDFF")),
        "REST" => Some((89, "RESTART")),
        "CONF" => Some((90, "CONFIGURATION")),
        "SCRE" => Some((91, "SCREEN")),
        "CIF" => Some((92, "CIF")),
        "EQUI" => Some((93, "EQUIVALENCE")),
        "COMP" => Some((94, "COMPTON")),
        "RHOZ" => Some((95, "RHOZZP")),
        "CGRI" => Some((96, "CGRID")),
        "CORV" => Some((97, "CORVAL")),
        "SIGG" => Some((98, "SIGGK")),
        "TEMP" => Some((99, "TEMP")),
        "DENS" => Some((100, "DENS")),
        "RIXS" => Some((101, "RIXS")),
        "RLPR" => Some((102, "RLPR")),
        "ICOR" => Some((103, "ICOR")),
        "HUBB" => Some((104, "HUBBARD")),
        "CRPA" => Some((105, "CRPA")),
        "FULL" => Some((106, "FULLSPECTRUM")),
        "SCXC" => Some((107, "SCXC")),
        "HIGH" => Some((108, "HIGHZ")),
        "SCFT" => Some((109, "SCFTH")),
        "WARN" => Some((110, "WARN")),
        "SCFR" => Some((111, "SCFR")),
        "TOLS" => Some((112, "TOLS")),
        _ => None,
    }
}

pub(super) fn parse_titles(input: &FeffInput) -> Result<Vec<String>> {
    let mut titles = Vec::new();
    for line in input.cards() {
        if let LineKind::Card {
            keyword, raw_args, ..
        } = &line.kind
            && feff_card_token(keyword)
                .map(|(_, display)| display == "TITLE")
                .unwrap_or(false)
        {
            titles.push(raw_args.clone());
        }
    }
    Ok(titles)
}
