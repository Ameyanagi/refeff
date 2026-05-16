//! FEFF `XSPH/getoccnorm.f90` default occupation normalization data.
//!
//! The FEFF table is represented as sorted change points over `(Z, ihole)`
//! instead of a full 100x29 literal. This keeps the static data compact while
//! preserving the exact table entries used by `GetOccNorm`.

pub(crate) const XSPH_OCC_NORM_ATOMIC_NUMBER_MAX: usize = 100;
pub(crate) const XSPH_OCC_NORM_HOLE_COUNT: usize = 29;

const XSPH_OCC_NORM_EVENTS: &[(u8, u8, u8)] = &[
    (1, 1, 1),
    (2, 1, 2),
    (3, 2, 1),
    (4, 2, 2),
    (5, 3, 1),
    (6, 2, 1),
    (6, 3, 2),
    (6, 4, 1),
    (7, 2, 2),
    (8, 4, 2),
    (9, 4, 3),
    (10, 4, 4),
    (11, 5, 1),
    (12, 6, 1),
    (13, 5, 2),
    (14, 6, 2),
    (15, 7, 1),
    (16, 7, 2),
    (17, 7, 3),
    (18, 7, 4),
    (19, 10, 1),
    (20, 11, 1),
    (21, 8, 1),
    (22, 8, 2),
    (23, 8, 3),
    (24, 8, 4),
    (25, 9, 1),
    (26, 9, 2),
    (27, 9, 3),
    (28, 9, 4),
    (29, 9, 6),
    (29, 11, 0),
    (30, 11, 1),
    (31, 10, 2),
    (32, 11, 2),
    (33, 12, 1),
    (34, 12, 2),
    (35, 12, 3),
    (36, 12, 4),
    (37, 17, 1),
    (38, 17, 2),
    (39, 13, 1),
    (40, 13, 2),
    (41, 13, 4),
    (41, 17, 1),
    (42, 14, 1),
    (43, 17, 2),
    (44, 14, 3),
    (44, 17, 1),
    (45, 14, 4),
    (46, 14, 6),
    (46, 17, 0),
    (47, 17, 1),
    (48, 17, 2),
    (49, 18, 1),
    (50, 18, 2),
    (51, 19, 1),
    (52, 19, 2),
    (53, 19, 3),
    (54, 19, 4),
    (55, 24, 1),
    (56, 24, 2),
    (57, 20, 1),
    (58, 15, 1),
    (59, 15, 2),
    (60, 15, 3),
    (61, 15, 4),
    (62, 15, 5),
    (63, 15, 6),
    (64, 16, 1),
    (65, 16, 2),
    (66, 16, 3),
    (67, 16, 4),
    (68, 16, 5),
    (69, 16, 6),
    (70, 16, 7),
    (71, 16, 8),
    (72, 20, 2),
    (73, 20, 3),
    (74, 21, 1),
    (75, 20, 4),
    (76, 21, 2),
    (77, 21, 3),
    (78, 21, 5),
    (78, 24, 1),
    (79, 21, 6),
    (80, 24, 2),
    (81, 25, 1),
    (82, 25, 2),
    (83, 26, 1),
    (84, 26, 2),
    (85, 26, 3),
    (86, 26, 4),
    (87, 29, 1),
    (88, 29, 2),
    (89, 27, 1),
    (90, 27, 2),
    (91, 22, 2),
    (91, 27, 1),
    (92, 22, 3),
    (93, 22, 4),
    (94, 22, 6),
    (94, 27, 0),
    (95, 23, 1),
    (96, 23, 2),
    (97, 23, 3),
    (98, 23, 4),
    (99, 23, 5),
    (100, 23, 6),
];

const XSPH_OCC_NORM_DENOMINATORS: [u8; XSPH_OCC_NORM_HOLE_COUNT] = [
    2, 2, 2, 4, 2, 2, 4, 4, 6, 2, 2, 4, 4, 6, 6, 8, 2, 2, 4, 4, 6, 6, 6, 2, 2, 4, 0, 0, 2,
];

pub(crate) fn xsph_occ_norm_numerator(atomic_number: usize, hole_index: usize) -> Option<u8> {
    if !(1..=XSPH_OCC_NORM_ATOMIC_NUMBER_MAX).contains(&atomic_number)
        || !(1..=XSPH_OCC_NORM_HOLE_COUNT).contains(&hole_index)
    {
        return None;
    }

    let value = XSPH_OCC_NORM_EVENTS
        .iter()
        .take_while(|&&(event_atomic_number, _, _)| {
            usize::from(event_atomic_number) <= atomic_number
        })
        .filter(|&&(_, event_hole_index, _)| usize::from(event_hole_index) == hole_index)
        .fold(0_u8, |_, &(_, _, occupation)| occupation);
    Some(value)
}

pub(crate) fn xsph_occ_norm_denominator(hole_index: usize) -> Option<u8> {
    if !(1..=XSPH_OCC_NORM_HOLE_COUNT).contains(&hole_index) {
        return None;
    }
    XSPH_OCC_NORM_DENOMINATORS.get(hole_index - 1).copied()
}
