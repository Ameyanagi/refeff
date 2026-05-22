use super::*;

/// Port of FEFF `ipack`: pack up to eight path indices into three integers.
///
/// Each path index must be in `0..=1289`, matching the base used by FEFF. The
/// first packed integer stores the path length and the first two indices; the
/// second and third packed integers store the remaining six indices.
pub fn pack_path_indices(indices: &[i32]) -> Result<[i32; 3], PathError> {
    if indices.len() > MAX_PACKED_PATH_INDICES {
        return Err(PathError::TooManyIndices {
            count: indices.len(),
            max: MAX_PACKED_PATH_INDICES,
        });
    }

    let mut padded = [0; MAX_PACKED_PATH_INDICES];
    for (position, &value) in indices.iter().enumerate() {
        validate_path_index(position, value)?;
        padded[position] = value;
    }

    Ok([
        i32::try_from(indices.len()).map_err(|_| PathError::TooManyIndices {
            count: indices.len(),
            max: MAX_PACKED_PATH_INDICES,
        })? + padded[0] * PATH_PACK_BASE
            + padded[1] * PATH_PACK_BASE_SQUARED,
        padded[2] + padded[3] * PATH_PACK_BASE + padded[4] * PATH_PACK_BASE_SQUARED,
        padded[5] + padded[6] * PATH_PACK_BASE + padded[7] * PATH_PACK_BASE_SQUARED,
    ])
}

/// Port of FEFF `upack`: unpack a three-integer path representation.
///
/// `capacity` mirrors FEFF's caller-provided maximum `n`: it must be at most
/// eight and must be no smaller than the packed path length.
pub fn unpack_path_indices(packed: [i32; 3], capacity: usize) -> Result<Vec<i32>, PathError> {
    if capacity > MAX_PACKED_PATH_INDICES {
        return Err(PathError::InvalidUnpackCapacity {
            capacity,
            max: MAX_PACKED_PATH_INDICES,
        });
    }
    for (position, &value) in packed.iter().enumerate() {
        if value < 0 {
            return Err(PathError::NegativePackedValue { position, value });
        }
    }

    let packed_count = usize::try_from(packed[0] % PATH_PACK_BASE).map_err(|_| {
        PathError::NegativePackedValue {
            position: 0,
            value: packed[0],
        }
    })?;
    if packed_count > MAX_PACKED_PATH_INDICES {
        return Err(PathError::TooManyIndices {
            count: packed_count,
            max: MAX_PACKED_PATH_INDICES,
        });
    }
    if packed_count > capacity {
        return Err(PathError::UnpackCapacityTooSmall {
            packed_count,
            capacity,
        });
    }

    let unpacked = [
        (packed[0] % PATH_PACK_BASE_SQUARED) / PATH_PACK_BASE,
        packed[0] / PATH_PACK_BASE_SQUARED,
        packed[1] % PATH_PACK_BASE,
        (packed[1] % PATH_PACK_BASE_SQUARED) / PATH_PACK_BASE,
        packed[1] / PATH_PACK_BASE_SQUARED,
        packed[2] % PATH_PACK_BASE,
        (packed[2] % PATH_PACK_BASE_SQUARED) / PATH_PACK_BASE,
        packed[2] / PATH_PACK_BASE_SQUARED,
    ];

    Ok(unpacked[..packed_count].to_vec())
}

fn validate_path_index(position: usize, value: i32) -> Result<(), PathError> {
    if (0..=MAX_PACKED_PATH_VALUE).contains(&value) {
        Ok(())
    } else {
        Err(PathError::IndexOutOfRange {
            position,
            value,
            max: MAX_PACKED_PATH_VALUE,
        })
    }
}
