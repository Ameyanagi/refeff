use super::*;

/// Port of FEFF `hup`: bubble the last min-heap element upward.
///
/// `keys` and `indices` are swapped together so callers can keep path metadata
/// associated with the heap key, matching FEFF's `h` and `ih` arrays.
pub fn path_heap_bubble_up(keys: &mut [Real], indices: &mut [i32]) -> Result<(), PathError> {
    validate_heap_inputs(keys, indices)?;
    let mut child = keys.len().saturating_sub(1);
    while child > 0 {
        let parent = (child - 1) / 2;
        if keys[child] < keys[parent] {
            keys.swap(child, parent);
            indices.swap(child, parent);
            child = parent;
        } else {
            break;
        }
    }
    Ok(())
}

/// Port of FEFF `hdown`: bubble the root min-heap element downward.
///
/// This is used after the root has been replaced. The function preserves FEFF's
/// choice of the smaller child and swaps the companion index array with the key.
pub fn path_heap_bubble_down(keys: &mut [Real], indices: &mut [i32]) -> Result<(), PathError> {
    validate_heap_inputs(keys, indices)?;
    let mut parent = 0;
    loop {
        let left = 2 * parent + 1;
        if left >= keys.len() {
            break;
        }
        let right = left + 1;
        let child = if right < keys.len() && keys[left] > keys[right] {
            right
        } else {
            left
        };

        if keys[parent] > keys[child] {
            keys.swap(parent, child);
            indices.swap(parent, child);
            parent = child;
        } else {
            break;
        }
    }
    Ok(())
}

fn validate_heap_inputs(keys: &[Real], indices: &[i32]) -> Result<(), PathError> {
    if keys.len() != indices.len() {
        return Err(PathError::HeapLengthMismatch {
            keys_len: keys.len(),
            indices_len: indices.len(),
        });
    }
    for (index, &value) in keys.iter().enumerate() {
        if !value.is_finite() {
            return Err(PathError::NonFiniteHeapKey { index, value });
        }
    }
    Ok(())
}
