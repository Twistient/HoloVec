use std::cmp::Ordering;
use std::ffi::c_void;
use std::slice;

#[derive(Clone, Copy, Debug)]
struct RankedScore {
    index: usize,
    score: f32,
}

fn rank_scores(scores: &[f32], k: usize) -> Vec<RankedScore> {
    let mut ranked: Vec<RankedScore> = scores
        .iter()
        .enumerate()
        .map(|(index, score)| RankedScore {
            index,
            score: *score,
        })
        .collect();
    ranked.sort_by(|left, right| match right.score.total_cmp(&left.score) {
        Ordering::Equal => left.index.cmp(&right.index),
        other => other,
    });
    ranked.truncate(k);
    ranked
}

struct ContinuousIndex {
    rows: usize,
    cols: usize,
    matrix: Vec<f32>,
    row_norms: Vec<f32>,
}

impl ContinuousIndex {
    fn new(matrix: &[f32], rows: usize, cols: usize) -> Self {
        let row_norms = matrix
            .chunks_exact(cols)
            .map(|row| row.iter().map(|value| value * value).sum::<f32>().sqrt())
            .collect();
        Self {
            rows,
            cols,
            matrix: matrix.to_vec(),
            row_norms,
        }
    }

    fn query(
        &self,
        query: &[f32],
        k: usize,
        out_indices: &mut [usize],
        out_scores: &mut [f32],
    ) -> i32 {
        if k == 0
            || k > self.rows
            || query.len() != self.cols
            || out_indices.len() < k
            || out_scores.len() < k
        {
            return -1;
        }

        let query_norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
        let mut scores = vec![0.0_f32; self.rows];
        if query_norm != 0.0 {
            for (row_index, row) in self.matrix.chunks_exact(self.cols).enumerate() {
                let dot = row
                    .iter()
                    .zip(query.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                let denom = self.row_norms[row_index] * query_norm;
                scores[row_index] = if denom == 0.0 {
                    0.0
                } else {
                    (dot / denom).clamp(-1.0, 1.0)
                };
            }
        }

        let ranked = rank_scores(&scores, k);
        for (out_index, ranked_score) in ranked.iter().enumerate() {
            out_indices[out_index] = ranked_score.index;
            out_scores[out_index] = ranked_score.score;
        }
        0
    }
}

struct DiscreteIndex {
    rows: usize,
    cols: usize,
    matrix: Vec<i8>,
}

impl DiscreteIndex {
    fn new(matrix: &[i8], rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            matrix: matrix.to_vec(),
        }
    }

    fn query(
        &self,
        query: &[i8],
        k: usize,
        out_indices: &mut [usize],
        out_scores: &mut [f32],
    ) -> i32 {
        if k == 0
            || k > self.rows
            || query.len() != self.cols
            || out_indices.len() < k
            || out_scores.len() < k
        {
            return -1;
        }

        let mut scores = vec![0.0_f32; self.rows];
        let scale = self.cols as f32;
        for (row_index, row) in self.matrix.chunks_exact(self.cols).enumerate() {
            let matches = row
                .iter()
                .zip(query.iter())
                .filter(|(left, right)| left == right)
                .count();
            scores[row_index] = matches as f32 / scale;
        }

        let ranked = rank_scores(&scores, k);
        for (out_index, ranked_score) in ranked.iter().enumerate() {
            out_indices[out_index] = ranked_score.index;
            out_scores[out_index] = ranked_score.score;
        }
        0
    }
}

struct SparseIndex {
    rows: usize,
    cols: usize,
    matrix: Vec<u8>,
    row_counts: Vec<u32>,
}

impl SparseIndex {
    fn new(matrix: &[u8], rows: usize, cols: usize) -> Self {
        let row_counts = matrix
            .chunks_exact(cols)
            .map(|row| row.iter().map(|value| *value as u32).sum::<u32>())
            .collect();
        Self {
            rows,
            cols,
            matrix: matrix.to_vec(),
            row_counts,
        }
    }

    fn query(
        &self,
        query: &[u8],
        k: usize,
        out_indices: &mut [usize],
        out_scores: &mut [f32],
    ) -> i32 {
        if k == 0
            || k > self.rows
            || query.len() != self.cols
            || out_indices.len() < k
            || out_scores.len() < k
        {
            return -1;
        }

        let query_count = query.iter().map(|value| *value as u32).sum::<u32>();
        let mut scores = vec![0.0_f32; self.rows];
        for (row_index, row) in self.matrix.chunks_exact(self.cols).enumerate() {
            let intersections = row
                .iter()
                .zip(query.iter())
                .map(|(left, right)| (*left as u32) * (*right as u32))
                .sum::<u32>();
            let denom = self.row_counts[row_index].min(query_count);
            scores[row_index] = if denom == 0 {
                0.0
            } else {
                intersections as f32 / denom as f32
            };
        }

        let ranked = rank_scores(&scores, k);
        for (out_index, ranked_score) in ranked.iter().enumerate() {
            out_indices[out_index] = ranked_score.index;
            out_scores[out_index] = ranked_score.score;
        }
        0
    }
}

struct SegmentIndex {
    rows: usize,
    segments: usize,
    patterns: Vec<u32>,
}

impl SegmentIndex {
    fn new(patterns: &[u32], rows: usize, segments: usize) -> Self {
        Self {
            rows,
            segments,
            patterns: patterns.to_vec(),
        }
    }

    fn query(
        &self,
        query: &[u32],
        k: usize,
        out_indices: &mut [usize],
        out_scores: &mut [f32],
    ) -> i32 {
        if k == 0
            || k > self.rows
            || query.len() != self.segments
            || out_indices.len() < k
            || out_scores.len() < k
        {
            return -1;
        }

        let mut scores = vec![0.0_f32; self.rows];
        let scale = self.segments as f32;
        for (row_index, row) in self.patterns.chunks_exact(self.segments).enumerate() {
            let matches = row
                .iter()
                .zip(query.iter())
                .filter(|(left, right)| left == right)
                .count();
            scores[row_index] = matches as f32 / scale;
        }

        let ranked = rank_scores(&scores, k);
        for (out_index, ranked_score) in ranked.iter().enumerate() {
            out_indices[out_index] = ranked_score.index;
            out_scores[out_index] = ranked_score.score;
        }
        0
    }
}

#[no_mangle]
pub extern "C" fn holovec_continuous_index_new(
    matrix_ptr: *const f32,
    rows: usize,
    cols: usize,
) -> *mut c_void {
    if matrix_ptr.is_null() || rows == 0 || cols == 0 {
        return std::ptr::null_mut();
    }
    let len = rows.saturating_mul(cols);
    let matrix = unsafe { slice::from_raw_parts(matrix_ptr, len) };
    Box::into_raw(Box::new(ContinuousIndex::new(matrix, rows, cols))) as *mut c_void
}

#[no_mangle]
pub extern "C" fn holovec_continuous_index_query(
    index_ptr: *const c_void,
    query_ptr: *const f32,
    k: usize,
    out_indices_ptr: *mut usize,
    out_scores_ptr: *mut f32,
) -> i32 {
    if index_ptr.is_null()
        || query_ptr.is_null()
        || out_indices_ptr.is_null()
        || out_scores_ptr.is_null()
    {
        return -1;
    }
    let index = unsafe { &*(index_ptr as *const ContinuousIndex) };
    let query = unsafe { slice::from_raw_parts(query_ptr, index.cols) };
    let out_indices = unsafe { slice::from_raw_parts_mut(out_indices_ptr, k) };
    let out_scores = unsafe { slice::from_raw_parts_mut(out_scores_ptr, k) };
    index.query(query, k, out_indices, out_scores)
}

#[no_mangle]
pub extern "C" fn holovec_continuous_index_free(index_ptr: *mut c_void) {
    if !index_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(index_ptr as *mut ContinuousIndex));
        }
    }
}

#[no_mangle]
pub extern "C" fn holovec_discrete_index_new(
    matrix_ptr: *const i8,
    rows: usize,
    cols: usize,
) -> *mut c_void {
    if matrix_ptr.is_null() || rows == 0 || cols == 0 {
        return std::ptr::null_mut();
    }
    let len = rows.saturating_mul(cols);
    let matrix = unsafe { slice::from_raw_parts(matrix_ptr, len) };
    Box::into_raw(Box::new(DiscreteIndex::new(matrix, rows, cols))) as *mut c_void
}

#[no_mangle]
pub extern "C" fn holovec_discrete_index_query(
    index_ptr: *const c_void,
    query_ptr: *const i8,
    k: usize,
    out_indices_ptr: *mut usize,
    out_scores_ptr: *mut f32,
) -> i32 {
    if index_ptr.is_null()
        || query_ptr.is_null()
        || out_indices_ptr.is_null()
        || out_scores_ptr.is_null()
    {
        return -1;
    }
    let index = unsafe { &*(index_ptr as *const DiscreteIndex) };
    let query = unsafe { slice::from_raw_parts(query_ptr, index.cols) };
    let out_indices = unsafe { slice::from_raw_parts_mut(out_indices_ptr, k) };
    let out_scores = unsafe { slice::from_raw_parts_mut(out_scores_ptr, k) };
    index.query(query, k, out_indices, out_scores)
}

#[no_mangle]
pub extern "C" fn holovec_discrete_index_free(index_ptr: *mut c_void) {
    if !index_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(index_ptr as *mut DiscreteIndex));
        }
    }
}

#[no_mangle]
pub extern "C" fn holovec_sparse_index_new(
    matrix_ptr: *const u8,
    rows: usize,
    cols: usize,
) -> *mut c_void {
    if matrix_ptr.is_null() || rows == 0 || cols == 0 {
        return std::ptr::null_mut();
    }
    let len = rows.saturating_mul(cols);
    let matrix = unsafe { slice::from_raw_parts(matrix_ptr, len) };
    Box::into_raw(Box::new(SparseIndex::new(matrix, rows, cols))) as *mut c_void
}

#[no_mangle]
pub extern "C" fn holovec_sparse_index_query(
    index_ptr: *const c_void,
    query_ptr: *const u8,
    k: usize,
    out_indices_ptr: *mut usize,
    out_scores_ptr: *mut f32,
) -> i32 {
    if index_ptr.is_null()
        || query_ptr.is_null()
        || out_indices_ptr.is_null()
        || out_scores_ptr.is_null()
    {
        return -1;
    }
    let index = unsafe { &*(index_ptr as *const SparseIndex) };
    let query = unsafe { slice::from_raw_parts(query_ptr, index.cols) };
    let out_indices = unsafe { slice::from_raw_parts_mut(out_indices_ptr, k) };
    let out_scores = unsafe { slice::from_raw_parts_mut(out_scores_ptr, k) };
    index.query(query, k, out_indices, out_scores)
}

#[no_mangle]
pub extern "C" fn holovec_sparse_index_free(index_ptr: *mut c_void) {
    if !index_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(index_ptr as *mut SparseIndex));
        }
    }
}

#[no_mangle]
pub extern "C" fn holovec_segment_index_new(
    patterns_ptr: *const u32,
    rows: usize,
    segments: usize,
) -> *mut c_void {
    if patterns_ptr.is_null() || rows == 0 || segments == 0 {
        return std::ptr::null_mut();
    }
    let len = rows.saturating_mul(segments);
    let patterns = unsafe { slice::from_raw_parts(patterns_ptr, len) };
    Box::into_raw(Box::new(SegmentIndex::new(patterns, rows, segments))) as *mut c_void
}

#[no_mangle]
pub extern "C" fn holovec_segment_index_query(
    index_ptr: *const c_void,
    query_ptr: *const u32,
    k: usize,
    out_indices_ptr: *mut usize,
    out_scores_ptr: *mut f32,
) -> i32 {
    if index_ptr.is_null()
        || query_ptr.is_null()
        || out_indices_ptr.is_null()
        || out_scores_ptr.is_null()
    {
        return -1;
    }
    let index = unsafe { &*(index_ptr as *const SegmentIndex) };
    let query = unsafe { slice::from_raw_parts(query_ptr, index.segments) };
    let out_indices = unsafe { slice::from_raw_parts_mut(out_indices_ptr, k) };
    let out_scores = unsafe { slice::from_raw_parts_mut(out_scores_ptr, k) };
    index.query(query, k, out_indices, out_scores)
}

#[no_mangle]
pub extern "C" fn holovec_segment_index_free(index_ptr: *mut c_void) {
    if !index_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(index_ptr as *mut SegmentIndex));
        }
    }
}
