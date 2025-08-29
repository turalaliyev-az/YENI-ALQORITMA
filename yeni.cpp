#include <bits/stdc++.h>
using namespace std;
using u32 = uint32_t;
using u64 = uint64_t;

// Block metadata
struct Block {
    u32 max_value;        // maximum value inside this block
    u64 cumulative;       // number of elements strictly before this block (global)
    size_t start_idx;     // start index in sorted array
    size_t size;          // size of block
};

// Generate Fibonacci numbers up to at least K terms
vector<u64> generate_fib_sequence(size_t k) {
    vector<u64> F;
    if (k == 0) return F;
    F.push_back(1);
    if (k == 1) return F;
    F.push_back(1);
    while (F.size() < k) {
        F.push_back(F[F.size()-1] + F[F.size()-2]);
    }
    return F;
}

// Build Fibonacci-proportional blocks from sorted_nums.
// Idea: compute some Fib ns (k terms) whose sum S; then block_size_i = round(Fi * N / S).
vector<Block> build_index_fib(const vector<u32>& sorted_nums, size_t desired_k = 40) {
    size_t N = sorted_nums.size();
    // choose k such that sum of fibs not too small/large; desired_k default 40 (enough granularity)
    vector<u64> F = generate_fib_sequence(desired_k);

    // compute sum
    long double sumF = 0;
    for (auto &x : F) sumF += (long double)x;

    // compute block sizes proportional to Fi
    vector<size_t> sizes;
    sizes.reserve(F.size());
    size_t allocated = 0;
    for (size_t i = 0; i < F.size(); ++i) {
        size_t s = (size_t)floor((long double)N * (long double)F[i] / sumF + 0.5L);
        if (s == 0) s = 1; // ensure non-zero
        sizes.push_back(s);
        allocated += s;
    }
    // adjust to match N (add/remove from last)
    if (allocated != N) {
        sizes.back() += (N - allocated);
    }

    // build blocks
    vector<Block> blocks;
    blocks.reserve(sizes.size());
    size_t idx = 0;
    u64 cumulative = 0;
    for (size_t i = 0; i < sizes.size() && idx < N; ++i) {
        size_t sz = sizes[i];
        if (idx + sz > N) sz = N - idx;
        Block b;
        b.start_idx = idx;
        b.size = sz;
        b.cumulative = cumulative;
        b.max_value = sorted_nums[idx + sz - 1];
        blocks.push_back(b);
        idx += sz;
        cumulative += sz;
    }
    // In case we ended earlier (very unlikely), merge remainder into last block
    if (idx < N) {
        if (blocks.empty()) {
            Block b; b.start_idx = 0; b.size = N; b.cumulative = 0; b.max_value = sorted_nums[N-1];
            blocks.push_back(b);
        } else {
            Block &last = blocks.back();
            last.size += (N - idx);
            last.max_value = sorted_nums[last.start_idx + last.size - 1];
        }
    }
    return blocks;
}

// Find block index by value x using block.max_value array (first block with max_value >= x)
int find_block(const vector<Block>& blocks, u32 x) {
    int lo = 0, hi = (int)blocks.size() - 1, ans = hi;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (blocks[mid].max_value >= x) { ans = mid; hi = mid - 1; }
        else lo = mid + 1;
    }
    return ans;
}

// Query: how many elements in sorted_nums are strictly < x
u64 query_count_less(const vector<u32>& sorted_nums, const vector<Block>& blocks, u32 x) {
    if (sorted_nums.empty()) return 0;
    int blk_idx = find_block(blocks, x);
    const Block &blk = blocks[blk_idx];
    u64 before = blk.cumulative;
    // within block, count elements < x using upper_bound
    auto begin_it = sorted_nums.begin() + (ptrdiff_t)blk.start_idx;
    auto end_it   = begin_it + (ptrdiff_t)blk.size;
    auto it = std::lower_bound(begin_it, end_it, x); // first >= x => # < x = it - begin_it
    size_t local = (size_t)(it - begin_it);
    return before + (u64)local;
}

// Pack vector<u64> results into minimal bits and write to vector<uint8_t> (bit-packed)
// This is a simple packer: finds max value -> bits_needed, packs sequentially MSB->LSB within bytes.
vector<uint8_t> pack_bits(const vector<u64>& results) {
    u64 maxv = 0;
    for (u64 v : results) if (v > maxv) maxv = v;
    unsigned bits = 1;
    while ((1ULL << bits) <= maxv) ++bits;
    if (bits == 0) bits = 1;
    size_t total_bits = (size_t)results.size() * bits;
    size_t total_bytes = (total_bits + 7) / 8;
    vector<uint8_t> out(total_bytes, 0);
    size_t bitpos = 0;
    for (u64 v : results) {
        for (unsigned b = 0; b < bits; ++b) {
            size_t byte_idx = bitpos >> 3;
            unsigned bit_in_byte = bitpos & 7;
            // store LSB-first into stream (you can flip ordering if desired)
            if ( (v >> b) & 1ULL ) out[byte_idx] |= (1u << bit_in_byte);
            ++bitpos;
        }
    }
    return out;
}

// Reference merge-based correct solution (for validation)
vector<int> countSmaller_merge_ref(const vector<int>& nums) {
    int n = nums.size();
    vector<int> idx(n), tmp(n), count(n, 0);
    iota(idx.begin(), idx.end(), 0);
    function<void(int,int)> merge_count = [&](int l, int r){
        if (r - l <= 1) return;
        int m = (l + r) >> 1;
        merge_count(l, m);
        merge_count(m, r);
        int i = l, j = m, k = l;
        int movedFromRight = 0;
        while (i < m && j < r) {
            if (nums[idx[j]] < nums[idx[i]]) {
                tmp[k++] = idx[j++];
                movedFromRight++;
            } else {
                count[idx[i]] += movedFromRight;
                tmp[k++] = idx[i++];
            }
        }
        while (i < m) {
            count[idx[i]] += movedFromRight;
            tmp[k++] = idx[i++];
        }
        while (j < r) tmp[k++] = idx[j++];
        for (int t = l; t < r; ++t) idx[t] = tmp[t];
    };
    merge_count(0, n);
    return count;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Example input (use the array you gave earlier)
    vector<u32> nums = {26,78,27,100,33,67,90,23,66,5,38,7,35,23,68,52,22,83,51,98,69,81,111,32,78,28,94,13,2,97,3,76,99,51,9,21,84,66,65,36,100,41,110,};

    // Build sorted array
    vector<u32> sorted_nums = nums;
    sort(sorted_nums.begin(), sorted_nums.end());

    // Build Fibonacci-based index (in-memory demo; for huge N, use external sort and build index during writing)
    size_t fib_k = 30; // tweak for granularity
    vector<Block> blocks = build_index_fib(sorted_nums, fib_k);

    // Print block summary
    cerr << "Blocks created: " << blocks.size() << "\n";
    for (size_t i = 0; i < blocks.size(); ++i) {
        cerr << "Block " << i << ": start=" << blocks[i].start_idx << " size=" << blocks[i].size
             << " max=" << blocks[i].max_value << " cum=" << blocks[i].cumulative << "\n";
    }

    // Query all original elements
    vector<u64> results;
    results.reserve(nums.size());
    for (u32 x : nums) {
        u64 cnt = query_count_less(sorted_nums, blocks, x);
        results.push_back(cnt);
    }

    // Print results
    cout << "Query results (counts of elements < x):\n";
    for (size_t i = 0; i < nums.size(); ++i) {
        cout << results[i] << (i+1==nums.size() ? "\n" : " ");
    }

    // Validate against merge-based correct solution (for small N)
    vector<int> ref = countSmaller_merge_ref(vector<int>(nums.begin(), nums.end()));
    bool ok = true;
    if (ref.size() == results.size()) {
        for (size_t i = 0; i < results.size(); ++i) {
            if ((u64)ref[i] != results[i]) { ok = false; break; }
        }
    } else ok = false;
    cerr << "Validation vs merge-ref: " << (ok ? "OK" : "MISMATCH") << "\n";

    // Pack results to bits
    vector<uint8_t> packed = pack_bits(results);
    cerr << "Packed results bytes: " << packed.size() << "\n";

    // Example: unpack and show first few entries (to check packing)
    // (unpacking omitted for brevity; pack_bits used LSB-first order)

    return 0;
}

