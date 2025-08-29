#include <bits/stdc++.h>
#include <omp.h>
#include <cstdint>   // <- burası əlavə olundu
using namespace std;

using u32 = uint32_t;  // 32-bit
using u64 = uint64_t;  // 64-bit

// Bit-pack helper
vector<uint8_t> pack_bits_from_u64s(const vector<u64>& vals) {
    u64 maxv = 0;
    for (u64 v : vals) if (v > maxv) maxv = v;
    unsigned bits = 1;
    while ((1ULL<<bits) <= maxv) ++bits;
    if(bits==0) bits = 1;
    size_t total_bits = vals.size() * bits;
    size_t total_bytes = (total_bits + 7) / 8;
    vector<uint8_t> out(total_bytes, 0);
    size_t bitpos = 0;
    for (u64 v : vals) {
        for (unsigned b = 0; b < bits; ++b) {
            size_t byte_idx = bitpos >> 3;
            unsigned bit_in_byte = bitpos & 7;
            if ((v >> b) & 1ULL) out[byte_idx] |= (1u << bit_in_byte);
            ++bitpos;
        }
    }
    return out;
}

// Block metadata
struct Block {
    u32 max_value;
    u64 cumulative;
    size_t start_idx;
    size_t size;
};

// Find block
int find_block(const vector<Block>& blocks, u32 x){
    int lo=0, hi=(int)blocks.size()-1, ans=hi;
    while(lo<=hi){
        int mid=(lo+hi)>>1;
        if(blocks[mid].max_value >= x){ ans=mid; hi=mid-1; }
        else lo=mid+1;
    }
    return ans;
}

// Count < x using blocks
u64 query_count_less_main(const vector<u32>& sorted, const vector<Block>& blocks, u32 x){
    if(sorted.empty()) return 0;
    if(x <= sorted.front()) return 0;
    if(x > sorted.back()) return sorted.size();
    int bi = find_block(blocks, x);
    const Block &b = blocks[bi];
    u64 before = b.cumulative;
    auto it_begin = sorted.begin() + (ptrdiff_t)b.start_idx;
    auto it_end   = it_begin + (ptrdiff_t)b.size;
    auto it = lower_bound(it_begin, it_end, x);
    size_t local = (size_t)(it - it_begin);
    return before + (u64)local;
}

int main() {
    const size_t N = 1'000'000;
    const size_t NUM_BLOCKS = 32;

    mt19937_64 rng(123456789ULL);
    uniform_int_distribution<u32> dist(1, 1'000'000'000);

    vector<u32> nums(N);
    for(size_t i=0;i<N;++i) nums[i]=dist(rng);

    auto t0 = chrono::high_resolution_clock::now();
    vector<u32> sorted = nums;
    sort(sorted.begin(), sorted.end());
    auto t1 = chrono::high_resolution_clock::now();
    cerr << "Sort time: " 
         << chrono::duration<double>(t1-t0).count() << " s\n";

    // Build blocks (simple equal-sized)
    vector<Block> blocks(NUM_BLOCKS);
    size_t block_size = N / NUM_BLOCKS;
    u64 cum=0;
    for(size_t i=0;i<NUM_BLOCKS;++i){
        blocks[i].start_idx = i*block_size;
        blocks[i].size = (i==NUM_BLOCKS-1) ? N-i*block_size : block_size;
        blocks[i].cumulative = cum;
        blocks[i].max_value = sorted[blocks[i].start_idx + blocks[i].size - 1];
        cum += blocks[i].size;
    }

    vector<u64> results(N);

    // ------------------- Multi-threaded queries -------------------
    t0 = chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 1000)
    for(size_t i=0;i<N;++i){
        results[i] = query_count_less_main(sorted, blocks, nums[i]);
    }
    t1 = chrono::high_resolution_clock::now();
    cerr << "Query all elements (multi-threaded): " 
         << chrono::duration<double>(t1-t0).count() << " s\n";

    auto packed = pack_bits_from_u64s(results);
    cerr << "Packed bytes: " << packed.size() << "\n";

    // Print sample
    for(size_t i=0;i<20;++i)
        cout << nums[i] << " -> " << results[i] << "\n";

    return 0;
}
