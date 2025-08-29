// filename: hybrid_fib_blocks.cpp
// compile: g++ -O3 -march=native -std=c++17 hybrid_fib_blocks.cpp -o hybrid_fib_blocks
// run: ./hybrid_fib_blocks
/*
#include <bits/stdc++.h>
using namespace std;
using u32 = uint32_t;
using u64 = uint64_t;
using chrono_clock = std::chrono::high_resolution_clock;

struct Block {
    u32 max_value;
    u64 cumulative;   // how many elements strictly before this block
    size_t start_idx;
    size_t size;
};

vector<u64> gen_fib(size_t k){
    vector<u64> F;
    if(k==0) return F;
    F.push_back(1);
    if(k==1) return F;
    F.push_back(1);
    while(F.size()<k) F.push_back(F[F.size()-1]+F[F.size()-2]);
    return F;
}

vector<Block> build_index_fib(const vector<u32>& sorted, size_t fib_k=30){
    size_t N = sorted.size();
    vector<u64> F = gen_fib(fib_k);
    long double sumF=0;
    for(auto &x: F) sumF += (long double)x;
    vector<size_t> sizes; sizes.reserve(F.size());
    size_t allocated = 0;
    for(size_t i=0;i<F.size();++i){
        size_t s = (size_t)floor((long double)N * (long double)F[i] / sumF + 0.5L);
        if(s==0) s=1;
        sizes.push_back(s);
        allocated += s;
    }
    if(allocated != N){
        sizes.back() += (N - allocated);
    }
    vector<Block> blocks;
    blocks.reserve(sizes.size());
    size_t idx=0; u64 cum=0;
    for(size_t i=0;i<sizes.size() && idx<N; ++i){
        size_t sz = sizes[i];
        if(idx+sz > N) sz = N - idx;
        Block b;
        b.start_idx = idx;
        b.size = sz;
        b.cumulative = cum;
        b.max_value = sorted[idx + sz - 1];
        blocks.push_back(b);
        idx += sz;
        cum += sz;
    }
    if(idx < N){
        if(blocks.empty()){
            Block b; b.start_idx=0; b.size=N; b.cumulative=0; b.max_value=sorted[N-1]; blocks.push_back(b);
        } else {
            Block &last = blocks.back();
            last.size += (N - idx);
            last.max_value = sorted[last.start_idx + last.size - 1];
        }
    }
    return blocks;
}

int find_block(const vector<Block>& blocks, u32 x){
    int lo=0, hi=(int)blocks.size()-1, ans=hi;
    while(lo<=hi){
        int mid=(lo+hi)>>1;
        if(blocks[mid].max_value >= x){ ans=mid; hi=mid-1; }
        else lo=mid+1;
    }
    return ans;
}

u64 query_count_less(const vector<u32>& sorted, const vector<Block>& blocks, u32 x){
    if(sorted.empty()) return 0;
    int bi = find_block(blocks, x);
    const Block &b = blocks[bi];
    u64 before = b.cumulative;
    auto it_begin = sorted.begin() + (ptrdiff_t)b.start_idx;
    auto it_end   = it_begin + (ptrdiff_t)b.size;
    auto it = lower_bound(it_begin, it_end, x);
    size_t local = (size_t)(it - it_begin);
    return before + (u64)local;
}

// reference merge-count (works for validation, O(n log n))
vector<int> merge_count_ref(const vector<int>& arr){
    int n=arr.size();
    vector<int> idx(n), tmp(n), res(n,0);
    iota(idx.begin(), idx.end(), 0);
    function<void(int,int)> rec = [&](int l,int r){
        if(r-l<=1) return;
        int m=(l+r)>>1;
        rec(l,m); rec(m,r);
        int i=l,j=m,k=l;
        int moved=0;
        while(i<m && j<r){
            if(arr[idx[j]] < arr[idx[i]]){
                tmp[k++]=idx[j++]; moved++;
            } else {
                res[idx[i]] += moved;
                tmp[k++]=idx[i++];
            }
        }
        while(i<m){ res[idx[i]] += moved; tmp[k++]=idx[i++]; }
        while(j<r) tmp[k++]=idx[j++];
        for(int t=l;t<r;++t) idx[t]=tmp[t];
    };
    rec(0,n);
    return res;
}

// bit-pack results (LSB-first per value, minimal bits for max)
vector<uint8_t> pack_bits(const vector<u64>& vals){
    u64 maxv=0;
    for(auto &v: vals) if(v>maxv) maxv=v;
    unsigned bits = 1;
    while((1ULL<<bits) <= maxv) ++bits;
    if(bits==0) bits=1;
    size_t total_bits = (size_t)vals.size()*bits;
    size_t total_bytes = (total_bits+7)/8;
    vector<uint8_t> out(total_bytes, 0);
    size_t bitpos=0;
    for(u64 v : vals){
        for(unsigned b=0;b<bits;++b){
            size_t byte_idx = bitpos >> 3;
            unsigned bit_in_byte = bitpos & 7;
            if( (v>>b) & 1ULL ) out[byte_idx] |= (1u << bit_in_byte);
            ++bitpos;
        }
    }
    return out;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const size_t N = 1'000'000;          // 1 million elements
    const unsigned FIB_K = 32;           // granularity of Fibonacci blocks

    // generate random data (use fixed seed for reproducibility)
    mt19937_64 rng(123456789ULL);
    uniform_int_distribution<u32> dist(1, 1'000'000'000);

    cerr << "Generating " << N << " random elements...\n";
    vector<u32> nums; nums.reserve(N);
    for(size_t i=0;i<N;++i) nums.push_back(dist(rng));

    // measure sorting
    auto t0 = chrono_clock::now();
    vector<u32> sorted = nums;
    sort(sorted.begin(), sorted.end());
    auto t1 = chrono_clock::now();
    double sort_time = chrono::duration<double>(t1 - t0).count();
    cerr << "Sorting done in " << sort_time << " s\n";

    // build Fibonacci blocks index
    t0 = chrono_clock::now();
    vector<Block> blocks = build_index_fib(sorted, FIB_K);
    t1 = chrono_clock::now();
    double build_time = chrono::duration<double>(t1 - t0).count();
    cerr << "Built " << blocks.size() << " blocks in " << build_time << " s\n";

    // Query all original elements and collect results
    t0 = chrono_clock::now();
    vector<u64> results; results.reserve(N);
    for(size_t i=0;i<N;++i){
        u64 c = query_count_less(sorted, blocks, nums[i]);
        results.push_back(c);
    }
    t1 = chrono_clock::now();
    double query_time = chrono::duration<double>(t1 - t0).count();
    cerr << "Queried all elements in " << query_time << " s (total " << N << " queries)\n";
    cerr << "Avg per query: " << (query_time / double(N)) << " s\n";

    // Validate correctness on a subset (to save time): take first 10000 elems
    cerr << "Validating correctness on 10000-sample via merge-ref...\n";
    size_t sample = 10000;
    vector<int> sample_input(sample);
    for(size_t i=0;i<sample;++i) sample_input[i] = (int)nums[i];
    auto tval0 = chrono_clock::now();
    vector<int> ref = merge_count_ref(sample_input);
    auto tval1 = chrono_clock::now();
    cerr << "Reference computed in " << chrono::duration<double>(tval1 - tval0).count() << " s\n";
    bool ok = true;
    for(size_t i=0;i<sample;++i){
        if((u64)ref[i] != results[i]) { ok=false; break; }
    }
    cerr << "Validation (first " << sample << "): " << (ok? "OK" : "MISMATCH") << "\n";

    // Pack results to bits
    t0 = chrono_clock::now();
    vector<uint8_t> packed = pack_bits(results);
    t1 = chrono_clock::now();
    cerr << "Packed results into " << packed.size() << " bytes in " 
         << chrono::duration<double>(t1 - t0).count() << " s\n";

    // Print summary to stdout (first 20 results)
    cout << "First 20 elements and their counts (<x):\n";
    for(size_t i=0;i<20 && i<N;++i){
        cout << nums[i] << " -> " << results[i] << "\n";
    }
    cout << "\nSummary:\n";
    cout << "N=" << N << "\n";
    cout << "sort_time_s=" << sort_time << "\n";
    cout << "index_build_s=" << build_time << "\n";
    cout << "query_time_s=" << query_time << "\n";
    cout << "packed_bytes=" << packed.size() << "\n";
    cout << "validation_sample_ok=" << (ok?1:0) << "\n";

    return 0;
}
*/


























// filename: debug_fib_blocks.cpp
// compile: g++ -O3 -march=native -std=c++17 debug_fib_blocks.cpp -o debug_fib_blocks
// run: ./debug_fib_blocks

#include <bits/stdc++.h>
using namespace std;
using u32 = uint32_t;
using u64 = uint64_t;
using chrono_clock = chrono::high_resolution_clock;

struct Block {
    u32 max_value;
    u64 cumulative;   // how many elements strictly before this block
    size_t start_idx;
    size_t size;
};

// generate Fibonacci sequence of length k (1,1,2,3,...)
vector<u64> gen_fib(size_t k){
    vector<u64> F;
    if(k==0) return F;
    F.push_back(1);
    if(k==1) return F;
    F.push_back(1);
    while(F.size() < k) F.push_back(F[F.size()-1] + F[F.size()-2]);
    return F;
}

// build blocks proportionally to Fibonacci numbers, but distribute rounding remainder fairly
vector<Block> build_index_fib(const vector<u32>& sorted, size_t fib_k=32){
    size_t N = sorted.size();
    vector<u64> F = gen_fib(fib_k);
    long double sumF = 0;
    for(auto &v: F) sumF += (long double)v;

    // initial sizes via floor, track remainder
    vector<size_t> sizes(F.size(), 0);
    u64 allocated = 0;
    for(size_t i=0;i<F.size();++i){
        long double exact = ((long double)N) * (long double)F[i] / sumF;
        sizes[i] = (size_t)floor(exact);
        allocated += sizes[i];
    }
    // distribute leftover (N - allocated) to blocks with largest fractional parts to minimize error
    size_t leftover = (size_t)(N - allocated);
    // compute fractional parts
    vector<pair<long double, size_t>> frac; // (fractional_part, index)
    frac.reserve(F.size());
    for(size_t i=0;i<F.size();++i){
        long double exact = ((long double)N) * (long double)F[i] / sumF;
        long double fractional = exact - floor(exact);
        frac.emplace_back(-fractional, i); // negative so sorting asc gives largest fractional first
    }
    sort(frac.begin(), frac.end());
    size_t idx = 0;
    while(leftover > 0 && idx < frac.size()){
        sizes[frac[idx].second] += 1;
        leftover -= 1;
        ++idx;
        if(idx == frac.size() && leftover > 0) idx = 0; // wrap if still leftover
    }
    // ensure no zero-size blocks (if any zero occurs and we have space, give 1)
    for(size_t i=0;i<sizes.size();++i){
        if(sizes[i] == 0 && N > 0){
            sizes[i] = 1;
        }
    }
    // Now create contiguous blocks from sizes (skip trailing zeros if total > N)
    vector<Block> blocks;
    blocks.reserve(sizes.size());
    size_t cur = 0;
    u64 cum = 0;
    for(size_t i=0;i<sizes.size() && cur < N; ++i){
        size_t sz = sizes[i];
        if(cur + sz > N) sz = N - cur;
        if(sz == 0) continue;
        Block b;
        b.start_idx = cur;
        b.size = sz;
        b.cumulative = cum;
        b.max_value = sorted[cur + sz - 1];
        blocks.push_back(b);
        cur += sz;
        cum += sz;
    }
    // final safety: if not covered everything, append last block
    if(cur < N){
        if(blocks.empty()){
            Block b; b.start_idx = 0; b.size = N; b.cumulative = 0; b.max_value = sorted[N-1];
            blocks.push_back(b);
        } else {
            Block &last = blocks.back();
            last.size += (N - cur);
            last.max_value = sorted[last.start_idx + last.size - 1];
        }
    }
    return blocks;
}

int find_block(const vector<Block>& blocks, u32 x){
    int lo=0, hi=(int)blocks.size()-1, ans=hi;
    while(lo<=hi){
        int mid=(lo+hi)>>1;
        if(blocks[mid].max_value >= x){ ans=mid; hi=mid-1; }
        else lo=mid+1;
    }
    return ans;
}

u64 query_via_blocks(const vector<u32>& sorted, const vector<Block>& blocks, u32 x){
    // special quick cases
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

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const size_t N = 1'000'000; // test with 1M
    const size_t FIB_K = 32;

    // generate data
    mt19937_64 rng(123456789ULL);
    uniform_int_distribution<u32> dist(1, 1'000'000'000);

    cerr << "Generating " << N << " random elements...\n";
    vector<u32> nums; nums.reserve(N);
    for(size_t i=0;i<N;++i) nums.push_back(dist(rng));

    // sort
    auto t0 = chrono_clock::now();
    vector<u32> sorted = nums;
    sort(sorted.begin(), sorted.end());
    auto t1 = chrono_clock::now();
    cerr << "Sort time: " << chrono::duration<double>(t1 - t0).count() << " s\n";

    // build fib blocks
    t0 = chrono_clock::now();
    vector<Block> blocks = build_index_fib(sorted, FIB_K);
    t1 = chrono_clock::now();
    cerr << "Blocks built: " << blocks.size() << ", build time: " << chrono::duration<double>(t1 - t0).count() << " s\n";

    // quick sanity print first few blocks
    for(size_t i=0;i<min<size_t>(10, blocks.size()); ++i){
        cerr << "B[" << i << "] start=" << blocks[i].start_idx << " sz=" << blocks[i].size
             << " cum=" << blocks[i].cumulative << " max=" << blocks[i].max_value << "\n";
    }

    // Query and validate: compare block-based answer vs global lower_bound
    cerr << "Running validation over sample (will stop at first mismatch)...\n";
    size_t checked = 0;
    for(size_t i=0;i<N;++i){
        u32 x = nums[i];
        // reference: number of elements < x
        u64 ref = (u64)(lower_bound(sorted.begin(), sorted.end(), x) - sorted.begin());
        u64 via = query_via_blocks(sorted, blocks, x);
        if(ref != via){
            cerr << "MISMATCH at i=" << i << " x=" << x << "\n";
            cerr << "reference (global lower_bound) = " << ref << "\n";
            cerr << "via blocks = " << via << "\n";
            // show block details for ref position
            size_t ref_idx = (size_t)ref;
            size_t blk_for_ref = 0;
            for(size_t b=0;b<blocks.size();++b){
                if(ref_idx >= blocks[b].start_idx && ref_idx < blocks[b].start_idx + blocks[b].size){ blk_for_ref = b; break; }
            }
            cerr << "ref pos in sorted = " << ref_idx << ", block " << blk_for_ref
                 << " (start=" << blocks[blk_for_ref].start_idx << ", sz=" << blocks[blk_for_ref].size << ", max=" << blocks[blk_for_ref].max_value << ", cum=" << blocks[blk_for_ref].cumulative << ")\n";
            // show block found by find_block(x)
            int bi = find_block(blocks, x);
            const Block &bb = blocks[bi];
            cerr << "block found by x: idx=" << bi << " start=" << bb.start_idx << " sz=" << bb.size << " cum=" << bb.cumulative << " max=" << bb.max_value << "\n";
            // print neighborhood of sorted around ref
            size_t start_print = (ref_idx > 5 ? ref_idx - 5 : 0);
            size_t end_print = min(sorted.size(), ref_idx + 6);
            cerr << "sorted around ref pos:\n";
            for(size_t p = start_print; p < end_print; ++p){
                cerr << "  [" << p << "]=" << sorted[p] << "\n";
            }
            return 0;
        }
        if(++checked % 100000 == 0) cerr << "checked " << checked << "...\n";
    }
    cerr << "All checks passed: no mismatch found for N=" << N << "\n";
    return 0;
}
