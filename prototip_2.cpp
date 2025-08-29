// filename: hybrid_fib_prime_blocks.cpp
// compile: g++ -O3 -march=native -std=c++17 hybrid_fib_prime_blocks.cpp -o hybrid_fib_prime_blocks
// run: ./hybrid_fib_prime_blocks

#include <bits/stdc++.h>
using namespace std;
using u32 = uint32_t;
using u64 = uint64_t;
using chrono_clock = chrono::high_resolution_clock;

// -------------------- Block metadata (same as before) --------------------
struct Block {
    u32 max_value;
    u64 cumulative;   // how many elements strictly before this block
    size_t start_idx;
    size_t size;
};

// -------------------- Fibonacci generator (values <= maxVal) --------------------
vector<u64> gen_fib_values_upto(u64 maxVal) {
    vector<u64> fib;
    u64 a = 1, b = 1;
    if (maxVal >= 1) fib.push_back(1);
    if (maxVal >= 1) fib.push_back(1);
    while (true) {
        u64 c = a + b;
        if (c > maxVal) break;
        fib.push_back(c);
        a = b; b = c;
    }
    // The above may have duplicate '1' twice — if desired we can unique-ify:
    sort(fib.begin(), fib.end());
    fib.erase(unique(fib.begin(), fib.end()), fib.end());
    return fib;
}

// -------------------- Sieve up to LIMIT (basic) --------------------
// Note: LIMIT should be chosen carefully (memory/time). Default in main uses 2e6.
vector<int> sieve_primes_upto(int LIMIT) {
    vector<char> isPrime(LIMIT+1, true);
    isPrime[0]=isPrime[1]=false;
    int r = (int)floor(sqrt((double)LIMIT));
    for (int p=2; p<=r; ++p) if (isPrime[p]) {
        for (long long q = 1LL*p*p; q <= LIMIT; q += p) isPrime[(size_t)q] = false;
    }
    vector<int> primes;
    for (int i=2;i<=LIMIT;++i) if (isPrime[i]) primes.push_back(i);
    return primes;
}

// -------------------- Fibonacci-proportional block builder (robust) --------------------
vector<u64> gen_fib_seq(size_t k){
    vector<u64> F;
    if(k==0) return F;
    F.push_back(1);
    if(k==1) return F;
    F.push_back(1);
    while(F.size()<k) F.push_back(F[F.size()-1] + F[F.size()-2]);
    return F;
}

vector<Block> build_index_fib(const vector<u32>& sorted, size_t fib_k=32){
    size_t N = sorted.size();
    vector<u64> F = gen_fib_seq(fib_k);
    long double sumF = 0;
    for(auto &v: F) sumF += (long double)v;

    // initial sizes via floor
    vector<size_t> sizes(F.size(), 0);
    size_t allocated = 0;
    vector<long double> exacts(F.size());
    for(size_t i=0;i<F.size();++i){
        long double exact = ((long double)N) * (long double)F[i] / sumF;
        exacts[i] = exact;
        sizes[i] = (size_t)floor(exact);
        allocated += sizes[i];
    }
    // distribute leftover to largest fractional parts
    size_t leftover = (size_t)(N - allocated);
    vector<pair<long double,size_t>> frac;
    for(size_t i=0;i<F.size();++i) frac.emplace_back(-(exacts[i]-floor(exacts[i])), i);
    sort(frac.begin(), frac.end());
    size_t idx = 0;
    while(leftover > 0 && idx < frac.size()){
        sizes[frac[idx].second] += 1;
        leftover -= 1;
        ++idx;
        if(idx == frac.size() && leftover > 0) idx = 0;
    }
    // ensure non-zero if N>0
    for(size_t i=0;i<sizes.size();++i) if(sizes[i]==0 && N>0) sizes[i]=1;

    // build blocks
    vector<Block> blocks;
    blocks.reserve(sizes.size());
    size_t cur = 0;
    u64 cum = 0;
    for(size_t i=0;i<sizes.size() && cur < N; ++i){
        size_t sz = sizes[i];
        if(cur + sz > N) sz = N - cur;
        if(sz==0) continue;
        Block b;
        b.start_idx = cur;
        b.size = sz;
        b.cumulative = cum;
        b.max_value = sorted[cur + sz - 1];
        blocks.push_back(b);
        cur += sz;
        cum += sz;
    }
    if(cur < N){
        if(blocks.empty()){
            Block b; b.start_idx=0; b.size=N; b.cumulative=0; b.max_value=sorted[N-1]; blocks.push_back(b);
        } else {
            Block &last = blocks.back();
            last.size += (N - cur);
            last.max_value = sorted[last.start_idx + last.size - 1];
        }
    }
    return blocks;
}

// -------------------- Block find + query (main array) --------------------
int find_block(const vector<Block>& blocks, u32 x){
    int lo=0, hi=(int)blocks.size()-1, ans=hi;
    while(lo<=hi){
        int mid=(lo+hi)>>1;
        if(blocks[mid].max_value >= x){ ans=mid; hi=mid-1; }
        else lo=mid+1;
    }
    return ans;
}

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

// -------------------- bit pack helper (same as earlier) --------------------
vector<uint8_t> pack_bits_from_u64s(const vector<u64>& vals) {
    u64 maxv = 0;
    for (u64 v : vals) if (v > maxv) maxv = v;
    unsigned bits = 1;
    while ((1ULL<<bits) <= maxv) ++bits;
    if(bits==0) bits = 1;
    size_t total_bits = (size_t)vals.size() * bits;
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

// -------------------- main demo --------------------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const size_t N = 1'000'000;        // 1 million
    const unsigned FIB_K = 32;         // number of Fibonacci-proportional blocks
    const int PRIME_LIMIT_DEFAULT = 2'000'000; // safe sieve limit (adjust if you have more RAM/time)

    // 1) Generate random data (reproducible)
    mt19937_64 rng(123456789ULL);
    uniform_int_distribution<u32> dist(1, 1'000'000'000);
    cerr << "Generating " << N << " random elements...\n";
    vector<u32> nums; nums.reserve(N);
    for(size_t i=0;i<N;++i) nums.push_back(dist(rng));

    // 2) Sort (in-memory for demo)
    auto t0 = chrono_clock::now();
    vector<u32> sorted = nums;
    sort(sorted.begin(), sorted.end());
    auto t1 = chrono_clock::now();
    double sort_time = chrono::duration<double>(t1 - t0).count();
    cerr << "Sorted N in " << sort_time << " s\n";

    // 3) Build fibonacci-proportional blocks for main array
    t0 = chrono_clock::now();
    vector<Block> blocks = build_index_fib(sorted, FIB_K);
    t1 = chrono_clock::now();
    double build_time = chrono::duration<double>(t1 - t0).count();
    cerr << "Built " << blocks.size() << " blocks in " << build_time << " s\n";

    // 4) Build Fibonacci values up to max(sorted)
    u64 maxVal = sorted.empty() ? 0 : sorted.back();
    vector<u64> fibvals = gen_fib_values_upto(maxVal);
    cerr << "Fibonacci values up to max (" << maxVal << "): count = " << fibvals.size() << "\n";

    // 5) Build primes up to a practical limit (default 2e6)
    int prime_limit = (int)min<u64>((u64)PRIME_LIMIT_DEFAULT, maxVal);
    vector<int> primes = sieve_primes_upto(prime_limit);
    cerr << "Primes up to " << prime_limit << ": count = " << primes.size() << "\n";

    // 6) For demo: query all original nums and compute:
    //    a) how many main elements < x (via blocks)
    //    b) how many fib values < x (via lower_bound on fibvals)
    //    c) how many primes < x (if x <= prime_limit via lower_bound on primes,
    //       else if x > prime_limit, we answer primes.size() (approx) — note limitation)
    t0 = chrono_clock::now();
    vector<u64> results_main; results_main.reserve(N);
    vector<u64> results_fib;  results_fib.reserve(N);
    vector<u64> results_pr;   results_pr.reserve(N);

    for(size_t i=0;i<N;++i){
        u32 x = nums[i];
        u64 cnt_main = query_count_less_main(sorted, blocks, x);
        // fib count
        u64 cnt_fib = (u64)(lower_bound(fibvals.begin(), fibvals.end(), (u64)x) - fibvals.begin());
        // prime count (limited)
        u64 cnt_pr;
        if (x <= (u32)prime_limit) {
            cnt_pr = (u64)(lower_bound(primes.begin(), primes.end(), (int)x) - primes.begin());
        } else {
            // we can't know primes beyond sieve limit here; return full sieve count as lower bound
            cnt_pr = primes.size();
        }
        results_main.push_back(cnt_main);
        results_fib.push_back(cnt_fib);
        results_pr.push_back(cnt_pr);
    }
    t1 = chrono_clock::now();
    double query_time = chrono::duration<double>(t1 - t0).count();
    cerr << "Queried all elements in " << query_time << " s (avg per query " << (query_time/(double)N) << " s)\n";

    // 7) Pack (example: pack main-results)
    auto packed_main = pack_bits_from_u64s(results_main);
    auto packed_fib  = pack_bits_from_u64s(results_fib);
    auto packed_pr   = pack_bits_from_u64s(results_pr);
    cerr << "Packed bytes (main/fib/pr): " << packed_main.size() << "/" << packed_fib.size() << "/" << packed_pr.size() << "\n";

    // 8) Print small sample
    cout << "Sample first 20 results: (value -> main,fib,pr)\n";
    for(size_t i=0;i<20 && i<N;++i){
        cout << nums[i] << " -> " << results_main[i] << "," << results_fib[i] << "," << results_pr[i] << "\n";
    }

    // 9) Summary
    cout << "\nSummary:\n";
    cout << "N=" << N << "\n";
    cout << "sort_s=" << sort_time << "\n";
    cout << "index_build_s=" << build_time << "\n";
    cout << "query_all_s=" << query_time << "\n";
    cout << "blocks=" << blocks.size() << "\n";
    cout << "fib_count=" << fibvals.size() << "\n";
    cout << "prime_limit=" << prime_limit << ", prime_count=" << primes.size() << "\n";
    cout << "packed_main_bytes=" << packed_main.size() << "\n";

    return 0;
}
