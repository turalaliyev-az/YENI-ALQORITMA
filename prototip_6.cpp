#include <bits/stdc++.h>
#include <omp.h>
#include <cmath>
#include <atomic>
#include <memory>
#include <vector>
#include <algorithm>
using namespace std;

using u32 = uint32_t;
using u64 = uint64_t;

// ===================== OPTİMİZE MATEMATİKSEL FONKSİYONLAR =====================
namespace MathUtils {
    static vector<u32> fib_cache = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144};
    static vector<u32> prime_cache = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
    
    u32 fibonacci(u32 n) {
        if (n < fib_cache.size()) return fib_cache[n];
        if (n == 0) return 0;
        if (n <= 2) return 1;
        double phi = (1.0 + sqrt(5.0)) / 2.0;
        return round(pow(phi, n) / sqrt(5.0));
    }

    bool is_prime(u32 n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (u32 i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    }

    u32 nth_prime(u32 n) {
        if (n < prime_cache.size()) return prime_cache[n];
        if (n == 0) return 2;
        u32 count = prime_cache.size();
        u32 num = prime_cache.back() + 2;
        while (count <= n) {
            if (is_prime(num)) {
                count++;
                if (count > prime_cache.size()) {
                    prime_cache.push_back(num);
                }
            }
            num += 2;
        }
        return prime_cache[n];
    }

    const double GOLDEN_RATIO = 1.6180339887498948482;
}

// ===================== OPTİMİZE FİBONACCİ-ASAL BLOKLAMA =====================
class OptimizedFibonacciPrimeBlockManager {
private:
    vector<u32> sorted_data;
    
    size_t calculate_optimized_block_size(size_t block_index, size_t remaining_data, size_t total_blocks) const {
        if (remaining_data <= 64) return remaining_data;
        u32 fib_val = MathUtils::fibonacci((block_index % 12) + 1);
        u32 prime_val = MathUtils::nth_prime((block_index % 15) + 1);
        double golden_factor = 1.0 + (block_index % 10) * 0.1;
        double scale_factor = min(10.0, max(0.1, log2(remaining_data) / 10.0));
        double dynamic_size = (fib_val * prime_val * golden_factor * scale_factor) / total_blocks;
        size_t block_size = static_cast<size_t>(dynamic_size) % remaining_data;
        size_t max_reasonable = max(size_t(64), min(remaining_data, size_t(10000)));
        size_t min_reasonable = max(size_t(64), remaining_data / 100);
        block_size = clamp(block_size, min_reasonable, max_reasonable);
        const size_t cache_line_size = 64;
        const size_t element_size = sizeof(u32);
        const size_t elements_per_line = cache_line_size / element_size;
        block_size = (block_size + elements_per_line - 1) / elements_per_line * elements_per_line;
        return min(block_size, remaining_data);
    }

public:
    struct Block {
        u32 mathematical_hash;
        u64 cumulative_count;
        size_t start_index;
        size_t block_size;
        double entropy;
        u32 max_value;
        Block() : mathematical_hash(0), cumulative_count(0), 
                 start_index(0), block_size(0), entropy(0.0), max_value(0) {}
    };

    vector<Block> blocks;

    OptimizedFibonacciPrimeBlockManager(const vector<u32>& data) : sorted_data(data) {
        sort(sorted_data.begin(), sorted_data.end());
        create_optimized_blocks();
    }

    void create_optimized_blocks() {
        size_t n = sorted_data.size();
        if (n == 0) return;
        size_t optimal_blocks = max(size_t(1), 
            static_cast<size_t>(2.0 * log(n) / log(MathUtils::GOLDEN_RATIO)));
        optimal_blocks = min(optimal_blocks, size_t(100));
        blocks.clear();
        size_t current_index = 0;
        u64 cumulative = 0;
        for (size_t i = 0; i < optimal_blocks && current_index < n; ++i) {
            Block block;
            block.start_index = current_index;
            size_t remaining = n - current_index;
            block.block_size = (i == optimal_blocks - 1) ? remaining : calculate_optimized_block_size(i, remaining, optimal_blocks);
            if (block.block_size == 0 || current_index + block.block_size > n) {
                block.block_size = n - current_index;
            }
            size_t end_index = current_index + block.block_size;
            if (end_index > n) end_index = n;
            block.max_value = (end_index > current_index) ? sorted_data[end_index - 1] : sorted_data[current_index];
            block.cumulative_count = cumulative;
            blocks.push_back(block);
            cumulative += block.block_size;
            current_index = end_index;
        }
        if (current_index < n) {
            Block block;
            block.start_index = current_index;
            block.block_size = n - current_index;
            block.max_value = sorted_data[n - 1];
            block.cumulative_count = cumulative;
            blocks.push_back(block);
        }
    }

    int find_target_block(u32 value) const {
        if (blocks.empty()) return 0;
        if (value <= sorted_data.front()) return 0;
        if (value > sorted_data.back()) return blocks.size() - 1;
        int left = 0, right = blocks.size() - 1;
        int result = 0;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (blocks[mid].max_value >= value) {
                result = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return result;
    }

    u64 mathematical_query(u32 value) const {
        if (sorted_data.empty()) return 0;
        if (value <= sorted_data.front()) return 0;
        if (value > sorted_data.back()) return sorted_data.size();
        int block_idx = find_target_block(value);
        if (block_idx < 0 || block_idx >= static_cast<int>(blocks.size())) {
            return sorted_data.size();
        }
        const Block& block = blocks[block_idx];
        auto start_it = sorted_data.begin() + block.start_index;
        auto end_it = start_it + block.block_size;
        if (end_it > sorted_data.end()) {
            end_it = sorted_data.end();
        }
        auto pos = lower_bound(start_it, end_it, value);
        return block.cumulative_count + static_cast<u64>(pos - start_it);
    }
};

// ===================== ANA PROGRAM =====================
int main() {
    cout << "🔬 OPTİMİZE FİBONACCİ-ASAL BLOKLAMA ALGORİTMASI\n\n";
    const size_t DATA_SIZE = 100000000;
    vector<u32> test_data(DATA_SIZE);
    mt19937 rng(42);
    uniform_int_distribution<u32> dist(1, 100000000);
    for (size_t i = 0; i < DATA_SIZE; ++i) {
        test_data[i] = dist(rng);
    }
    vector<u32> test_queries = {0, 1, 50000, 100000, 100001};
    for (int i = 0; i < 20; ++i) {
        test_queries.push_back(dist(rng));
    }
    auto start_time = chrono::high_resolution_clock::now();
    OptimizedFibonacciPrimeBlockManager manager(test_data);
    vector<u64> results;
    for (const auto& query : test_queries) {
        results.push_back(manager.mathematical_query(query));
    }
    auto end_time = chrono::high_resolution_clock::now();
    double duration = chrono::duration<double>(end_time - start_time).count();
    cout << "=== PERFORMANS İSTATİSTİKLERİ ===\n";
    cout << "Toplam blok sayısı: " << manager.blocks.size() << "\n";
    size_t min_size = numeric_limits<size_t>::max();
    size_t max_size = 0;
    for (const auto& block : manager.blocks) {
        min_size = min(min_size, block.block_size);
        max_size = max(max_size, block.block_size);
    }
    cout << "Blok boyutu dağılımı: " << min_size << " - " << max_size << "\n";
    cout << "\n=== BLOK DETAYLARI ===\n";
    for (size_t i = 0; i < manager.blocks.size(); ++i) {
        const auto& block = manager.blocks[i];
        cout << "Blok " << i << ": start=" << block.start_index 
             << ", size=" << block.block_size 
             << ", max_value=" << block.max_value 
             << ", cumulative=" << block.cumulative_count << "\n";
    }
    cout << "\n=== DOĞRULAMA ===\n";
    bool all_correct = true;
    int error_count = 0;
    vector<u32> sorted_test = test_data;
    sort(sorted_test.begin(), sorted_test.end());
    for (size_t i = 0; i < test_queries.size(); ++i) {
        u64 expected = lower_bound(sorted_test.begin(), sorted_test.end(), test_queries[i]) - sorted_test.begin();
        if (results[i] != expected) {
            if (error_count < 5) {
                cout << "✗ HATA: Sorgu " << test_queries[i] << " → Beklenen: " << expected 
                     << ", Algoritma: " << results[i] << "\n";
            }
            error_count++;
            all_correct = false;
        }
    }
    if (all_correct) {
        cout << "✓ Tüm sorgular doğru!\n";
    } else {
        cout << "✗ Toplam " << error_count << " hata bulundu!\n";
    }
    cout << "\n⏱️  Toplam süre: " << duration << " saniye\n";
    return 0;
}