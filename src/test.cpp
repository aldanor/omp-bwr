#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <typeinfo>
#include <vector>

#include <omp.h>

template<typename T>
std::vector<T> generate_data(int n) {
    std::vector<T> arr {};
    T num = 1;
    for (int i = 0; i < n; ++i) {
        arr.push_back(num / (static_cast<T>(i) + num));
    }
    return arr;
}

struct params_t {
    int size;
    int num_threads;
    bool dynamic;
};

std::ostream& operator<<(std::ostream& os, const params_t& p) {
    os << "size:" << p.size << " num_threads:" << p.num_threads << " dynamic:" << p.dynamic;
    return os;
}

template<typename T>
T run(const params_t& p) {
    auto data = generate_data<T>(p.size); 
    omp_set_num_threads(p.num_threads);
    omp_set_dynamic(p.dynamic);
    T result = 0;
    #pragma omp parallel for schedule(static) reduction(+:result)
    for (int i = 0; i < data.size(); ++i) {
        result += data[i];
    }
    return result;
}

template<typename T>
struct results_t {
    params_t params;
    std::map<T, int> map;
    double elapsed;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const results_t<T>& r) {
    os << "type:" << typeid(T).name();
    os << " " << r.params;
    os << " elapsed:" << (r.elapsed * 1e3) << "ms";
    os << std::hexfloat;
    for (const auto& kv : r.map) {
        os << std::endl << "  " << kv.first << " (" << kv.second << ")";
    }
    os << std::defaultfloat;
    return os;
}

template<typename T>
results_t<T> run_many(const params_t& params, int n_trials) {
    std::map<T, int> map;
    auto t0 = omp_get_wtime();
    for (int i = 0; i < n_trials; ++i) {
        map[run<T>(params)]++;
    }
    auto t1 = omp_get_wtime();
    return {params, map, t1 - t0};
}

int main() {
    const int size = 234567;
    const int n = 1000;
    for (int num_threads : {1, 2, 3, 4}) {
        std::cout << "---" << std::endl;
        for (bool dynamic : {false, true}) {
            std::cout << run_many<float>({size, num_threads, dynamic}, n) << std::endl;
            std::cout << run_many<double>({size, num_threads, dynamic}, n) << std::endl;;
        }
    }
}
