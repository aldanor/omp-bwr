#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
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
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const results_t<T>& r) {
    os << r.params;
    for (const auto& kv : r.map) {
        os << std::endl << "  " << std::hexfloat << kv.first << " (" << kv.second << ")";
    }
    return os;
}

template<typename T>
results_t<T> run_many(const params_t& params, int n_trials) {
    std::map<T, int> map;
    for (int i = 0; i < n_trials; ++i) {
        map[run<T>(params)]++;
    }
    return {params, map};
}

int main() {
    params_t p1 {12345, 4, true};
    params_t p2 {12345, 4, false};
    int n = 10000;
    std::cout << run_many<float>(p1, n) << std::endl;
    std::cout << run_many<double>(p1, n) << std::endl;
    std::cout << run_many<float>(p2, n) << std::endl;
    std::cout << run_many<double>(p2, n) << std::endl;
}
