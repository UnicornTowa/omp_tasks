//
// Created by tosha on 26/10/2023.
//
#include "funcs.h"

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<int> dist(1, 65536);
uniform_int_distribution<int> dist_1k(1, 1024);
uniform_int_distribution<int> dist_mil(0, 1048575);
uniform_real_distribution<double> dist_double(1, 65536);

int find_min(vector<int>& nums){
    auto min = nums[0];
#pragma omp parallel for shared(min, nums) default(none)
    for (int i = 1; i < nums.size(); i++) {
        auto num = nums[i];
        if (num < min) {
#pragma omp critical
            if (num < min)
                min = num;
        }
    }
    return min;
}

void task_1(){
    cout << "Task 1, find min, threads: 1, 2, ... , 20; len: 10, 100, ... , 10 000 000; \n"
            " each row - same len, col - same thread_num, time in microseconds" << endl;

    for (int len = 10; len <= 10000000; len *= 10){
        for (int thread_num = 1; thread_num <= 20; thread_num++){
            vector<int> vec = {};
            for (int i = 0; i < len; i++){
                auto num = dist(gen);
                vec.push_back(num);
            }
            omp_set_num_threads(thread_num);
            auto start_time = chrono::high_resolution_clock::now();
            auto res = find_min(vec);
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            cout << duration.count() << ' ';
        }
        cout << endl;
    }
}

ULL dot_product(const vector<ULL>& v1, const vector<ULL>& v2){
    ULL result = 0;
#pragma omp parallel for shared(v1, v2) reduction(+: result) default(none)
    for(int i = 0; i < v1.size(); i++){
        result += v1[i]*v2[i];
    }
    return result;
}

ULL dot_product_np(const vector<ULL>& v1, const vector<ULL>& v2){
    ULL result = 0;
    for(int i = 0; i < v1.size(); i++){
        result += v1[i]*v2[i];
    }
    return result;
}

void task_2(){
    cout << "Task 2, find dot product, threads: 1, 2, ... , 20; len: 10, 100, ... , 10 000 000; \n"
            " each row - same len, col - same thread_num, time in microseconds" << endl;

    for (int len = 10; len <= 10000000; len *= 10){
        for (int thread_num = 1; thread_num <= 20; thread_num++){
            vector<ULL > vec1 = {};
            vector<ULL> vec2 = {};
            for (int i = 0; i < len; i++){
                vec1.push_back(dist_mil(gen));
                vec2.push_back(dist_mil(gen));
            }
            omp_set_num_threads(thread_num);
            auto start_time = chrono::high_resolution_clock::now();
            auto res = dot_product(vec1, vec2);
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            cout << duration.count() << ' ';
        }
        cout << endl;
    }
}

auto func(double x){
    return pow(cos(x), 2);
}

double integrate(double a, double b, int n){
    auto h = (b - a) / n;
    double res = 0;
#pragma omp parallel for default(none) shared(a, h, n) reduction(+: res)
    for (int i = 0; i < n; i++){
        res += func(a + i*h);
    }
    return res * h;
}

void task_3(){
    cout << "Task 3, find integral of cos(x)^2, threads: 1, 2, ... , 50; len: 10, 100, ... , 10 000 000; \n"
            " each row - same len, col - same thread_num, time in microseconds" << endl;
    ofstream resFile("results.txt");
    for (int len = 10; len <= 10000000; len *= 10){
        for (int thread_num = 1; thread_num <= 50; thread_num++){
            omp_set_num_threads(thread_num);
            auto start_time = chrono::high_resolution_clock::now();
            auto res = integrate(0, M_PI / 2, len);
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            resFile << duration.count() << ' ';
        }
        resFile << endl;
    }
    resFile.close();
}

int max_min(vector<vector<int>> matrix){
    auto res = 0;
#pragma omp parallel for default(none) shared(matrix, res, cout)
    for(int i = 0; i < matrix.size(); i++){
        auto row_min = matrix[i][0] > 0 ? matrix[i][0] : 65537;
#pragma omp parallel for default(none) shared(matrix, i, row_min)
        for(int j = 1; j < matrix.size(); j++){
            if (matrix[i][j] > 0 && matrix[i][j] < row_min){
#pragma omp critical
                if (matrix[i][j] > 0 && matrix[i][j] < row_min) {
                    row_min = matrix[i][j];
                }
            }
        }
        if (row_min > res){
#pragma omp critical
            if(row_min > res) {
                res = row_min;
            }
        }
    }
    return res;
}

void task_4(){
    cout << "Task 4, find minmax of matrix, threads: 1, 2, ... , 30; len: 10, 100, 1000, 10 000; \n"
            " each row - same len, col - same thread_num, time in microseconds" << endl;
    ofstream resFile("results.txt");
    for (int len = 10; len <= 10000; len *= 10){
        for (int thread_num = 1; thread_num <= 30; thread_num++){
            vector<vector<int>> matrix(len, vector<int>(len, 0));
            for (int i = 0; i < len; i++){
                for(int j = 0; j < len; j++){
                    matrix[i][j] = dist(gen);
                }
            }
            omp_set_nested(1);
            omp_set_num_threads(thread_num);
            auto start_time = chrono::high_resolution_clock::now();
            auto res = max_min(matrix);
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            resFile << duration.count() << ' ';
        }
        resFile << endl;
    }
    resFile.close();
}

int max_min_tri(vector<vector<int>> matrix){
    auto res = 0;
#pragma omp parallel for default(none) shared(matrix, res, cout) schedule(dynamic)
    for(int i = 0; i < matrix.size(); i++){
        auto row_min = matrix[i][0];
        for(int j = 1; j <= i; j++){
            if (matrix[i][j] < row_min){
                row_min = matrix[i][j];
            }
        }
        if (row_min > res){
#pragma omp critical
            if(row_min > res) {
                res = row_min;
            }
        }
    }
    return res;
}

void task_5_a(){
    cout << "Task 5, find minmax of lower triangular matrix, threads: 1, 2, ... , 30; len: 10, 100, 1000, 10 000; \n"
            " each row - same len, col - same thread_num, time in microseconds" << endl;
    ofstream resFile("results.txt");
    for (int len = 10; len <= 10000; len *= 10){
        for (int thread_num = 1; thread_num <= 30; thread_num++){
            vector<vector<int>> matrix(len, vector<int>(len, 0));
            for (int i = 0; i < len; i++){
                for(int j = 0; j <= i; j++){
                    matrix[i][j] = dist(gen);
                }
            }
            omp_set_num_threads(thread_num);
            auto start_time = chrono::high_resolution_clock::now();
            auto res = max_min_tri(matrix);
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            resFile << duration.count() << ' ';
        }
        resFile << endl;
    }
    resFile.close();
}

int max_min_band(vector<vector<int>> matrix, int k){
    auto res = 0;
    auto n = int(matrix.size());
    int first, last;
#pragma omp parallel for default(none) shared(matrix, res, k, n) private(first, last) schedule(dynamic)
    for(int i = 0; i < n; i++){
        first = (i - k) > 0 ? (i - k) : 0;
        last = (i + k) > (n - 1) ? (n - 1) : (i + k);
        auto row_min = matrix[i][first];
        for(int j = first + 1; j <= last; j++){
            if (matrix[i][j] < row_min){
                row_min = matrix[i][j];
            }
        }
        if (row_min > res){
#pragma omp critical
            if(row_min > res) {
                res = row_min;
            }
        }
    }
    return res;
}

void task_5_b(){
    cout << "Task 5, find minmax of band matrix, threads: 1, 2, ... , 30; len: 10, 100, 1000, 10 000; \n"
            " each row - same len, col - same thread_num, time in microseconds" << endl;
    ofstream resFile("results.txt");
    for (int len = 10; len <= 10000; len *= 10){
        auto k = static_cast<int>(round((1 - 3.0 / sqrt(10)) * len));
        for (int thread_num = 1; thread_num <= 30; thread_num++){
            vector<vector<int>> matrix(len, vector<int>(len, 0));
            omp_set_num_threads(thread_num);
            for(int i = 0; i < len; i++) {
                auto first = (i - k) > 0 ? (i - k) : 0;
                auto last = (i + k) > (len - 1) ? (len - 1) : (i + k);
                for(int j = first; j <= last; j++){
                    matrix[i][j] = dist(gen);
                }
            }
            omp_set_num_threads(thread_num);
            auto start_time = chrono::high_resolution_clock::now();
            auto res = max_min_band(matrix, k);
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            resFile << duration.count() << ' ';
        }
        resFile << endl;
    }
    resFile.close();
}

int reduction_sum(int n){
    int i = 0;
#pragma omp parallel for reduction(+: i) default(none) shared(n)
    for (auto j = 0; j < n; j++) {
        i += 1;
    }    return i;
}

int critical_sum(int n){
    int i = 0;
#pragma omp parallel for shared(i, n) default(none)
    for(auto j = 0; j < n; j++){
#pragma omp critical
        i += 1;
    }
    return i;
}

int atomic_sum(int n){
    int i = 0;
#pragma omp parallel for shared(i,n ) default(none)
    for(auto j = 0; j < n; j++){
#pragma omp atomic
        i += 1;
    }
    return i;
}

int lock_sum(int n){
    int i = 0;
    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel for shared(i, lock, n) default(none)
    for(auto j = 0; j < n; j++){
        omp_set_lock(&lock);
        i += 1;
        omp_unset_lock(&lock);
    }
    omp_destroy_lock(&lock);
    return i;
}

void task_6(int mode) {
    cout << "Task 6, different reduction realisations, threads: 1, 2, ... , 50; n: 10, 100,... , 100 000; \n"
            " each row - same len, col - same thread_num, time in microseconds" << endl;
    ofstream resFile("results.txt");
    for (int n = 1000; n <= 100000; n *= 10){
        for (int thread_num = 1; thread_num <= 50; thread_num++){
            omp_set_num_threads(thread_num);
            int res;
            auto start_time = chrono::high_resolution_clock::now();
            switch (mode) {
                case 0:
                    res = reduction_sum(n);
                    break;
                case 1:
                    res = critical_sum(n);
                    break;
                case 2:
                    res = atomic_sum(n);
                    break;
                case 3:
                    res = lock_sum(n);
                    break;
            }
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            resFile << duration.count() << ' ';
        }
        resFile << endl;
    }
    resFile.close();
}

void write_vectors(int vec_count, int vec_len){
    ofstream vecFile("vectors.txt");
    for (int i = 0; i < vec_count; i++){
        for (int j = 0; j < 2 * vec_len; j++){
            vecFile << dist(gen) << ' ';
        }
    }
    vecFile.close();
}

void task_7(){
    cout << "Task 7, sequential dot products; vec_count: 10, 1000, 10000; \n"
            " each row - same count, col - same thread_num, time in microseconds" << endl;
    ofstream resFile("results.txt");
    for (int vec_count = 10; vec_count <= 10000; vec_count *= 10){
        cout << "start generating for " << vec_count << endl;
        write_vectors(vec_count, 10000);
        cout << "finished" << endl;
        for (int thread_num = 10; thread_num <= 10; thread_num++) {
            auto start_time = chrono::high_resolution_clock::now();
            queue<vector<ULL>> vectors_l;
            queue<vector<ULL>> vectors_r;
            int ready = -1;
            omp_set_num_threads(2);
#pragma omp parallel sections default(none) shared(vec_count, ready, vectors_l, vectors_r, cout)
            {
#pragma omp section
                {
                    ifstream vecFile;
                    vecFile.open("vectors.txt");
                    if (vecFile.is_open()) {
                        int number;
                        for (int i = 0; i < vec_count; i += 1) {
                            vector<ULL> vec_l;
                            vector<ULL> vec_r;
                            for (int j = 0; j < 10000; j++){
                                ULL num;
                                vecFile >> num;
                                vec_l.push_back(num);
                            }
                            for (int j = 0; j < 10000; j++){
                                ULL num;
                                vecFile >> num;
                                vec_r.push_back(num);
                            }
                            vectors_l.push(vec_l);
                            vectors_r.push(vec_r);
#pragma omp atomic update release
                            ready += 1;
                        }
                        vecFile.close();
                    }
                    else
                        cout << "File cannot be opened" << endl;
                }
#pragma omp section
                {
                    int check = -1;
                    for (int i = 0; i < vec_count; i += 1){
                        while (check < i){
#pragma omp atomic read acquire
                            check = ready;
                        }
                        auto res = dot_product_np(vectors_l.front(), vectors_r.front());
                        vectors_r.pop();
                        vectors_l.pop();
                    }
                }
            }
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            cout << duration.count() << ' ';
        }
        cout << endl;
    }
    resFile.close();
}

void transpose(vector<vector<ULL>> matrix){
    auto n = matrix.size();
    for (int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            swap(matrix[i][j], matrix[j][i]);
        }
    }
}

vector<vector<ULL>> matrix_multiplication(const vector<vector<ULL>>& matrix_a, const vector<vector<ULL>>& matrix_b, int mode, int k){
    auto n = matrix_a.size();
    vector<vector<ULL>> res = vector<vector<ULL>>(n, vector<ULL>(n, 0));
#pragma omp parallel for shared(matrix_a, matrix_b, n, res, mode, k) default(none) if (mode >= 1) num_threads(k)
    for (int i = 0; i < n; i++){
        auto my_row = matrix_a[i];
#pragma omp parallel for shared(i, n, my_row, matrix_b, res, mode) default(none) if (mode >= 2) num_threads(k)
        for (int j = 0; j < n; j++){
            res[i][j] = dot_product_np(my_row, matrix_b[j]);
        }
    }
    return res;
}

void task_8(int mode) {
    cout << "Task 8, matrix multiplication, nested parallelism, threads: 1, 2, ... , 50; n: 100, 500; \n"
            " each row - same len, col - same thread_num, time in microseconds" << endl;
    ofstream resFile("results.txt");
    for (int n = 100; n <= 500; n *= 5){
        omp_set_nested(1);
        cout << "start preparing " << n << endl;
        vector<vector<ULL>> matrix_a = vector<vector<ULL>>(n, vector<ULL>(n, 0));
        vector<vector<ULL>> matrix_b = vector<vector<ULL>>(n, vector<ULL>(n, 0));
        for(int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                matrix_a[i][j] = dist(gen);
                matrix_b[i][j] = dist(gen);
            }
        }
        cout << "start calculating " << n << endl;
        for (int thread_num = 1; thread_num <= 50; thread_num++){
            auto start_time = chrono::high_resolution_clock::now();
            transpose(matrix_b);
            auto res = matrix_multiplication(matrix_a, matrix_b, mode, thread_num);
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            resFile << duration.count() << ' ';
        }
        resFile << endl;
    }
    resFile.close();
}

void no_grow(int n){
#pragma omp parallel for default(none) shared(n, dist_double, gen) schedule(SCHEDULE_MODE)
    for(int i = 0; i < n; i++){
        auto res = 0.0;
        for(int j = 0; j < n; j++){
            res += dist_double(gen);
        }
    }
}

void lin_grow(int n){
#pragma omp parallel for default(none) shared(n, dist_double, gen) schedule(SCHEDULE_MODE)
    for(int i = 0; i < n; i++){
        auto res = 0.0;
        for(int j = 0; j < i; j++){
            res += dist_double(gen);
        }
    }
}

void exp_grow(int n){
#pragma omp parallel for default(none) shared(n, dist_double, gen) schedule(SCHEDULE_MODE)
    for(int i = 0; i < n; i++){
        auto res = 0.0;
        for(int j = 0; j < static_cast<int>(pow(2, i)); j++){
            res += dist_double(gen);
        }
    }
}

void chaotic_load(int n){
#pragma omp parallel for default(none) shared(n, dist_double, dist_1k, gen) schedule(SCHEDULE_MODE)
    for(int i = 0; i < n; i++){
        auto res = 0.0;
        for(int j = 0; j < dist_1k(gen); j++){
            res += dist_double(gen);
        }
    }
}

void task_9(){
    cout << "Task 9, different types of load and scheduling, threads 1, 2, ... , 50; \n"
            " each row - same thread_num, time in microseconds" << endl;
    ofstream resFile("results.txt");
    cout << "start testing uniform load" << endl;
    for (int thread_num = 1; thread_num <= 30; thread_num++) {
        omp_set_num_threads(thread_num);
        auto start_time = chrono::high_resolution_clock::now();
        no_grow(1000);
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        resFile << duration.count() << ' ';
    }
    resFile << endl;
    cout << "start testing linear growing load" << endl;
    for (int thread_num = 1; thread_num <= 30; thread_num++) {
        omp_set_num_threads(thread_num);
        auto start_time = chrono::high_resolution_clock::now();
        lin_grow(1414);
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        resFile << duration.count() << ' ';
    }
    resFile << endl;
    cout << "start testing exponential growing load" << endl;
    for (int thread_num = 1; thread_num <= 30; thread_num++) {
        omp_set_num_threads(thread_num);
        auto start_time = chrono::high_resolution_clock::now();
        exp_grow(20);
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        resFile << duration.count() << ' ';
    }
    resFile << endl;
    cout << "start testing chaotic load" << endl;
    for (int thread_num = 1; thread_num <= 30; thread_num++) {
        omp_set_num_threads(thread_num);
        auto start_time = chrono::high_resolution_clock::now();
        chaotic_load(25500);
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        resFile << duration.count() << ' ';
    }
    resFile << endl;
    resFile.close();
}