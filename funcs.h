//
// Created by tosha on 26/10/2023.
//

#ifndef OMP_REPORT_FUNCS_H
#define OMP_REPORT_FUNCS_H

#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <queue>
#include <random>

#define REDUCTION_MODE 0
#define CRITICAL_MODE 1
#define ATOMIC_MODE 2
#define LOCK_MODE 3

#define SCHEDULE_MODE guided

using namespace std;

typedef unsigned long long ULL;

void task_1();

void task_2();

void task_3();

void task_4();

void task_5_a();

void task_5_b();

void task_6(int mode);

void task_7();

void task_8(int mode);

void task_9();


#endif //OMP_REPORT_FUNCS_H
