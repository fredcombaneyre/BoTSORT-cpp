// Directly taken from: https://github.com/ifzhang/ByteTrack/blob/main/deploy/ncnn/cpp/include/lapjv.h

#ifndef LAPJV_H
#define LAPJV_H

#include <stdio.h> 

namespace bot_sort {

extern int lapjv_internal(const size_t n, double *cost[], int *x, int *y);

} // namespace botsort

#endif// LAPJV_H