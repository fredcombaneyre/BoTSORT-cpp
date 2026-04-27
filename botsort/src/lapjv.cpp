// Directly taken from: https://github.com/ifzhang/ByteTrack/blob/main/deploy/ncnn/cpp/src/lapjv.cpp

#include "lapjv.h"

#include <stdio.h>

#include <stdlib.h>
#include <string.h>

#define LARGE 1000000

#if !defined TRUE
#define TRUE 1
#endif
#if !defined FALSE
#define FALSE 0
#endif

#define LAPJV_NEW(x, t, n)                                                           \
    if ((x = (t *) malloc(sizeof(t) * (n))) == 0) { return -1; }
#define LAPJV_FREE(x)                                                                \
    if (x != 0)                                                                \
    {                                                                          \
        free(x);                                                               \
        x = 0;                                                                 \
    }
#define LAPJV_SWAP_INDICES(a, b)                                                     \
    {                                                                          \
        int _temp_index = a;                                                   \
        a = b;                                                                 \
        b = _temp_index;                                                       \
    }

#if 0
#include <assert.h>
#define ASSERT(cond) assert(cond)
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define PRINT_COST_ARRAY(a, n)                                                 \
    while (1)                                                                  \
    {                                                                          \
        printf(#a " = [");                                                     \
        if ((n) > 0)                                                           \
        {                                                                      \
            printf("%f", (a)[0]);                                              \
            for (size_t j = 1; j < n; j++) { printf(", %f", (a)[j]); }         \
        }                                                                      \
        printf("]\n");                                                         \
        break;                                                                 \
    }
#define PRINT_INDEX_ARRAY(a, n)                                                \
    while (1)                                                                  \
    {                                                                          \
        printf(#a " = [");                                                     \
        if ((n) > 0)                                                           \
        {                                                                      \
            printf("%d", (a)[0]);                                              \
            for (size_t j = 1; j < n; j++) { printf(", %d", (a)[j]); }         \
        }                                                                      \
        printf("]\n");                                                         \
        break;                                                                 \
    }
#else
#define ASSERT(cond)
#define PRINTF(fmt, ...)
#define PRINT_COST_ARRAY(a, n)
#define PRINT_INDEX_ARRAY(a, n)
#endif


namespace
{
typedef enum fp_t
{
    FP_1 = 1,
    FP_2 = 2,
    FP_DYNAMIC = 3
} fp_t;


/** Column-reduction and reduction transfer for a dense cost matrix.
 */
int _ccrrt_dense(const size_t n, double *cost[], int *free_rows, int *x,
                   int *y, double *v)
{
    int n_free_rows;
    bool *unique;

    for (size_t i = 0; i < n; i++)
    {
        x[i] = -1;
        v[i] = LARGE;
        y[i] = 0;
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            const double c = cost[i][j];
            if (c < v[j])
            {
                v[j] = c;
                y[j] = i;
            }
            PRINTF("i=%d, j=%d, c[i,j]=%f, v[j]=%f y[j]=%d\n", i, j, c, v[j],
                   y[j]);
        }
    }
    PRINT_COST_ARRAY(v, n);
    PRINT_INDEX_ARRAY(y, n);
    LAPJV_NEW(unique, bool, n);
    memset(unique, TRUE, n);
    {
        int j = n;
        do {
            j--;
            const int i = y[j];
            if (x[i] < 0) { 
                x[i] = j; 
            }
            else
            {
                unique[i] = FALSE;
                y[j] = -1;
            }
        } while (j > 0);
    }
    n_free_rows = 0;
    for (size_t i = 0; i < n; i++)
    {
        if (x[i] < 0) { 
            free_rows[n_free_rows++] = i; 
        }
        else if (unique[i])
        {
            const int j = x[i];
            double min = LARGE;
            for (size_t j2 = 0; j2 < n; j2++)
            {
                if (j2 == (size_t) j) { 
                    continue; 
                }
                const double c = cost[i][j2] - v[j2];
                if (c < min) { 
                    min = c; 
                }
            }
            PRINTF("v[%d] = %f - %f\n", j, v[j], min);
            v[j] -= min;
        }
    }
    LAPJV_FREE(unique);
    return n_free_rows;
}


/** Augmenting row reduction for a dense cost matrix.
 */
int _carr_dense(const size_t n, double *cost[], const size_t n_free_rows,
                  int *free_rows, int *x, int *y, double *v)
{
    size_t current = 0;
    int new_free_rows = 0;
    size_t rr_cnt = 0;
    PRINT_INDEX_ARRAY(x, n);
    PRINT_INDEX_ARRAY(y, n);
    PRINT_COST_ARRAY(v, n);
    PRINT_INDEX_ARRAY(free_rows, n_free_rows);
    while (current < n_free_rows)
    {
        int i0;
        int j1, j2;
        double v1, v2, v1_new;
        bool v1_lowers;

        rr_cnt++;
        PRINTF("current = %d rr_cnt = %d\n", current, rr_cnt);
        const int free_i = free_rows[current++];
        j1 = 0;
        v1 = cost[free_i][0] - v[0];
        j2 = -1;
        v2 = LARGE;
        for (size_t j = 1; j < n; j++)
        {
            PRINTF("%d = %f %d = %f\n", j1, v1, j2, v2);
            const double c = cost[free_i][j] - v[j];
            if (c < v2)
            {
                if (c >= v1)
                {
                    v2 = c;
                    j2 = j;
                }
                else
                {
                    v2 = v1;
                    v1 = c;
                    j2 = j1;
                    j1 = j;
                }
            }
        }
        i0 = y[j1];
        v1_new = v[j1] - (v2 - v1);
        v1_lowers = v1_new < v[j1];
        PRINTF("%d %d 1=%d,%f 2=%d,%f v1'=%f(%d,%g) \n", free_i, i0, j1, v1, j2,
               v2, v1_new, v1_lowers, v[j1] - v1_new);
        if (rr_cnt < current * n)
        {
            if (v1_lowers) { 
                v[j1] = v1_new; 
            }
            else if (i0 >= 0 && j2 >= 0)
            {
                j1 = j2;
                i0 = y[j2];
            }
            if (i0 >= 0)
            {
                if (v1_lowers) { 
                    free_rows[--current] = i0; 
                }
                else { 
                    free_rows[new_free_rows++] = i0; 
                }
            }
        }
        else
        {
            PRINTF("rr_cnt=%d >= %d (current=%d * n=%d)\n", rr_cnt, current * n,
                   current, n);
            if (i0 >= 0) { 
                free_rows[new_free_rows++] = i0; 
            }
        }
        x[free_i] = j1;
        y[j1] = free_i;
    }
    return new_free_rows;
}


/** Find columns with minimum d[j] and put them on the SCAN list.
 */
size_t _find_dense(const size_t n, size_t lo, double *d, int *cols, int *y)
{
    size_t hi = lo + 1;
    double mind = d[cols[lo]];
    for (size_t k = hi; k < n; k++)
    {
        int j = cols[k];
        if (d[j] <= mind)
        {
            if (d[j] < mind)
            {
                hi = lo;
                mind = d[j];
            }
            cols[k] = cols[hi];
            cols[hi++] = j;
        }
    }
    return hi;
}


// Scan all columns in TODO starting from arbitrary column in SCAN
// and try to decrease d of the TODO columns using the SCAN column.
int _scan_dense(const size_t n, double *cost[], size_t *plo, size_t *phi,
                  double *d, int *cols, int *pred, int *y, double *v)
{
    size_t lo = *plo;
    size_t hi = *phi;
    double h, cred_ij;

    while (lo != hi)
    {
        int j = cols[lo++];
        const int i = y[j];
        const double mind = d[j];
        h = cost[i][j] - v[j] - mind;
        PRINTF("i=%d j=%d h=%f\n", i, j, h);
        // For all columns in TODO
        for (size_t k = hi; k < n; k++)
        {
            j = cols[k];
            cred_ij = cost[i][j] - v[j] - h;
            if (cred_ij < d[j])
            {
                d[j] = cred_ij;
                pred[j] = i;
                if (cred_ij == mind)
                {
                    if (y[j] < 0) { 
                        return j; 
                    }
                    cols[k] = cols[hi];
                    cols[hi++] = j;
                }
            }
        }
    }
    *plo = lo;
    *phi = hi;
    return -1;
}


/** Single iteration of modified Dijkstra shortest path algorithm as explained in the JV paper.
 *
 * This is a dense matrix version.
 *
 * \return The closest free column index.
 */
int find_path_dense(const size_t n, double *cost[], const int start_i,
                      int *y, double *v, int *pred)
{
    size_t lo = 0, hi = 0;
    int final_j = -1;
    size_t n_ready = 0;
    int *cols;
    double *d;

    LAPJV_NEW(cols, int, n);
    LAPJV_NEW(d, double, n);

    for (size_t i = 0; i < n; i++)
    {
        cols[i] = i;
        pred[i] = start_i;
        d[i] = cost[start_i][i] - v[i];
    }
    PRINT_COST_ARRAY(d, n);
    while (final_j == -1)
    {
        // No columns left on the SCAN list.
        if (lo == hi) {
            PRINTF("%d..%d -> find\n", lo, hi);

            n_ready = lo;
            hi = _find_dense(n, lo, d, cols, y);

            PRINTF("check %d..%d\n", lo, hi);
            PRINT_INDEX_ARRAY(cols, n);

            for (size_t k = lo; k < hi; k++){
                const int j = cols[k];
                if (y[j] < 0) { 
                    final_j = j; 
                }
            }
        }
        if (final_j == -1)
        {
            PRINTF("%d..%d -> scan\n", lo, hi);

            final_j = _scan_dense(n, cost, &lo, &hi, d, cols, pred, y, v);

            PRINT_COST_ARRAY(d, n);
            PRINT_INDEX_ARRAY(cols, n);
            PRINT_INDEX_ARRAY(pred, n);
        }
    }

    PRINTF("found final_j=%d\n", final_j);
    PRINT_INDEX_ARRAY(cols, n);

    {
        const double mind = d[cols[lo]];
        for (size_t k = 0; k < n_ready; k++)
        {
            const int j = cols[k];
            v[j] += d[j] - mind;
        }
    }

    LAPJV_FREE(cols);
    LAPJV_FREE(d);

    return final_j;
}


/** Augment for a dense cost matrix.
 */
int _ca_dense(const size_t n, double *cost[], const size_t n_free_rows,
                int *free_rows, int *x, int *y, double *v)
{
    int *pred;

    LAPJV_NEW(pred, int, n);

    for (int *pfree_i = free_rows; pfree_i < free_rows + n_free_rows;pfree_i++){
        int i = -1, j;
        size_t k = 0;

        PRINTF("looking at free_i=%d\n", *pfree_i);

        j = find_path_dense(n, cost, *pfree_i, y, v, pred);

        ASSERT(j >= 0);
        ASSERT(j < n);
        while (i != *pfree_i)
        {
            PRINTF("augment %d\n", j);
            PRINT_INDEX_ARRAY(pred, n);

            i = pred[j];

            PRINTF("y[%d]=%d -> %d\n", j, y[j], i);

            y[j] = i;

            PRINT_INDEX_ARRAY(x, n);
            LAPJV_SWAP_INDICES(j, x[i]);
            k++;
            if (k >= n) { 
                ASSERT(FALSE); 
            }
        }
    }
    LAPJV_FREE(pred);
    return 0;
}

} // namespace

/** Solve dense sparse LAP.
 */
int bot_sort::lapjv_internal(const size_t n, double *cost[], int *x, int *y)
{
    int ret;
    int *free_rows;
    double *v;

    LAPJV_NEW(free_rows, int, n);
    LAPJV_NEW(v, double, n);
    ret = _ccrrt_dense(n, cost, free_rows, x, y, v);
    int i = 0;
    while (ret > 0 && i < 2)
    {
        ret = _carr_dense(n, cost, ret, free_rows, x, y, v);
        i++;
    }
    if (ret > 0) { ret = _ca_dense(n, cost, ret, free_rows, x, y, v); }
    LAPJV_FREE(v);
    LAPJV_FREE(free_rows);
    return ret;
}

#undef LAPJV_NEW
#undef LAPJV_FREE
#undef LAPJV_SWAP_INDICES