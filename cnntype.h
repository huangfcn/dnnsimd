#ifndef __CNN_TYPE_H__
#define __CNN_TYPE_H__

#define CNN_BCHSIZ  (8)
#define BCHSIZX4    (CNN_BCHSIZ * 4)

typedef float cnn_type_t;
typedef float acc_type_t;

#ifdef _WIN32
/* mingw */
#include <malloc.h>
#else
/* linux */
#include <stdlib.h>
#define _aligned_malloc(x, s) (aligned_alloc((s), (x)))
#define _aligned_free(x)      (free(x))
#endif

#if defined (_MSC_VER)
#define BEGIN_ALIGNED(x) __declspec(align(x))
#define DECLA_ALIGNED(x)

#define BEGIN_NOINLINE    __declspec(noinline)
#define DECLA_NOINLINE

#define __builtin_assume_aligned(v, x)    (v)
#define restrict __restrict
#endif

#if defined (__GNUC__)
#define BEGIN_ALIGNED(x) 
#define DECLA_ALIGNED(x) __attribute__ ((aligned(x)))

#define BEGIN_NOINLINE
#define DECLA_NOINLINE __attribute__ ((noinline))
#endif

/* padding mode */
#define PADDING_SAME   (1)
#define PADDING_VALID  (0)

#define tf_assert(cond) // (assert(cond))

#endif  // __CNN_TYPE_H__
