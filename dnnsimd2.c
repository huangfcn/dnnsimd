#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <assert.h>

#include "cnntype.h"
#include "dnnsimd.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define dw_conv3x3_p1s1_bnrelu_core_08 \
{  \
   cnn_type_t y00[CNN_BCHSIZ];  \
   cnn_type_t y01[CNN_BCHSIZ];  \
   cnn_type_t y02[CNN_BCHSIZ];  \
   \
   cnn_type_t y10[CNN_BCHSIZ];  \
   cnn_type_t y11[CNN_BCHSIZ];  \
   cnn_type_t y12[CNN_BCHSIZ];  \
   \
   cnn_type_t y20[CNN_BCHSIZ];  \
   cnn_type_t y21[CNN_BCHSIZ];  \
   cnn_type_t y22[CNN_BCHSIZ];  \
   \
   cnn_type_t  yd[CNN_BCHSIZ];  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y00[p] = x00[p                ] * w00[p                ];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y01[p] = x00[p + __L1x        ] * w00[p + __L1x        ];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y02[p] = x00[p + __L2x        ] * w00[p + __L2x        ];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y10[p] = x00[p         + __L1n] * w00[p         + __L3x];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y11[p] = x00[p + __L1x + __L1n] * w00[p + __L1x + __L3x];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y12[p] = x00[p + __L2x + __L1n] * w00[p + __L2x + __L3x];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y20[p] = x00[p         + __L2n] * w00[p         + __L6x];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y21[p] = x00[p + __L1x + __L2n] * w00[p + __L1x + __L6x];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y22[p] = x00[p + __L2x + __L2n] * w00[p + __L2x + __L6x];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = (y00[p] + y01[p] + y02[p] + y10[p] + y11[p] + y12[p] + y20[p] + y21[p] + y22[p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = ( yd[p] - uloc[p]) * sloc[p] + bloc[p];                                          };  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {_py[p] = ( yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) :(yd[p]));                         };  \
   \
   \
   uloc += CNN_BCHSIZ; sloc += CNN_BCHSIZ; bloc += CNN_BCHSIZ;  \
   x00  += CNN_BCHSIZ;  \
   w00  += CNN_BCHSIZ;  \
   \
   _py  += CNN_BCHSIZ;  \
}

#define dw_conv3x3_p1s1_bnrelu_core_16 \
{  \
   dw_conv3x3_p1s1_bnrelu_core_08; \
   dw_conv3x3_p1s1_bnrelu_core_08; \
}

#define dw_conv3x3_p1s1_bnrelu_core_32 \
{  \
   dw_conv3x3_p1s1_bnrelu_core_16; \
   dw_conv3x3_p1s1_bnrelu_core_16; \
}

#define dw_conv3x3_p1s1_bnrelu_core_64 \
{  \
   dw_conv3x3_p1s1_bnrelu_core_32; \
   dw_conv3x3_p1s1_bnrelu_core_32; \
}

#define dw_conv3x3_p1s1_bnrelu_core_128 \
{  \
   dw_conv3x3_p1s1_bnrelu_core_64; \
   dw_conv3x3_p1s1_bnrelu_core_64; \
}

#define dw_conv3x3_p1s1_bnrelu_core_256 \
{  \
   dw_conv3x3_p1s1_bnrelu_core_128; \
   dw_conv3x3_p1s1_bnrelu_core_128; \
}

#define dw_conv3x3_p1s1_bnrelu_core_512 \
{  \
   dw_conv3x3_p1s1_bnrelu_core_256; \
   dw_conv3x3_p1s1_bnrelu_core_256; \
}

#define dw_conv3x3_p1s1_bnrelu_core_1024 \
{  \
   dw_conv3x3_p1s1_bnrelu_core_512; \
   dw_conv3x3_p1s1_bnrelu_core_512; \
}

#define dw_conv3x3_p1s1_bnrelu_core_pre \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \
   \
   cnn_type_t * wloc = _pw;  \
   cnn_type_t * uloc = _pu;  \
   cnn_type_t * sloc = _ps;  \
   cnn_type_t * bloc = _pb;  \
   \
   cnn_type_t * x00  = _px ;        \
   \
   cnn_type_t * w00  = wloc;        \
   \
   int __L2x = __Lx * 2;            \
   int __L3x = __Lx * 3;            \
   int __L6x = __Lx * 6;            \
   int __L2n = __Ln * 2;            \
   int __L1n = __Ln ;               \
   int __L1x = __Lx ;               \

static void _dw_conv3x3_p1s1_bnrelu_core_16(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_16;
}

static void _dw_conv3x3_p1s1_bnrelu_core_32(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_32;
}

static void _dw_conv3x3_p1s1_bnrelu_core_48(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_32;
   dw_conv3x3_p1s1_bnrelu_core_16;
}

static void _dw_conv3x3_p1s1_bnrelu_core_64(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_64;
}

static void _dw_conv3x3_p1s1_bnrelu_core_80(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_16;
   dw_conv3x3_p1s1_bnrelu_core_64;
}

static void _dw_conv3x3_p1s1_bnrelu_core_96(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_32;
   dw_conv3x3_p1s1_bnrelu_core_64;
}

static void _dw_conv3x3_p1s1_bnrelu_core_112(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_16;
   dw_conv3x3_p1s1_bnrelu_core_32;
   dw_conv3x3_p1s1_bnrelu_core_64;
}

static void _dw_conv3x3_p1s1_bnrelu_core_128(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_128;
}

static void _dw_conv3x3_p1s1_bnrelu_core_256(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_256;
}

static void _dw_conv3x3_p1s1_bnrelu_core_384(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_128;
   dw_conv3x3_p1s1_bnrelu_core_256;
}

static void _dw_conv3x3_p1s1_bnrelu_core_512(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_256;
   dw_conv3x3_p1s1_bnrelu_core_256;
}

static void _dw_conv3x3_p1s1_bnrelu_core_640(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_128;

   dw_conv3x3_p1s1_bnrelu_core_256;
   dw_conv3x3_p1s1_bnrelu_core_256;
}

static void _dw_conv3x3_p1s1_bnrelu_core_768(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_256;

   dw_conv3x3_p1s1_bnrelu_core_256;
   dw_conv3x3_p1s1_bnrelu_core_256;
}

static void _dw_conv3x3_p1s1_bnrelu_core_896(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_128;
   dw_conv3x3_p1s1_bnrelu_core_256;

   dw_conv3x3_p1s1_bnrelu_core_256;
   dw_conv3x3_p1s1_bnrelu_core_256;
}

static void _dw_conv3x3_p1s1_bnrelu_core_1024(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_1024;
}

/*
static void _dw_conv3x3_p1s1_bnrelu_core_2048(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_1024;
   dw_conv3x3_p1s1_bnrelu_core_1024;
}

static void _dw_conv3x3_p1s1_bnrelu_core_3072(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_core_pre;

   dw_conv3x3_p1s1_bnrelu_core_1024;
   dw_conv3x3_p1s1_bnrelu_core_1024;
   dw_conv3x3_p1s1_bnrelu_core_1024;
}
*/

static void _dw_conv3x3_p1s1_bnrelu_core_null(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
}

typedef void (*dw_conv3x3_p1s1_bnrelu_core_func)(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
);

static dw_conv3x3_p1s1_bnrelu_core_func _dw_conv3x3_p1s1_bnrelu_core_fptr_016[] = {
   _dw_conv3x3_p1s1_bnrelu_core_null,
   _dw_conv3x3_p1s1_bnrelu_core_16,
   _dw_conv3x3_p1s1_bnrelu_core_32,
   _dw_conv3x3_p1s1_bnrelu_core_48,
   _dw_conv3x3_p1s1_bnrelu_core_64,
   _dw_conv3x3_p1s1_bnrelu_core_80,
   _dw_conv3x3_p1s1_bnrelu_core_96,
   _dw_conv3x3_p1s1_bnrelu_core_112,
};

static dw_conv3x3_p1s1_bnrelu_core_func _dw_conv3x3_p1s1_bnrelu_core_fptr_128[] = {
   _dw_conv3x3_p1s1_bnrelu_core_null,
   _dw_conv3x3_p1s1_bnrelu_core_128,
   _dw_conv3x3_p1s1_bnrelu_core_256,
   _dw_conv3x3_p1s1_bnrelu_core_384,
   _dw_conv3x3_p1s1_bnrelu_core_512,
   _dw_conv3x3_p1s1_bnrelu_core_640,
   _dw_conv3x3_p1s1_bnrelu_core_768,
   _dw_conv3x3_p1s1_bnrelu_core_896,
   _dw_conv3x3_p1s1_bnrelu_core_1024,
};

/*
static dw_conv3x3_p1s1_bnrelu_core_func _dw_conv3x3_p1s1_bnrelu_core_fptr_xxx[] = {
   _dw_conv3x3_p1s1_bnrelu_core_null,
   _dw_conv3x3_p1s1_bnrelu_core_1024,
   _dw_conv3x3_p1s1_bnrelu_core_2048,
   _dw_conv3x3_p1s1_bnrelu_core_3072,
};
*/

void dw_conv3x3_p1s1_bnrelu_core(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int C1 = __Lx;

   int __i016 = (C1 >>  4) & 7;
   int __i128 = (C1 >>  7) & 7;
   int __ixxx = (C1 >> 10) ;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      _dw_conv3x3_p1s1_bnrelu_core_fptr_128[8](
         __Ln, __Lx, __Ly, 

         __px + (__i << 10), 
         __pw + (__i << 10),
         __pu + (__i << 10),
         __ps + (__i << 10), 
         __pb + (__i << 10),

         maxd, 

         __py + (__i << 10)
         );
   }

   _dw_conv3x3_p1s1_bnrelu_core_fptr_128[__i128](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10), 
      __pw + (__ixxx << 10),
      __pu + (__ixxx << 10),
      __ps + (__ixxx << 10), 
      __pb + (__ixxx << 10),

      maxd, 

      __py + (__ixxx << 10)
      );

   _dw_conv3x3_p1s1_bnrelu_core_fptr_016[__i016](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10) + (__i128 << 7), 
      __pw + (__ixxx << 10) + (__i128 << 7),
      __pu + (__ixxx << 10) + (__i128 << 7),
      __ps + (__ixxx << 10) + (__i128 << 7), 
      __pb + (__ixxx << 10) + (__i128 << 7),

      maxd, 

      __py + (__ixxx << 10) + (__i128 << 7)
      );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define dw_conv3x3_p1s1_bnrelu_rr_08 \
{  \
   cnn_type_t y00[CNN_BCHSIZ];  \
   cnn_type_t y01[CNN_BCHSIZ];  \
   \
   cnn_type_t y10[CNN_BCHSIZ];  \
   cnn_type_t y11[CNN_BCHSIZ];  \
   \
   cnn_type_t y20[CNN_BCHSIZ];  \
   cnn_type_t y21[CNN_BCHSIZ];  \
   \
   cnn_type_t yd[CNN_BCHSIZ];   \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y00[p] = x00[p] * w00[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y01[p] = x01[p] * w01[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y10[p] = x10[p] * w10[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y11[p] = x11[p] * w11[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y20[p] = x20[p] * w20[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y21[p] = x21[p] * w21[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = (y00[p] + y01[p] + y10[p] + y11[p] + y20[p] + y21[p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = ( yd[p] - uloc[p]) * sloc[p] + bloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {_py[p] = ( yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) :(yd[p]));}; \
   \
   \
   uloc += CNN_BCHSIZ; sloc += CNN_BCHSIZ; bloc += CNN_BCHSIZ;  \
   x00  += CNN_BCHSIZ;  x01 += CNN_BCHSIZ;  \
   x10  += CNN_BCHSIZ;  x11 += CNN_BCHSIZ;  \
   x20  += CNN_BCHSIZ;  x21 += CNN_BCHSIZ;  \
   w00  += CNN_BCHSIZ;  w01 += CNN_BCHSIZ;  \
   w10  += CNN_BCHSIZ;  w11 += CNN_BCHSIZ;  \
   w20  += CNN_BCHSIZ;  w21 += CNN_BCHSIZ;  \
   \
   _py  += CNN_BCHSIZ;  \
}

#define dw_conv3x3_p1s1_bnrelu_rr_16 \
{  \
   dw_conv3x3_p1s1_bnrelu_rr_08; \
   dw_conv3x3_p1s1_bnrelu_rr_08; \
}

#define dw_conv3x3_p1s1_bnrelu_rr_32 \
{  \
   dw_conv3x3_p1s1_bnrelu_rr_16; \
   dw_conv3x3_p1s1_bnrelu_rr_16; \
}

#define dw_conv3x3_p1s1_bnrelu_rr_64 \
{  \
   dw_conv3x3_p1s1_bnrelu_rr_32; \
   dw_conv3x3_p1s1_bnrelu_rr_32; \
}

#define dw_conv3x3_p1s1_bnrelu_rr_128 \
{  \
   dw_conv3x3_p1s1_bnrelu_rr_64; \
   dw_conv3x3_p1s1_bnrelu_rr_64; \
}

#define dw_conv3x3_p1s1_bnrelu_rr_256 \
{  \
   dw_conv3x3_p1s1_bnrelu_rr_128; \
   dw_conv3x3_p1s1_bnrelu_rr_128; \
}

#define dw_conv3x3_p1s1_bnrelu_rr_512 \
{  \
   dw_conv3x3_p1s1_bnrelu_rr_256; \
   dw_conv3x3_p1s1_bnrelu_rr_256; \
}

#define dw_conv3x3_p1s1_bnrelu_rr_1024 \
{  \
   dw_conv3x3_p1s1_bnrelu_rr_512; \
   dw_conv3x3_p1s1_bnrelu_rr_512; \
}

#define dw_conv3x3_p1s1_bnrelu_rr_pre \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \
   \
   cnn_type_t * wloc = _pw;  \
   cnn_type_t * uloc = _pu;  \
   cnn_type_t * sloc = _ps;  \
   cnn_type_t * bloc = _pb;  \
   \
   cnn_type_t * x00  = _px ;        \
   cnn_type_t * x01  = x00 + __Lx;  \
   \
   cnn_type_t * x10  = x00 + __Ln;  \
   cnn_type_t * x11  = x10 + __Lx;  \
   \
   cnn_type_t * x20  = x10 + __Ln;  \
   cnn_type_t * x21  = x20 + __Lx;  \
   \
   cnn_type_t * w00  = wloc;        \
   cnn_type_t * w01  = w00 + __Lx;  \
   cnn_type_t * w02  = w01 + __Lx;  \
   \
   cnn_type_t * w10  = w02 + __Lx;  \
   cnn_type_t * w11  = w10 + __Lx;  \
   cnn_type_t * w12  = w11 + __Lx;  \
   \
   cnn_type_t * w20  = w12 + __Lx;  \
   cnn_type_t * w21  = w20 + __Lx;  \

static void _dw_conv3x3_p1s1_bnrelu_rr_16(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_16;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_32(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_32;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_48(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_32;
   dw_conv3x3_p1s1_bnrelu_rr_16;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_64(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_64;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_80(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_16;
   dw_conv3x3_p1s1_bnrelu_rr_64;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_96(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_32;
   dw_conv3x3_p1s1_bnrelu_rr_64;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_112(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_16;
   dw_conv3x3_p1s1_bnrelu_rr_32;
   dw_conv3x3_p1s1_bnrelu_rr_64;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_128(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_128;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_256(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_384(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_128;
   dw_conv3x3_p1s1_bnrelu_rr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_512(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_256;
   dw_conv3x3_p1s1_bnrelu_rr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_640(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_128;

   dw_conv3x3_p1s1_bnrelu_rr_256;
   dw_conv3x3_p1s1_bnrelu_rr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_768(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_256;

   dw_conv3x3_p1s1_bnrelu_rr_256;
   dw_conv3x3_p1s1_bnrelu_rr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_896(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_128;
   dw_conv3x3_p1s1_bnrelu_rr_256;

   dw_conv3x3_p1s1_bnrelu_rr_256;
   dw_conv3x3_p1s1_bnrelu_rr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_1024(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_1024;
}

/*
static void _dw_conv3x3_p1s1_bnrelu_rr_2048(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_1024;
   dw_conv3x3_p1s1_bnrelu_rr_1024;
}

static void _dw_conv3x3_p1s1_bnrelu_rr_3072(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_rr_pre;

   dw_conv3x3_p1s1_bnrelu_rr_1024;
   dw_conv3x3_p1s1_bnrelu_rr_1024;
   dw_conv3x3_p1s1_bnrelu_rr_1024;
}
*/

static void _dw_conv3x3_p1s1_bnrelu_rr_null(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
}

typedef void (*dw_conv3x3_p1s1_bnrelu_rr_func)(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
);

static dw_conv3x3_p1s1_bnrelu_rr_func _dw_conv3x3_p1s1_bnrelu_rr_fptr_016[] = {
   _dw_conv3x3_p1s1_bnrelu_rr_null,
   _dw_conv3x3_p1s1_bnrelu_rr_16,
   _dw_conv3x3_p1s1_bnrelu_rr_32,
   _dw_conv3x3_p1s1_bnrelu_rr_48,
   _dw_conv3x3_p1s1_bnrelu_rr_64,
   _dw_conv3x3_p1s1_bnrelu_rr_80,
   _dw_conv3x3_p1s1_bnrelu_rr_96,
   _dw_conv3x3_p1s1_bnrelu_rr_112,
};

static dw_conv3x3_p1s1_bnrelu_rr_func _dw_conv3x3_p1s1_bnrelu_rr_fptr_128[] = {
   _dw_conv3x3_p1s1_bnrelu_rr_null,
   _dw_conv3x3_p1s1_bnrelu_rr_128,
   _dw_conv3x3_p1s1_bnrelu_rr_256,
   _dw_conv3x3_p1s1_bnrelu_rr_384,
   _dw_conv3x3_p1s1_bnrelu_rr_512,
   _dw_conv3x3_p1s1_bnrelu_rr_640,
   _dw_conv3x3_p1s1_bnrelu_rr_768,
   _dw_conv3x3_p1s1_bnrelu_rr_896,
   _dw_conv3x3_p1s1_bnrelu_rr_1024,
};

/*
static dw_conv3x3_p1s1_bnrelu_rr_func _dw_conv3x3_p1s1_bnrelu_rr_fptr_xxx[] = {
   _dw_conv3x3_p1s1_bnrelu_rr_null,
   _dw_conv3x3_p1s1_bnrelu_rr_1024,
   _dw_conv3x3_p1s1_bnrelu_rr_2048,
   _dw_conv3x3_p1s1_bnrelu_rr_3072,
};
*/

void dw_conv3x3_p1s1_bnrelu_rr(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int C1 = __Lx;

   int __i016 = (C1 >>  4) & 7;
   int __i128 = (C1 >>  7) & 7;
   int __ixxx = (C1 >> 10) ;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      _dw_conv3x3_p1s1_bnrelu_rr_fptr_128[8](
         __Ln, __Lx, __Ly, 

         __px + (__i << 10), 
         __pw + (__i << 10),
         __pu + (__i << 10),
         __ps + (__i << 10), 
         __pb + (__i << 10),

         maxd, 

         __py + (__i << 10)
         );
   }

   _dw_conv3x3_p1s1_bnrelu_rr_fptr_128[__i128](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10), 
      __pw + (__ixxx << 10),
      __pu + (__ixxx << 10),
      __ps + (__ixxx << 10), 
      __pb + (__ixxx << 10),

      maxd, 

      __py + (__ixxx << 10)
      );

   _dw_conv3x3_p1s1_bnrelu_rr_fptr_016[__i016](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10) + (__i128 << 7), 
      __pw + (__ixxx << 10) + (__i128 << 7),
      __pu + (__ixxx << 10) + (__i128 << 7),
      __ps + (__ixxx << 10) + (__i128 << 7), 
      __pb + (__ixxx << 10) + (__i128 << 7),

      maxd, 

      __py + (__ixxx << 10) + (__i128 << 7)
      );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define dw_conv3x3_p1s1_bnrelu_bb_08 \
{  \
   cnn_type_t y00[CNN_BCHSIZ];  \
   cnn_type_t y01[CNN_BCHSIZ];  \
   cnn_type_t y02[CNN_BCHSIZ];  \
   \
   cnn_type_t y10[CNN_BCHSIZ];  \
   cnn_type_t y11[CNN_BCHSIZ];  \
   cnn_type_t y12[CNN_BCHSIZ];  \
   \
   cnn_type_t yd[CNN_BCHSIZ];   \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y00[p] = x00[p] * w00[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y01[p] = x01[p] * w01[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y02[p] = x02[p] * w02[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y10[p] = x10[p] * w10[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y11[p] = x11[p] * w11[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y12[p] = x12[p] * w12[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = (y00[p] + y01[p] + y02[p] + y10[p] + y11[p] + y12[p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = ( yd[p] - uloc[p]) * sloc[p] + bloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {_py[p] = ( yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) :(yd[p]));}; \
   \
   \
   uloc += CNN_BCHSIZ; sloc += CNN_BCHSIZ; bloc += CNN_BCHSIZ;  \
   x00  += CNN_BCHSIZ;  x01 += CNN_BCHSIZ;  x02 += CNN_BCHSIZ;  \
   x10  += CNN_BCHSIZ;  x11 += CNN_BCHSIZ;  x12 += CNN_BCHSIZ;  \
   w00  += CNN_BCHSIZ;  w01 += CNN_BCHSIZ;  w02 += CNN_BCHSIZ;  \
   w10  += CNN_BCHSIZ;  w11 += CNN_BCHSIZ;  w12 += CNN_BCHSIZ;  \
   \
   _py  += CNN_BCHSIZ;  \
}

#define dw_conv3x3_p1s1_bnrelu_bb_16 \
{  \
   dw_conv3x3_p1s1_bnrelu_bb_08; \
   dw_conv3x3_p1s1_bnrelu_bb_08; \
}

#define dw_conv3x3_p1s1_bnrelu_bb_32 \
{  \
   dw_conv3x3_p1s1_bnrelu_bb_16; \
   dw_conv3x3_p1s1_bnrelu_bb_16; \
}

#define dw_conv3x3_p1s1_bnrelu_bb_64 \
{  \
   dw_conv3x3_p1s1_bnrelu_bb_32; \
   dw_conv3x3_p1s1_bnrelu_bb_32; \
}

#define dw_conv3x3_p1s1_bnrelu_bb_128 \
{  \
   dw_conv3x3_p1s1_bnrelu_bb_64; \
   dw_conv3x3_p1s1_bnrelu_bb_64; \
}

#define dw_conv3x3_p1s1_bnrelu_bb_256 \
{  \
   dw_conv3x3_p1s1_bnrelu_bb_128; \
   dw_conv3x3_p1s1_bnrelu_bb_128; \
}

#define dw_conv3x3_p1s1_bnrelu_bb_512 \
{  \
   dw_conv3x3_p1s1_bnrelu_bb_256; \
   dw_conv3x3_p1s1_bnrelu_bb_256; \
}

#define dw_conv3x3_p1s1_bnrelu_bb_1024 \
{  \
   dw_conv3x3_p1s1_bnrelu_bb_512; \
   dw_conv3x3_p1s1_bnrelu_bb_512; \
}

#define dw_conv3x3_p1s1_bnrelu_bb_pre \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \
   \
   cnn_type_t * wloc = _pw;  \
   cnn_type_t * uloc = _pu;  \
   cnn_type_t * sloc = _ps;  \
   cnn_type_t * bloc = _pb;  \
   \
   cnn_type_t * x00  = _px ;        \
   cnn_type_t * x01  = x00 + __Lx;  \
   cnn_type_t * x02  = x01 + __Lx;  \
   \
   cnn_type_t * x10  = x00 + __Ln;  \
   cnn_type_t * x11  = x10 + __Lx;  \
   cnn_type_t * x12  = x11 + __Lx;  \
   \
   cnn_type_t * w00  = wloc;        \
   cnn_type_t * w01  = w00 + __Lx;  \
   cnn_type_t * w02  = w01 + __Lx;  \
   \
   cnn_type_t * w10  = w02 + __Lx;  \
   cnn_type_t * w11  = w10 + __Lx;  \
   cnn_type_t * w12  = w11 + __Lx;  \

static void _dw_conv3x3_p1s1_bnrelu_bb_16(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_16;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_32(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_32;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_48(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_32;
   dw_conv3x3_p1s1_bnrelu_bb_16;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_64(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_64;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_80(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_16;
   dw_conv3x3_p1s1_bnrelu_bb_64;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_96(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_32;
   dw_conv3x3_p1s1_bnrelu_bb_64;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_112(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_16;
   dw_conv3x3_p1s1_bnrelu_bb_32;
   dw_conv3x3_p1s1_bnrelu_bb_64;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_128(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_128;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_256(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_384(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_128;
   dw_conv3x3_p1s1_bnrelu_bb_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_512(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_256;
   dw_conv3x3_p1s1_bnrelu_bb_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_640(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_128;

   dw_conv3x3_p1s1_bnrelu_bb_256;
   dw_conv3x3_p1s1_bnrelu_bb_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_768(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_256;

   dw_conv3x3_p1s1_bnrelu_bb_256;
   dw_conv3x3_p1s1_bnrelu_bb_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_896(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_128;
   dw_conv3x3_p1s1_bnrelu_bb_256;

   dw_conv3x3_p1s1_bnrelu_bb_256;
   dw_conv3x3_p1s1_bnrelu_bb_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_1024(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_1024;
}

/*
static void _dw_conv3x3_p1s1_bnrelu_bb_2048(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_1024;
   dw_conv3x3_p1s1_bnrelu_bb_1024;
}

static void _dw_conv3x3_p1s1_bnrelu_bb_3072(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bb_pre;

   dw_conv3x3_p1s1_bnrelu_bb_1024;
   dw_conv3x3_p1s1_bnrelu_bb_1024;
   dw_conv3x3_p1s1_bnrelu_bb_1024;
}
*/

static void _dw_conv3x3_p1s1_bnrelu_bb_null(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
}

typedef void (*dw_conv3x3_p1s1_bnrelu_bb_func)(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
);

static dw_conv3x3_p1s1_bnrelu_bb_func _dw_conv3x3_p1s1_bnrelu_bb_fptr_016[] = {
   _dw_conv3x3_p1s1_bnrelu_bb_null,
   _dw_conv3x3_p1s1_bnrelu_bb_16,
   _dw_conv3x3_p1s1_bnrelu_bb_32,
   _dw_conv3x3_p1s1_bnrelu_bb_48,
   _dw_conv3x3_p1s1_bnrelu_bb_64,
   _dw_conv3x3_p1s1_bnrelu_bb_80,
   _dw_conv3x3_p1s1_bnrelu_bb_96,
   _dw_conv3x3_p1s1_bnrelu_bb_112,
};

static dw_conv3x3_p1s1_bnrelu_bb_func _dw_conv3x3_p1s1_bnrelu_bb_fptr_128[] = {
   _dw_conv3x3_p1s1_bnrelu_bb_null,
   _dw_conv3x3_p1s1_bnrelu_bb_128,
   _dw_conv3x3_p1s1_bnrelu_bb_256,
   _dw_conv3x3_p1s1_bnrelu_bb_384,
   _dw_conv3x3_p1s1_bnrelu_bb_512,
   _dw_conv3x3_p1s1_bnrelu_bb_640,
   _dw_conv3x3_p1s1_bnrelu_bb_768,
   _dw_conv3x3_p1s1_bnrelu_bb_896,
   _dw_conv3x3_p1s1_bnrelu_bb_1024,
};

// static dw_conv3x3_p1s1_bnrelu_bb_func _dw_conv3x3_p1s1_bnrelu_bb_fptr_xxx[] = {
//    _dw_conv3x3_p1s1_bnrelu_bb_null,
//    _dw_conv3x3_p1s1_bnrelu_bb_1024,
//    _dw_conv3x3_p1s1_bnrelu_bb_2048,
//    _dw_conv3x3_p1s1_bnrelu_bb_3072,
// };

void dw_conv3x3_p1s1_bnrelu_bb(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int C1 = __Lx;

   int __i016 = (C1 >>  4) & 7;
   int __i128 = (C1 >>  7) & 7;
   int __ixxx = (C1 >> 10) ;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      _dw_conv3x3_p1s1_bnrelu_bb_fptr_128[8](
         __Ln, __Lx, __Ly, 

         __px + (__i << 10), 
         __pw + (__i << 10),
         __pu + (__i << 10),
         __ps + (__i << 10), 
         __pb + (__i << 10),

         maxd, 

         __py + (__i << 10)
         );
   }

   _dw_conv3x3_p1s1_bnrelu_bb_fptr_128[__i128](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10), 
      __pw + (__ixxx << 10),
      __pu + (__ixxx << 10),
      __ps + (__ixxx << 10), 
      __pb + (__ixxx << 10),

      maxd, 

      __py + (__ixxx << 10)
      );

   _dw_conv3x3_p1s1_bnrelu_bb_fptr_016[__i016](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10) + (__i128 << 7), 
      __pw + (__ixxx << 10) + (__i128 << 7),
      __pu + (__ixxx << 10) + (__i128 << 7),
      __ps + (__ixxx << 10) + (__i128 << 7), 
      __pb + (__ixxx << 10) + (__i128 << 7),

      maxd, 

      __py + (__ixxx << 10) + (__i128 << 7)
      );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define dw_conv3x3_p1s1_bnrelu_tt_08 \
{  \
   cnn_type_t y10[CNN_BCHSIZ];  \
   cnn_type_t y11[CNN_BCHSIZ];  \
   cnn_type_t y12[CNN_BCHSIZ];  \
   \
   cnn_type_t y20[CNN_BCHSIZ];  \
   cnn_type_t y21[CNN_BCHSIZ];  \
   cnn_type_t y22[CNN_BCHSIZ];  \
   \
   cnn_type_t yd[CNN_BCHSIZ];   \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y10[p] = x10[p] * w10[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y11[p] = x11[p] * w11[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y12[p] = x12[p] * w12[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y20[p] = x20[p] * w20[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y21[p] = x21[p] * w21[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y22[p] = x22[p] * w22[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = (y10[p] + y11[p] + y12[p] + y20[p] + y21[p] + y22[p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = ( yd[p] - uloc[p]) * sloc[p] + bloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {_py[p] = ( yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) :(yd[p]));}; \
   \
   \
   uloc += CNN_BCHSIZ; sloc += CNN_BCHSIZ; bloc += CNN_BCHSIZ;  \
   x10  += CNN_BCHSIZ;  x11 += CNN_BCHSIZ;  x12 += CNN_BCHSIZ;  \
   x20  += CNN_BCHSIZ;  x21 += CNN_BCHSIZ;  x22 += CNN_BCHSIZ;  \
   w10  += CNN_BCHSIZ;  w11 += CNN_BCHSIZ;  w12 += CNN_BCHSIZ;  \
   w20  += CNN_BCHSIZ;  w21 += CNN_BCHSIZ;  w22 += CNN_BCHSIZ;  \
   \
   _py  += CNN_BCHSIZ;  \
}

#define dw_conv3x3_p1s1_bnrelu_tt_16 \
{  \
   dw_conv3x3_p1s1_bnrelu_tt_08; \
   dw_conv3x3_p1s1_bnrelu_tt_08; \
}

#define dw_conv3x3_p1s1_bnrelu_tt_32 \
{  \
   dw_conv3x3_p1s1_bnrelu_tt_16; \
   dw_conv3x3_p1s1_bnrelu_tt_16; \
}

#define dw_conv3x3_p1s1_bnrelu_tt_64 \
{  \
   dw_conv3x3_p1s1_bnrelu_tt_32; \
   dw_conv3x3_p1s1_bnrelu_tt_32; \
}

#define dw_conv3x3_p1s1_bnrelu_tt_128 \
{  \
   dw_conv3x3_p1s1_bnrelu_tt_64; \
   dw_conv3x3_p1s1_bnrelu_tt_64; \
}

#define dw_conv3x3_p1s1_bnrelu_tt_256 \
{  \
   dw_conv3x3_p1s1_bnrelu_tt_128; \
   dw_conv3x3_p1s1_bnrelu_tt_128; \
}

#define dw_conv3x3_p1s1_bnrelu_tt_512 \
{  \
   dw_conv3x3_p1s1_bnrelu_tt_256; \
   dw_conv3x3_p1s1_bnrelu_tt_256; \
}

#define dw_conv3x3_p1s1_bnrelu_tt_1024 \
{  \
   dw_conv3x3_p1s1_bnrelu_tt_512; \
   dw_conv3x3_p1s1_bnrelu_tt_512; \
}

#define dw_conv3x3_p1s1_bnrelu_tt_pre \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \
   \
   cnn_type_t * wloc = _pw;  \
   cnn_type_t * uloc = _pu;  \
   cnn_type_t * sloc = _ps;  \
   cnn_type_t * bloc = _pb;  \
   \
   cnn_type_t * x00  = _px ;        \
   \
   cnn_type_t * x10  = x00 + __Ln;  \
   cnn_type_t * x11  = x10 + __Lx;  \
   cnn_type_t * x12  = x11 + __Lx;  \
   \
   cnn_type_t * x20  = x10 + __Ln;  \
   cnn_type_t * x21  = x20 + __Lx;  \
   cnn_type_t * x22  = x21 + __Lx;  \
   \
   cnn_type_t * w00  = wloc;        \
   cnn_type_t * w01  = w00 + __Lx;  \
   cnn_type_t * w02  = w01 + __Lx;  \
   \
   cnn_type_t * w10  = w02 + __Lx;  \
   cnn_type_t * w11  = w10 + __Lx;  \
   cnn_type_t * w12  = w11 + __Lx;  \
   \
   cnn_type_t * w20  = w12 + __Lx;  \
   cnn_type_t * w21  = w20 + __Lx;  \
   cnn_type_t * w22  = w21 + __Lx;  \

static void _dw_conv3x3_p1s1_bnrelu_tt_16(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_16;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_32(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_32;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_48(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_32;
   dw_conv3x3_p1s1_bnrelu_tt_16;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_64(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_80(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_16;
   dw_conv3x3_p1s1_bnrelu_tt_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_96(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_32;
   dw_conv3x3_p1s1_bnrelu_tt_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_112(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_16;
   dw_conv3x3_p1s1_bnrelu_tt_32;
   dw_conv3x3_p1s1_bnrelu_tt_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_128(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_128;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_256(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_384(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_128;
   dw_conv3x3_p1s1_bnrelu_tt_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_512(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_256;
   dw_conv3x3_p1s1_bnrelu_tt_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_640(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_128;

   dw_conv3x3_p1s1_bnrelu_tt_256;
   dw_conv3x3_p1s1_bnrelu_tt_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_768(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_256;

   dw_conv3x3_p1s1_bnrelu_tt_256;
   dw_conv3x3_p1s1_bnrelu_tt_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_896(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_128;
   dw_conv3x3_p1s1_bnrelu_tt_256;

   dw_conv3x3_p1s1_bnrelu_tt_256;
   dw_conv3x3_p1s1_bnrelu_tt_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_1024(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_1024;
}

/*
static void _dw_conv3x3_p1s1_bnrelu_tt_2048(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_1024;
   dw_conv3x3_p1s1_bnrelu_tt_1024;
}

static void _dw_conv3x3_p1s1_bnrelu_tt_3072(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tt_pre;

   dw_conv3x3_p1s1_bnrelu_tt_1024;
   dw_conv3x3_p1s1_bnrelu_tt_1024;
   dw_conv3x3_p1s1_bnrelu_tt_1024;
}
*/

static void _dw_conv3x3_p1s1_bnrelu_tt_null(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
}

typedef void (*dw_conv3x3_p1s1_bnrelu_tt_func)(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
);

static dw_conv3x3_p1s1_bnrelu_tt_func _dw_conv3x3_p1s1_bnrelu_tt_fptr_016[] = {
   _dw_conv3x3_p1s1_bnrelu_tt_null,
   _dw_conv3x3_p1s1_bnrelu_tt_16,
   _dw_conv3x3_p1s1_bnrelu_tt_32,
   _dw_conv3x3_p1s1_bnrelu_tt_48,
   _dw_conv3x3_p1s1_bnrelu_tt_64,
   _dw_conv3x3_p1s1_bnrelu_tt_80,
   _dw_conv3x3_p1s1_bnrelu_tt_96,
   _dw_conv3x3_p1s1_bnrelu_tt_112,
};

static dw_conv3x3_p1s1_bnrelu_tt_func _dw_conv3x3_p1s1_bnrelu_tt_fptr_128[] = {
   _dw_conv3x3_p1s1_bnrelu_tt_null,
   _dw_conv3x3_p1s1_bnrelu_tt_128,
   _dw_conv3x3_p1s1_bnrelu_tt_256,
   _dw_conv3x3_p1s1_bnrelu_tt_384,
   _dw_conv3x3_p1s1_bnrelu_tt_512,
   _dw_conv3x3_p1s1_bnrelu_tt_640,
   _dw_conv3x3_p1s1_bnrelu_tt_768,
   _dw_conv3x3_p1s1_bnrelu_tt_896,
   _dw_conv3x3_p1s1_bnrelu_tt_1024,
};

// static dw_conv3x3_p1s1_bnrelu_tt_func _dw_conv3x3_p1s1_bnrelu_tt_fptr_xxx[] = {
//    _dw_conv3x3_p1s1_bnrelu_tt_null,
//    _dw_conv3x3_p1s1_bnrelu_tt_1024,
//    _dw_conv3x3_p1s1_bnrelu_tt_2048,
//    _dw_conv3x3_p1s1_bnrelu_tt_3072,
// };

void dw_conv3x3_p1s1_bnrelu_tt(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int C1 = __Lx;

   int __i016 = (C1 >>  4) & 7;
   int __i128 = (C1 >>  7) & 7;
   int __ixxx = (C1 >> 10) ;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      _dw_conv3x3_p1s1_bnrelu_tt_fptr_128[8](
         __Ln, __Lx, __Ly, 

         __px + (__i << 10), 
         __pw + (__i << 10),
         __pu + (__i << 10),
         __ps + (__i << 10), 
         __pb + (__i << 10),

         maxd, 

         __py + (__i << 10)
         );
   }

   _dw_conv3x3_p1s1_bnrelu_tt_fptr_128[__i128](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10), 
      __pw + (__ixxx << 10),
      __pu + (__ixxx << 10),
      __ps + (__ixxx << 10), 
      __pb + (__ixxx << 10),

      maxd, 

      __py + (__ixxx << 10)
      );

   _dw_conv3x3_p1s1_bnrelu_tt_fptr_016[__i016](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10) + (__i128 << 7), 
      __pw + (__ixxx << 10) + (__i128 << 7),
      __pu + (__ixxx << 10) + (__i128 << 7),
      __ps + (__ixxx << 10) + (__i128 << 7), 
      __pb + (__ixxx << 10) + (__i128 << 7),

      maxd, 

      __py + (__ixxx << 10) + (__i128 << 7)
      );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define dw_conv3x3_p1s1_bnrelu_ll_08 \
{  \
   cnn_type_t y01[CNN_BCHSIZ];  \
   cnn_type_t y02[CNN_BCHSIZ];  \
   \
   cnn_type_t y11[CNN_BCHSIZ];  \
   cnn_type_t y12[CNN_BCHSIZ];  \
   \
   cnn_type_t y21[CNN_BCHSIZ];  \
   cnn_type_t y22[CNN_BCHSIZ];  \
   \
   cnn_type_t yd[CNN_BCHSIZ];   \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y01[p] = x01[p                ] * w01[p        ];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y02[p] = x01[p + __L1x        ] * w01[p + __L1x];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y11[p] = x01[p         + __L1n] * w01[p + __L3x];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y12[p] = x01[p + __L1x + __L1n] * w01[p + __L4x];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y21[p] = x01[p         + __L2n] * w01[p + __L6x];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y22[p] = x01[p + __L1x + __L2n] * w01[p + __L7x];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = (y01[p] + y02[p] + y11[p] + y12[p] + y21[p] + y22[p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = ( yd[p] - uloc[p]) * sloc[p] + bloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {_py[p] = ( yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) :(yd[p]));}; \
   \
   \
   uloc += CNN_BCHSIZ; sloc += CNN_BCHSIZ; bloc += CNN_BCHSIZ;  \
   x01 += CNN_BCHSIZ;  \
   w01 += CNN_BCHSIZ;  \
   \
   _py  += CNN_BCHSIZ;  \
}

#define dw_conv3x3_p1s1_bnrelu_ll_16 \
{  \
   dw_conv3x3_p1s1_bnrelu_ll_08; \
   dw_conv3x3_p1s1_bnrelu_ll_08; \
}

#define dw_conv3x3_p1s1_bnrelu_ll_32 \
{  \
   dw_conv3x3_p1s1_bnrelu_ll_16; \
   dw_conv3x3_p1s1_bnrelu_ll_16; \
}

#define dw_conv3x3_p1s1_bnrelu_ll_64 \
{  \
   dw_conv3x3_p1s1_bnrelu_ll_32; \
   dw_conv3x3_p1s1_bnrelu_ll_32; \
}

#define dw_conv3x3_p1s1_bnrelu_ll_128 \
{  \
   dw_conv3x3_p1s1_bnrelu_ll_64; \
   dw_conv3x3_p1s1_bnrelu_ll_64; \
}

#define dw_conv3x3_p1s1_bnrelu_ll_256 \
{  \
   dw_conv3x3_p1s1_bnrelu_ll_128; \
   dw_conv3x3_p1s1_bnrelu_ll_128; \
}

#define dw_conv3x3_p1s1_bnrelu_ll_512 \
{  \
   dw_conv3x3_p1s1_bnrelu_ll_256; \
   dw_conv3x3_p1s1_bnrelu_ll_256; \
}

#define dw_conv3x3_p1s1_bnrelu_ll_1024 \
{  \
   dw_conv3x3_p1s1_bnrelu_ll_512; \
   dw_conv3x3_p1s1_bnrelu_ll_512; \
}

#define dw_conv3x3_p1s1_bnrelu_ll_pre \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \
   \
   cnn_type_t * wloc = _pw;  \
   cnn_type_t * uloc = _pu;  \
   cnn_type_t * sloc = _ps;  \
   cnn_type_t * bloc = _pb;  \
   \
   cnn_type_t * x00  = _px ;        \
   cnn_type_t * x01  = x00 + __Lx;  \
   \
   cnn_type_t * w00  = wloc;        \
   cnn_type_t * w01  = w00 + __Lx;  \
   \
   int __L1x = __Lx ;               \
   int __L3x = __Lx * 3;            \
   int __L4x = __Lx * 4;            \
   int __L6x = __Lx * 6;            \
   int __L7x = __Lx * 7;            \
   \
   int __L1n = __Ln ;               \
   int __L2n = __Ln * 2;            \

static void _dw_conv3x3_p1s1_bnrelu_ll_16(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_16;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_32(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_32;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_48(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_32;
   dw_conv3x3_p1s1_bnrelu_ll_16;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_64(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_64;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_80(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_16;
   dw_conv3x3_p1s1_bnrelu_ll_64;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_96(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_32;
   dw_conv3x3_p1s1_bnrelu_ll_64;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_112(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_16;
   dw_conv3x3_p1s1_bnrelu_ll_32;
   dw_conv3x3_p1s1_bnrelu_ll_64;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_128(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_128;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_256(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_256;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_384(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_128;
   dw_conv3x3_p1s1_bnrelu_ll_256;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_512(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_256;
   dw_conv3x3_p1s1_bnrelu_ll_256;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_640(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_128;

   dw_conv3x3_p1s1_bnrelu_ll_256;
   dw_conv3x3_p1s1_bnrelu_ll_256;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_768(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_256;

   dw_conv3x3_p1s1_bnrelu_ll_256;
   dw_conv3x3_p1s1_bnrelu_ll_256;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_896(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_128;
   dw_conv3x3_p1s1_bnrelu_ll_256;

   dw_conv3x3_p1s1_bnrelu_ll_256;
   dw_conv3x3_p1s1_bnrelu_ll_256;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_1024(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_1024;
}

/*
static void _dw_conv3x3_p1s1_bnrelu_ll_2048(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_1024;
   dw_conv3x3_p1s1_bnrelu_ll_1024;
}

static void _dw_conv3x3_p1s1_bnrelu_ll_3072(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_ll_pre;

   dw_conv3x3_p1s1_bnrelu_ll_1024;
   dw_conv3x3_p1s1_bnrelu_ll_1024;
   dw_conv3x3_p1s1_bnrelu_ll_1024;
}
*/

static void _dw_conv3x3_p1s1_bnrelu_ll_null(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
}

typedef void (*dw_conv3x3_p1s1_bnrelu_ll_func)(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
);

static dw_conv3x3_p1s1_bnrelu_ll_func _dw_conv3x3_p1s1_bnrelu_ll_fptr_016[] = {
   _dw_conv3x3_p1s1_bnrelu_ll_null,
   _dw_conv3x3_p1s1_bnrelu_ll_16,
   _dw_conv3x3_p1s1_bnrelu_ll_32,
   _dw_conv3x3_p1s1_bnrelu_ll_48,
   _dw_conv3x3_p1s1_bnrelu_ll_64,
   _dw_conv3x3_p1s1_bnrelu_ll_80,
   _dw_conv3x3_p1s1_bnrelu_ll_96,
   _dw_conv3x3_p1s1_bnrelu_ll_112,
};

static dw_conv3x3_p1s1_bnrelu_ll_func _dw_conv3x3_p1s1_bnrelu_ll_fptr_128[] = {
   _dw_conv3x3_p1s1_bnrelu_ll_null,
   _dw_conv3x3_p1s1_bnrelu_ll_128,
   _dw_conv3x3_p1s1_bnrelu_ll_256,
   _dw_conv3x3_p1s1_bnrelu_ll_384,
   _dw_conv3x3_p1s1_bnrelu_ll_512,
   _dw_conv3x3_p1s1_bnrelu_ll_640,
   _dw_conv3x3_p1s1_bnrelu_ll_768,
   _dw_conv3x3_p1s1_bnrelu_ll_896,
   _dw_conv3x3_p1s1_bnrelu_ll_1024,
};

// static dw_conv3x3_p1s1_bnrelu_ll_func _dw_conv3x3_p1s1_bnrelu_ll_fptr_xxx[] = {
//    _dw_conv3x3_p1s1_bnrelu_ll_null,
//    _dw_conv3x3_p1s1_bnrelu_ll_1024,
//    _dw_conv3x3_p1s1_bnrelu_ll_2048,
//    _dw_conv3x3_p1s1_bnrelu_ll_3072,
// };

void dw_conv3x3_p1s1_bnrelu_ll(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int C1 = __Lx;

   int __i016 = (C1 >>  4) & 7;
   int __i128 = (C1 >>  7) & 7;
   int __ixxx = (C1 >> 10) ;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      _dw_conv3x3_p1s1_bnrelu_ll_fptr_128[8](
         __Ln, __Lx, __Ly, 

         __px + (__i << 10), 
         __pw + (__i << 10),
         __pu + (__i << 10),
         __ps + (__i << 10), 
         __pb + (__i << 10),

         maxd, 

         __py + (__i << 10)
         );
   }

   _dw_conv3x3_p1s1_bnrelu_ll_fptr_128[__i128](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10), 
      __pw + (__ixxx << 10),
      __pu + (__ixxx << 10),
      __ps + (__ixxx << 10), 
      __pb + (__ixxx << 10),

      maxd, 

      __py + (__ixxx << 10)
      );

   _dw_conv3x3_p1s1_bnrelu_ll_fptr_016[__i016](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10) + (__i128 << 7), 
      __pw + (__ixxx << 10) + (__i128 << 7),
      __pu + (__ixxx << 10) + (__i128 << 7),
      __ps + (__ixxx << 10) + (__i128 << 7), 
      __pb + (__ixxx << 10) + (__i128 << 7),

      maxd, 

      __py + (__ixxx << 10) + (__i128 << 7)
      );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define dw_conv3x3_p1s1_bnrelu_br_08 \
{  \
   cnn_type_t y00[CNN_BCHSIZ];  \
   cnn_type_t y01[CNN_BCHSIZ];  \
   \
   cnn_type_t y10[CNN_BCHSIZ];  \
   cnn_type_t y11[CNN_BCHSIZ];  \
   \
   cnn_type_t yd[CNN_BCHSIZ];   \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y00[p] = x00[p] * w00[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y01[p] = x01[p] * w01[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y10[p] = x10[p] * w10[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y11[p] = x11[p] * w11[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = (y00[p] + y01[p] + y10[p] + y11[p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = ( yd[p] - uloc[p]) * sloc[p] + bloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {_py[p] = ( yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) :(yd[p]));}; \
   \
   \
   uloc += CNN_BCHSIZ; sloc += CNN_BCHSIZ; bloc += CNN_BCHSIZ;  \
   x00  += CNN_BCHSIZ;  x01 += CNN_BCHSIZ;  \
   x10  += CNN_BCHSIZ;  x11 += CNN_BCHSIZ;  \
   w00  += CNN_BCHSIZ;  w01 += CNN_BCHSIZ;  \
   w10  += CNN_BCHSIZ;  w11 += CNN_BCHSIZ;  \
   \
   _py  += CNN_BCHSIZ;  \
}

#define dw_conv3x3_p1s1_bnrelu_br_16 \
{  \
   dw_conv3x3_p1s1_bnrelu_br_08; \
   dw_conv3x3_p1s1_bnrelu_br_08; \
}

#define dw_conv3x3_p1s1_bnrelu_br_32 \
{  \
   dw_conv3x3_p1s1_bnrelu_br_16; \
   dw_conv3x3_p1s1_bnrelu_br_16; \
}

#define dw_conv3x3_p1s1_bnrelu_br_64 \
{  \
   dw_conv3x3_p1s1_bnrelu_br_32; \
   dw_conv3x3_p1s1_bnrelu_br_32; \
}

#define dw_conv3x3_p1s1_bnrelu_br_128 \
{  \
   dw_conv3x3_p1s1_bnrelu_br_64; \
   dw_conv3x3_p1s1_bnrelu_br_64; \
}

#define dw_conv3x3_p1s1_bnrelu_br_256 \
{  \
   dw_conv3x3_p1s1_bnrelu_br_128; \
   dw_conv3x3_p1s1_bnrelu_br_128; \
}

#define dw_conv3x3_p1s1_bnrelu_br_512 \
{  \
   dw_conv3x3_p1s1_bnrelu_br_256; \
   dw_conv3x3_p1s1_bnrelu_br_256; \
}

#define dw_conv3x3_p1s1_bnrelu_br_1024 \
{  \
   dw_conv3x3_p1s1_bnrelu_br_512; \
   dw_conv3x3_p1s1_bnrelu_br_512; \
}

#define dw_conv3x3_p1s1_bnrelu_br_pre \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \
   \
   cnn_type_t * wloc = _pw;  \
   cnn_type_t * uloc = _pu;  \
   cnn_type_t * sloc = _ps;  \
   cnn_type_t * bloc = _pb;  \
   \
   cnn_type_t * x00  = _px ;        \
   cnn_type_t * x01  = x00 + __Lx;  \
   \
   cnn_type_t * x10  = x00 + __Ln;  \
   cnn_type_t * x11  = x10 + __Lx;  \
   \
   cnn_type_t * w00  = wloc;        \
   cnn_type_t * w01  = w00 + __Lx;  \
   cnn_type_t * w02  = w01 + __Lx;  \
   \
   cnn_type_t * w10  = w02 + __Lx;  \
   cnn_type_t * w11  = w10 + __Lx;  \


static void _dw_conv3x3_p1s1_bnrelu_br_16(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_16;
}

static void _dw_conv3x3_p1s1_bnrelu_br_32(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_32;
}

static void _dw_conv3x3_p1s1_bnrelu_br_48(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_32;
   dw_conv3x3_p1s1_bnrelu_br_16;
}

static void _dw_conv3x3_p1s1_bnrelu_br_64(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_64;
}

static void _dw_conv3x3_p1s1_bnrelu_br_80(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_16;
   dw_conv3x3_p1s1_bnrelu_br_64;
}

static void _dw_conv3x3_p1s1_bnrelu_br_96(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_32;
   dw_conv3x3_p1s1_bnrelu_br_64;
}

static void _dw_conv3x3_p1s1_bnrelu_br_112(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_16;
   dw_conv3x3_p1s1_bnrelu_br_32;
   dw_conv3x3_p1s1_bnrelu_br_64;
}

static void _dw_conv3x3_p1s1_bnrelu_br_128(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_128;
}

static void _dw_conv3x3_p1s1_bnrelu_br_256(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_256;
}

static void _dw_conv3x3_p1s1_bnrelu_br_384(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_128;
   dw_conv3x3_p1s1_bnrelu_br_256;
}

static void _dw_conv3x3_p1s1_bnrelu_br_512(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_256;
   dw_conv3x3_p1s1_bnrelu_br_256;
}

static void _dw_conv3x3_p1s1_bnrelu_br_640(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_128;

   dw_conv3x3_p1s1_bnrelu_br_256;
   dw_conv3x3_p1s1_bnrelu_br_256;
}

static void _dw_conv3x3_p1s1_bnrelu_br_768(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_256;

   dw_conv3x3_p1s1_bnrelu_br_256;
   dw_conv3x3_p1s1_bnrelu_br_256;
}

static void _dw_conv3x3_p1s1_bnrelu_br_896(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_128;
   dw_conv3x3_p1s1_bnrelu_br_256;

   dw_conv3x3_p1s1_bnrelu_br_256;
   dw_conv3x3_p1s1_bnrelu_br_256;
}

static void _dw_conv3x3_p1s1_bnrelu_br_1024(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_1024;
}

/*
static void _dw_conv3x3_p1s1_bnrelu_br_2048(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_1024;
   dw_conv3x3_p1s1_bnrelu_br_1024;
}

static void _dw_conv3x3_p1s1_bnrelu_br_3072(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_br_pre;

   dw_conv3x3_p1s1_bnrelu_br_1024;
   dw_conv3x3_p1s1_bnrelu_br_1024;
   dw_conv3x3_p1s1_bnrelu_br_1024;
}
*/

static void _dw_conv3x3_p1s1_bnrelu_br_null(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
}

typedef void (*dw_conv3x3_p1s1_bnrelu_br_func)(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
);

static dw_conv3x3_p1s1_bnrelu_br_func _dw_conv3x3_p1s1_bnrelu_br_fptr_016[] = {
   _dw_conv3x3_p1s1_bnrelu_br_null,
   _dw_conv3x3_p1s1_bnrelu_br_16,
   _dw_conv3x3_p1s1_bnrelu_br_32,
   _dw_conv3x3_p1s1_bnrelu_br_48,
   _dw_conv3x3_p1s1_bnrelu_br_64,
   _dw_conv3x3_p1s1_bnrelu_br_80,
   _dw_conv3x3_p1s1_bnrelu_br_96,
   _dw_conv3x3_p1s1_bnrelu_br_112,
};

static dw_conv3x3_p1s1_bnrelu_br_func _dw_conv3x3_p1s1_bnrelu_br_fptr_128[] = {
   _dw_conv3x3_p1s1_bnrelu_br_null,
   _dw_conv3x3_p1s1_bnrelu_br_128,
   _dw_conv3x3_p1s1_bnrelu_br_256,
   _dw_conv3x3_p1s1_bnrelu_br_384,
   _dw_conv3x3_p1s1_bnrelu_br_512,
   _dw_conv3x3_p1s1_bnrelu_br_640,
   _dw_conv3x3_p1s1_bnrelu_br_768,
   _dw_conv3x3_p1s1_bnrelu_br_896,
   _dw_conv3x3_p1s1_bnrelu_br_1024,
};

// static dw_conv3x3_p1s1_bnrelu_br_func _dw_conv3x3_p1s1_bnrelu_br_fptr_xxx[] = {
//    _dw_conv3x3_p1s1_bnrelu_br_null,
//    _dw_conv3x3_p1s1_bnrelu_br_1024,
//    _dw_conv3x3_p1s1_bnrelu_br_2048,
//    _dw_conv3x3_p1s1_bnrelu_br_3072,
// };

void dw_conv3x3_p1s1_bnrelu_br(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int C1 = __Lx;

   int __i016 = (C1 >>  4) & 7;
   int __i128 = (C1 >>  7) & 7;
   int __ixxx = (C1 >> 10) ;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      _dw_conv3x3_p1s1_bnrelu_br_fptr_128[8](
         __Ln, __Lx, __Ly, 

         __px + (__i << 10), 
         __pw + (__i << 10),
         __pu + (__i << 10),
         __ps + (__i << 10), 
         __pb + (__i << 10),

         maxd, 

         __py + (__i << 10)
         );
   }

   _dw_conv3x3_p1s1_bnrelu_br_fptr_128[__i128](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10), 
      __pw + (__ixxx << 10),
      __pu + (__ixxx << 10),
      __ps + (__ixxx << 10), 
      __pb + (__ixxx << 10),

      maxd, 

      __py + (__ixxx << 10)
      );

   _dw_conv3x3_p1s1_bnrelu_br_fptr_016[__i016](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10) + (__i128 << 7), 
      __pw + (__ixxx << 10) + (__i128 << 7),
      __pu + (__ixxx << 10) + (__i128 << 7),
      __ps + (__ixxx << 10) + (__i128 << 7), 
      __pb + (__ixxx << 10) + (__i128 << 7),

      maxd, 

      __py + (__ixxx << 10) + (__i128 << 7)
      );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define dw_conv3x3_p1s1_bnrelu_bl_08 \
{  \
   cnn_type_t y01[CNN_BCHSIZ];  \
   cnn_type_t y02[CNN_BCHSIZ];  \
   \
   cnn_type_t y11[CNN_BCHSIZ];  \
   cnn_type_t y12[CNN_BCHSIZ];  \
   \
   cnn_type_t yd[CNN_BCHSIZ];   \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y01[p] = x01[p] * w01[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y02[p] = x02[p] * w02[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y11[p] = x11[p] * w11[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y12[p] = x12[p] * w12[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = (y01[p] + y02[p] + y11[p] + y12[p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = ( yd[p] - uloc[p]) * sloc[p] + bloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {_py[p] = ( yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) :(yd[p]));}; \
   \
   \
   uloc += CNN_BCHSIZ; sloc += CNN_BCHSIZ; bloc += CNN_BCHSIZ;  \
   x01 += CNN_BCHSIZ;  x02 += CNN_BCHSIZ;  \
   x11 += CNN_BCHSIZ;  x12 += CNN_BCHSIZ;  \
   w01 += CNN_BCHSIZ;  w02 += CNN_BCHSIZ;  \
   w11 += CNN_BCHSIZ;  w12 += CNN_BCHSIZ;  \
   \
   _py  += CNN_BCHSIZ;  \
}

#define dw_conv3x3_p1s1_bnrelu_bl_16 \
{  \
   dw_conv3x3_p1s1_bnrelu_bl_08; \
   dw_conv3x3_p1s1_bnrelu_bl_08; \
}

#define dw_conv3x3_p1s1_bnrelu_bl_32 \
{  \
   dw_conv3x3_p1s1_bnrelu_bl_16; \
   dw_conv3x3_p1s1_bnrelu_bl_16; \
}

#define dw_conv3x3_p1s1_bnrelu_bl_64 \
{  \
   dw_conv3x3_p1s1_bnrelu_bl_32; \
   dw_conv3x3_p1s1_bnrelu_bl_32; \
}

#define dw_conv3x3_p1s1_bnrelu_bl_128 \
{  \
   dw_conv3x3_p1s1_bnrelu_bl_64; \
   dw_conv3x3_p1s1_bnrelu_bl_64; \
}

#define dw_conv3x3_p1s1_bnrelu_bl_256 \
{  \
   dw_conv3x3_p1s1_bnrelu_bl_128; \
   dw_conv3x3_p1s1_bnrelu_bl_128; \
}

#define dw_conv3x3_p1s1_bnrelu_bl_512 \
{  \
   dw_conv3x3_p1s1_bnrelu_bl_256; \
   dw_conv3x3_p1s1_bnrelu_bl_256; \
}

#define dw_conv3x3_p1s1_bnrelu_bl_1024 \
{  \
   dw_conv3x3_p1s1_bnrelu_bl_512; \
   dw_conv3x3_p1s1_bnrelu_bl_512; \
}

#define dw_conv3x3_p1s1_bnrelu_bl_pre \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \
   \
   cnn_type_t * wloc = _pw;  \
   cnn_type_t * uloc = _pu;  \
   cnn_type_t * sloc = _ps;  \
   cnn_type_t * bloc = _pb;  \
   \
   cnn_type_t * x00  = _px ;        \
   cnn_type_t * x01  = x00 + __Lx;  \
   cnn_type_t * x02  = x01 + __Lx;  \
   \
   cnn_type_t * x10  = x00 + __Ln;  \
   cnn_type_t * x11  = x10 + __Lx;  \
   cnn_type_t * x12  = x11 + __Lx;  \
   \
   cnn_type_t * w00  = wloc;        \
   cnn_type_t * w01  = w00 + __Lx;  \
   cnn_type_t * w02  = w01 + __Lx;  \
   \
   cnn_type_t * w10  = w02 + __Lx;  \
   cnn_type_t * w11  = w10 + __Lx;  \
   cnn_type_t * w12  = w11 + __Lx;  \

static void _dw_conv3x3_p1s1_bnrelu_bl_16(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_16;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_32(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_32;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_48(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_32;
   dw_conv3x3_p1s1_bnrelu_bl_16;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_64(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_64;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_80(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_16;
   dw_conv3x3_p1s1_bnrelu_bl_64;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_96(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_32;
   dw_conv3x3_p1s1_bnrelu_bl_64;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_112(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_16;
   dw_conv3x3_p1s1_bnrelu_bl_32;
   dw_conv3x3_p1s1_bnrelu_bl_64;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_128(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_128;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_256(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_384(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_128;
   dw_conv3x3_p1s1_bnrelu_bl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_512(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_256;
   dw_conv3x3_p1s1_bnrelu_bl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_640(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_128;

   dw_conv3x3_p1s1_bnrelu_bl_256;
   dw_conv3x3_p1s1_bnrelu_bl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_768(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_256;

   dw_conv3x3_p1s1_bnrelu_bl_256;
   dw_conv3x3_p1s1_bnrelu_bl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_896(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_128;
   dw_conv3x3_p1s1_bnrelu_bl_256;

   dw_conv3x3_p1s1_bnrelu_bl_256;
   dw_conv3x3_p1s1_bnrelu_bl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_1024(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_1024;
}

/*
static void _dw_conv3x3_p1s1_bnrelu_bl_2048(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_1024;
   dw_conv3x3_p1s1_bnrelu_bl_1024;
}

static void _dw_conv3x3_p1s1_bnrelu_bl_3072(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_bl_pre;

   dw_conv3x3_p1s1_bnrelu_bl_1024;
   dw_conv3x3_p1s1_bnrelu_bl_1024;
   dw_conv3x3_p1s1_bnrelu_bl_1024;
}
*/

static void _dw_conv3x3_p1s1_bnrelu_bl_null(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
}

typedef void (*dw_conv3x3_p1s1_bnrelu_bl_func)(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
);

static dw_conv3x3_p1s1_bnrelu_bl_func _dw_conv3x3_p1s1_bnrelu_bl_fptr_016[] = {
   _dw_conv3x3_p1s1_bnrelu_bl_null,
   _dw_conv3x3_p1s1_bnrelu_bl_16,
   _dw_conv3x3_p1s1_bnrelu_bl_32,
   _dw_conv3x3_p1s1_bnrelu_bl_48,
   _dw_conv3x3_p1s1_bnrelu_bl_64,
   _dw_conv3x3_p1s1_bnrelu_bl_80,
   _dw_conv3x3_p1s1_bnrelu_bl_96,
   _dw_conv3x3_p1s1_bnrelu_bl_112,
};

static dw_conv3x3_p1s1_bnrelu_bl_func _dw_conv3x3_p1s1_bnrelu_bl_fptr_128[] = {
   _dw_conv3x3_p1s1_bnrelu_bl_null,
   _dw_conv3x3_p1s1_bnrelu_bl_128,
   _dw_conv3x3_p1s1_bnrelu_bl_256,
   _dw_conv3x3_p1s1_bnrelu_bl_384,
   _dw_conv3x3_p1s1_bnrelu_bl_512,
   _dw_conv3x3_p1s1_bnrelu_bl_640,
   _dw_conv3x3_p1s1_bnrelu_bl_768,
   _dw_conv3x3_p1s1_bnrelu_bl_896,
   _dw_conv3x3_p1s1_bnrelu_bl_1024,
};

// static dw_conv3x3_p1s1_bnrelu_bl_func _dw_conv3x3_p1s1_bnrelu_bl_fptr_xxx[] = {
//    _dw_conv3x3_p1s1_bnrelu_bl_null,
//    _dw_conv3x3_p1s1_bnrelu_bl_1024,
//    _dw_conv3x3_p1s1_bnrelu_bl_2048,
//    _dw_conv3x3_p1s1_bnrelu_bl_3072,
// };

void dw_conv3x3_p1s1_bnrelu_bl(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int C1 = __Lx;

   int __i016 = (C1 >>  4) & 7;
   int __i128 = (C1 >>  7) & 7;
   int __ixxx = (C1 >> 10) ;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      _dw_conv3x3_p1s1_bnrelu_bl_fptr_128[8](
         __Ln, __Lx, __Ly, 

         __px + (__i << 10), 
         __pw + (__i << 10),
         __pu + (__i << 10),
         __ps + (__i << 10), 
         __pb + (__i << 10),

         maxd, 

         __py + (__i << 10)
         );
   }

   _dw_conv3x3_p1s1_bnrelu_bl_fptr_128[__i128](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10), 
      __pw + (__ixxx << 10),
      __pu + (__ixxx << 10),
      __ps + (__ixxx << 10), 
      __pb + (__ixxx << 10),

      maxd, 

      __py + (__ixxx << 10)
      );

   _dw_conv3x3_p1s1_bnrelu_bl_fptr_016[__i016](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10) + (__i128 << 7), 
      __pw + (__ixxx << 10) + (__i128 << 7),
      __pu + (__ixxx << 10) + (__i128 << 7),
      __ps + (__ixxx << 10) + (__i128 << 7), 
      __pb + (__ixxx << 10) + (__i128 << 7),

      maxd, 

      __py + (__ixxx << 10) + (__i128 << 7)
      );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define dw_conv3x3_p1s1_bnrelu_tl_08 \
{  \
   cnn_type_t y11[CNN_BCHSIZ];  \
   cnn_type_t y12[CNN_BCHSIZ];  \
   \
   cnn_type_t y21[CNN_BCHSIZ];  \
   cnn_type_t y22[CNN_BCHSIZ];  \
   \
   cnn_type_t yd[CNN_BCHSIZ];   \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y11[p] = x11[p] * w11[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y12[p] = x12[p] * w12[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y21[p] = x21[p] * w21[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y22[p] = x22[p] * w22[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = (y11[p] + y12[p] +y21[p] + y22[p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = ( yd[p] - uloc[p]) * sloc[p] + bloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {_py[p] = ( yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) :(yd[p]));}; \
   \
   \
   uloc += CNN_BCHSIZ; sloc += CNN_BCHSIZ; bloc += CNN_BCHSIZ;  \
   x11 += CNN_BCHSIZ;  x12 += CNN_BCHSIZ;  \
   x21 += CNN_BCHSIZ;  x22 += CNN_BCHSIZ;  \
   w11 += CNN_BCHSIZ;  w12 += CNN_BCHSIZ;  \
   w21 += CNN_BCHSIZ;  w22 += CNN_BCHSIZ;  \
   \
   _py  += CNN_BCHSIZ;  \
}

#define dw_conv3x3_p1s1_bnrelu_tl_16 \
{  \
   dw_conv3x3_p1s1_bnrelu_tl_08; \
   dw_conv3x3_p1s1_bnrelu_tl_08; \
}

#define dw_conv3x3_p1s1_bnrelu_tl_32 \
{  \
   dw_conv3x3_p1s1_bnrelu_tl_16; \
   dw_conv3x3_p1s1_bnrelu_tl_16; \
}

#define dw_conv3x3_p1s1_bnrelu_tl_64 \
{  \
   dw_conv3x3_p1s1_bnrelu_tl_32; \
   dw_conv3x3_p1s1_bnrelu_tl_32; \
}

#define dw_conv3x3_p1s1_bnrelu_tl_128 \
{  \
   dw_conv3x3_p1s1_bnrelu_tl_64; \
   dw_conv3x3_p1s1_bnrelu_tl_64; \
}

#define dw_conv3x3_p1s1_bnrelu_tl_256 \
{  \
   dw_conv3x3_p1s1_bnrelu_tl_128; \
   dw_conv3x3_p1s1_bnrelu_tl_128; \
}

#define dw_conv3x3_p1s1_bnrelu_tl_512 \
{  \
   dw_conv3x3_p1s1_bnrelu_tl_256; \
   dw_conv3x3_p1s1_bnrelu_tl_256; \
}

#define dw_conv3x3_p1s1_bnrelu_tl_1024 \
{  \
   dw_conv3x3_p1s1_bnrelu_tl_512; \
   dw_conv3x3_p1s1_bnrelu_tl_512; \
}

#define dw_conv3x3_p1s1_bnrelu_tl_pre \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \
   \
   cnn_type_t * wloc = _pw;  \
   cnn_type_t * uloc = _pu;  \
   cnn_type_t * sloc = _ps;  \
   cnn_type_t * bloc = _pb;  \
   \
   cnn_type_t * x00  = _px ;        \
   \
   cnn_type_t * x10  = x00 + __Ln;  \
   cnn_type_t * x11  = x10 + __Lx;  \
   cnn_type_t * x12  = x11 + __Lx;  \
   \
   cnn_type_t * x20  = x10 + __Ln;  \
   cnn_type_t * x21  = x20 + __Lx;  \
   cnn_type_t * x22  = x21 + __Lx;  \
   \
   cnn_type_t * w00  = wloc;        \
   cnn_type_t * w01  = w00 + __Lx;  \
   cnn_type_t * w02  = w01 + __Lx;  \
   \
   cnn_type_t * w10  = w02 + __Lx;  \
   cnn_type_t * w11  = w10 + __Lx;  \
   cnn_type_t * w12  = w11 + __Lx;  \
   \
   cnn_type_t * w20  = w12 + __Lx;  \
   cnn_type_t * w21  = w20 + __Lx;  \
   cnn_type_t * w22  = w21 + __Lx;  \

static void _dw_conv3x3_p1s1_bnrelu_tl_16(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_16;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_32(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_32;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_48(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_32;
   dw_conv3x3_p1s1_bnrelu_tl_16;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_64(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_80(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_16;
   dw_conv3x3_p1s1_bnrelu_tl_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_96(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_32;
   dw_conv3x3_p1s1_bnrelu_tl_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_112(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_16;
   dw_conv3x3_p1s1_bnrelu_tl_32;
   dw_conv3x3_p1s1_bnrelu_tl_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_128(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_128;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_256(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_384(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_128;
   dw_conv3x3_p1s1_bnrelu_tl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_512(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_256;
   dw_conv3x3_p1s1_bnrelu_tl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_640(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_128;

   dw_conv3x3_p1s1_bnrelu_tl_256;
   dw_conv3x3_p1s1_bnrelu_tl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_768(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_256;

   dw_conv3x3_p1s1_bnrelu_tl_256;
   dw_conv3x3_p1s1_bnrelu_tl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_896(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_128;
   dw_conv3x3_p1s1_bnrelu_tl_256;

   dw_conv3x3_p1s1_bnrelu_tl_256;
   dw_conv3x3_p1s1_bnrelu_tl_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_1024(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_1024;
}

/*
static void _dw_conv3x3_p1s1_bnrelu_tl_2048(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_1024;
   dw_conv3x3_p1s1_bnrelu_tl_1024;
}

static void _dw_conv3x3_p1s1_bnrelu_tl_3072(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tl_pre;

   dw_conv3x3_p1s1_bnrelu_tl_1024;
   dw_conv3x3_p1s1_bnrelu_tl_1024;
   dw_conv3x3_p1s1_bnrelu_tl_1024;
}
*/

static void _dw_conv3x3_p1s1_bnrelu_tl_null(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
}

typedef void (*dw_conv3x3_p1s1_bnrelu_tl_func)(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
);

static dw_conv3x3_p1s1_bnrelu_tl_func _dw_conv3x3_p1s1_bnrelu_tl_fptr_016[] = {
   _dw_conv3x3_p1s1_bnrelu_tl_null,
   _dw_conv3x3_p1s1_bnrelu_tl_16,
   _dw_conv3x3_p1s1_bnrelu_tl_32,
   _dw_conv3x3_p1s1_bnrelu_tl_48,
   _dw_conv3x3_p1s1_bnrelu_tl_64,
   _dw_conv3x3_p1s1_bnrelu_tl_80,
   _dw_conv3x3_p1s1_bnrelu_tl_96,
   _dw_conv3x3_p1s1_bnrelu_tl_112,
};

static dw_conv3x3_p1s1_bnrelu_tl_func _dw_conv3x3_p1s1_bnrelu_tl_fptr_128[] = {
   _dw_conv3x3_p1s1_bnrelu_tl_null,
   _dw_conv3x3_p1s1_bnrelu_tl_128,
   _dw_conv3x3_p1s1_bnrelu_tl_256,
   _dw_conv3x3_p1s1_bnrelu_tl_384,
   _dw_conv3x3_p1s1_bnrelu_tl_512,
   _dw_conv3x3_p1s1_bnrelu_tl_640,
   _dw_conv3x3_p1s1_bnrelu_tl_768,
   _dw_conv3x3_p1s1_bnrelu_tl_896,
   _dw_conv3x3_p1s1_bnrelu_tl_1024,
};

// static dw_conv3x3_p1s1_bnrelu_tl_func _dw_conv3x3_p1s1_bnrelu_tl_fptr_xxx[] = {
//    _dw_conv3x3_p1s1_bnrelu_tl_null,
//    _dw_conv3x3_p1s1_bnrelu_tl_1024,
//    _dw_conv3x3_p1s1_bnrelu_tl_2048,
//    _dw_conv3x3_p1s1_bnrelu_tl_3072,
// };

void dw_conv3x3_p1s1_bnrelu_tl(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int C1 = __Lx;

   int __i016 = (C1 >>  4) & 7;
   int __i128 = (C1 >>  7) & 7;
   int __ixxx = (C1 >> 10) ;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      _dw_conv3x3_p1s1_bnrelu_tl_fptr_128[8](
         __Ln, __Lx, __Ly, 

         __px + (__i << 10), 
         __pw + (__i << 10),
         __pu + (__i << 10),
         __ps + (__i << 10), 
         __pb + (__i << 10),

         maxd, 

         __py + (__i << 10)
         );
   }

   _dw_conv3x3_p1s1_bnrelu_tl_fptr_128[__i128](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10), 
      __pw + (__ixxx << 10),
      __pu + (__ixxx << 10),
      __ps + (__ixxx << 10), 
      __pb + (__ixxx << 10),

      maxd, 

      __py + (__ixxx << 10)
      );

   _dw_conv3x3_p1s1_bnrelu_tl_fptr_016[__i016](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10) + (__i128 << 7), 
      __pw + (__ixxx << 10) + (__i128 << 7),
      __pu + (__ixxx << 10) + (__i128 << 7),
      __ps + (__ixxx << 10) + (__i128 << 7), 
      __pb + (__ixxx << 10) + (__i128 << 7),

      maxd, 

      __py + (__ixxx << 10) + (__i128 << 7)
      );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define dw_conv3x3_p1s1_bnrelu_tr_08 \
{  \
   cnn_type_t y10[CNN_BCHSIZ];  \
   cnn_type_t y11[CNN_BCHSIZ];  \
   \
   cnn_type_t y20[CNN_BCHSIZ];  \
   cnn_type_t y21[CNN_BCHSIZ];  \
   \
   cnn_type_t yd[CNN_BCHSIZ];   \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y10[p] = x10[p] * w10[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y11[p] = x11[p] * w11[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y20[p] = x20[p] * w20[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {y21[p] = x21[p] * w21[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = (y10[p] + y11[p] + y20[p] + y21[p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) { yd[p] = ( yd[p] - uloc[p]) * sloc[p] + bloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++) {_py[p] = ( yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) :(yd[p]));}; \
   \
   \
   uloc += CNN_BCHSIZ; sloc += CNN_BCHSIZ; bloc += CNN_BCHSIZ;  \
   x10  += CNN_BCHSIZ;  x11 += CNN_BCHSIZ;  \
   x20  += CNN_BCHSIZ;  x21 += CNN_BCHSIZ;  \
   w10  += CNN_BCHSIZ;  w11 += CNN_BCHSIZ;  \
   w20  += CNN_BCHSIZ;  w21 += CNN_BCHSIZ;  \
   \
   _py  += CNN_BCHSIZ;  \
}

#define dw_conv3x3_p1s1_bnrelu_tr_16 \
{  \
   dw_conv3x3_p1s1_bnrelu_tr_08; \
   dw_conv3x3_p1s1_bnrelu_tr_08; \
}

#define dw_conv3x3_p1s1_bnrelu_tr_32 \
{  \
   dw_conv3x3_p1s1_bnrelu_tr_16; \
   dw_conv3x3_p1s1_bnrelu_tr_16; \
}

#define dw_conv3x3_p1s1_bnrelu_tr_64 \
{  \
   dw_conv3x3_p1s1_bnrelu_tr_32; \
   dw_conv3x3_p1s1_bnrelu_tr_32; \
}

#define dw_conv3x3_p1s1_bnrelu_tr_128 \
{  \
   dw_conv3x3_p1s1_bnrelu_tr_64; \
   dw_conv3x3_p1s1_bnrelu_tr_64; \
}

#define dw_conv3x3_p1s1_bnrelu_tr_256 \
{  \
   dw_conv3x3_p1s1_bnrelu_tr_128; \
   dw_conv3x3_p1s1_bnrelu_tr_128; \
}

#define dw_conv3x3_p1s1_bnrelu_tr_512 \
{  \
   dw_conv3x3_p1s1_bnrelu_tr_256; \
   dw_conv3x3_p1s1_bnrelu_tr_256; \
}

#define dw_conv3x3_p1s1_bnrelu_tr_1024 \
{  \
   dw_conv3x3_p1s1_bnrelu_tr_512; \
   dw_conv3x3_p1s1_bnrelu_tr_512; \
}

#define dw_conv3x3_p1s1_bnrelu_tr_pre \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \
   \
   cnn_type_t * wloc = _pw;  \
   cnn_type_t * uloc = _pu;  \
   cnn_type_t * sloc = _ps;  \
   cnn_type_t * bloc = _pb;  \
   \
   cnn_type_t * x00  = _px ;        \
   \
   cnn_type_t * x10  = x00 + __Ln;  \
   cnn_type_t * x11  = x10 + __Lx;  \
   \
   \
   cnn_type_t * x20  = x10 + __Ln;  \
   cnn_type_t * x21  = x20 + __Lx;  \
   \
   \
   cnn_type_t * w00  = wloc;        \
   cnn_type_t * w01  = w00 + __Lx;  \
   cnn_type_t * w02  = w01 + __Lx;  \
   \
   cnn_type_t * w10  = w02 + __Lx;  \
   cnn_type_t * w11  = w10 + __Lx;  \
   cnn_type_t * w12  = w11 + __Lx;  \
   \
   cnn_type_t * w20  = w12 + __Lx;  \
   cnn_type_t * w21  = w20 + __Lx;  \


static void _dw_conv3x3_p1s1_bnrelu_tr_16(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_16;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_32(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_32;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_48(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_32;
   dw_conv3x3_p1s1_bnrelu_tr_16;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_64(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_80(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_16;
   dw_conv3x3_p1s1_bnrelu_tr_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_96(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_32;
   dw_conv3x3_p1s1_bnrelu_tr_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_112(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_16;
   dw_conv3x3_p1s1_bnrelu_tr_32;
   dw_conv3x3_p1s1_bnrelu_tr_64;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_128(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_128;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_256(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_384(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_128;
   dw_conv3x3_p1s1_bnrelu_tr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_512(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_256;
   dw_conv3x3_p1s1_bnrelu_tr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_640(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_128;

   dw_conv3x3_p1s1_bnrelu_tr_256;
   dw_conv3x3_p1s1_bnrelu_tr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_768(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_256;

   dw_conv3x3_p1s1_bnrelu_tr_256;
   dw_conv3x3_p1s1_bnrelu_tr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_896(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_128;
   dw_conv3x3_p1s1_bnrelu_tr_256;

   dw_conv3x3_p1s1_bnrelu_tr_256;
   dw_conv3x3_p1s1_bnrelu_tr_256;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_1024(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_1024;
}

/*
static void _dw_conv3x3_p1s1_bnrelu_tr_2048(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_1024;
   dw_conv3x3_p1s1_bnrelu_tr_1024;
}

static void _dw_conv3x3_p1s1_bnrelu_tr_3072(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   dw_conv3x3_p1s1_bnrelu_tr_pre;

   dw_conv3x3_p1s1_bnrelu_tr_1024;
   dw_conv3x3_p1s1_bnrelu_tr_1024;
   dw_conv3x3_p1s1_bnrelu_tr_1024;
}
*/

static void _dw_conv3x3_p1s1_bnrelu_tr_null(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
}

typedef void (*dw_conv3x3_p1s1_bnrelu_tr_func)(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
);

static dw_conv3x3_p1s1_bnrelu_tr_func _dw_conv3x3_p1s1_bnrelu_tr_fptr_016[] = {
   _dw_conv3x3_p1s1_bnrelu_tr_null,
   _dw_conv3x3_p1s1_bnrelu_tr_16,
   _dw_conv3x3_p1s1_bnrelu_tr_32,
   _dw_conv3x3_p1s1_bnrelu_tr_48,
   _dw_conv3x3_p1s1_bnrelu_tr_64,
   _dw_conv3x3_p1s1_bnrelu_tr_80,
   _dw_conv3x3_p1s1_bnrelu_tr_96,
   _dw_conv3x3_p1s1_bnrelu_tr_112,
};

static dw_conv3x3_p1s1_bnrelu_tr_func _dw_conv3x3_p1s1_bnrelu_tr_fptr_128[] = {
   _dw_conv3x3_p1s1_bnrelu_tr_null,
   _dw_conv3x3_p1s1_bnrelu_tr_128,
   _dw_conv3x3_p1s1_bnrelu_tr_256,
   _dw_conv3x3_p1s1_bnrelu_tr_384,
   _dw_conv3x3_p1s1_bnrelu_tr_512,
   _dw_conv3x3_p1s1_bnrelu_tr_640,
   _dw_conv3x3_p1s1_bnrelu_tr_768,
   _dw_conv3x3_p1s1_bnrelu_tr_896,
   _dw_conv3x3_p1s1_bnrelu_tr_1024,
};

// static dw_conv3x3_p1s1_bnrelu_tr_func _dw_conv3x3_p1s1_bnrelu_tr_fptr_xxx[] = {
//    _dw_conv3x3_p1s1_bnrelu_tr_null,
//    _dw_conv3x3_p1s1_bnrelu_tr_1024,
//    _dw_conv3x3_p1s1_bnrelu_tr_2048,
//    _dw_conv3x3_p1s1_bnrelu_tr_3072,
// };

void dw_conv3x3_p1s1_bnrelu_tr(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int C1 = __Lx;

   int __i016 = (C1 >>  4) & 7;
   int __i128 = (C1 >>  7) & 7;
   int __ixxx = (C1 >> 10) ;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      _dw_conv3x3_p1s1_bnrelu_tr_fptr_128[8](
         __Ln, __Lx, __Ly, 

         __px + (__i << 10), 
         __pw + (__i << 10),
         __pu + (__i << 10),
         __ps + (__i << 10), 
         __pb + (__i << 10),

         maxd, 

         __py + (__i << 10)
         );
   }

   _dw_conv3x3_p1s1_bnrelu_tr_fptr_128[__i128](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10), 
      __pw + (__ixxx << 10),
      __pu + (__ixxx << 10),
      __ps + (__ixxx << 10), 
      __pb + (__ixxx << 10),

      maxd, 

      __py + (__ixxx << 10)
      );

   _dw_conv3x3_p1s1_bnrelu_tr_fptr_016[__i016](
      __Ln, __Lx, __Ly, 

      __px + (__ixxx << 10) + (__i128 << 7), 
      __pw + (__ixxx << 10) + (__i128 << 7),
      __pu + (__ixxx << 10) + (__i128 << 7),
      __ps + (__ixxx << 10) + (__i128 << 7), 
      __pb + (__ixxx << 10) + (__i128 << 7),

      maxd, 

      __py + (__ixxx << 10) + (__i128 << 7)
      );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define macpool_bnrelu_inp_core_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   cnn_type_t s1[4] = {0};  \
   cnn_type_t s2[4] = {0};  \
   cnn_type_t s3[4] = {0};  \
   cnn_type_t s4[4] = {0};  \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 8]) * (wloc[p +  8]);}; \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 8]) * (wloc[p + 20]);}; \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x20[p + 0]) * (wloc[p + 24]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x20[p + 4]) * (wloc[p + 28]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x20[p + 8]) * (wloc[p + 32]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 8]) * (wloc[p +  8]);}; \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 8]) * (wloc[p + 20]);}; \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x20[p + 0]) * (wloc[p + 24]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x20[p + 4]) * (wloc[p + 28]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x20[p + 8]) * (wloc[p + 32]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 8]) * (wloc[p +  8]);}; \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 8]) * (wloc[p + 20]);}; \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x20[p + 0]) * (wloc[p + 24]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x20[p + 4]) * (wloc[p + 28]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x20[p + 8]) * (wloc[p + 32]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 8]) * (wloc[p +  8]);}; \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 8]) * (wloc[p + 20]);}; \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x20[p + 0]) * (wloc[p + 24]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x20[p + 4]) * (wloc[p + 28]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x20[p + 8]) * (wloc[p + 32]);}; \
   \
   wloc += 36;  \
   \
   cnn_type_t s[4];              \
   s[0] = s1[0] + s1[1] + s1[2] + s1[3]; \
   s[1] = s2[0] + s2[1] + s2[2] + s2[3]; \
   s[2] = s3[0] + s3[1] + s3[2] + s3[3]; \
   s[3] = s4[0] + s4[1] + s4[2] + s4[3]; \
   \
   for (int p = 0; p < 4; p++){s[p] = (s[p] - uloc[p]) * sloc[p] + bloc[p];};                  \
   for (int p = 0; p < 4; p++){y[p] = (s[p] < 0) ? (0) : ((s[p] > maxd) ? (maxd) : (s[p]));};  \
   \
   y += 4;  uloc += 4; sloc += 4; bloc += 4;  \
}

#define macpool_bnrelu_inp_core_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_core_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_core_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_core_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_core_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_core_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_core_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_core_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_core_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_core_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_core_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_core_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_core_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_core_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_core_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_core_256(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_core_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_core_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

void macpool_bnrelu_inp_core(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);

   cnn_type_t x00[12], x10[12], x20[12];

   for (int p = 0; p < 12; p++){x00[p] = _px[              p];};
   for (int p = 0; p < 12; p++){x10[p] = _px[       __Ln + p];};
   for (int p = 0; p < 12; p++){x20[p] = _px[__Ln + __Ln + p];};

   macpool_bnrelu_inp_core_16 (x00, x10, x20, _pw, _pu, _ps, _pb, maxd, _py);
   macpool_bnrelu_inp_core_32 (x00, x10, x20, _pw, _pu, _ps, _pb, maxd, _py);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define macpool_bnrelu_inp_rr_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   cnn_type_t s1[4] = {0};  \
   cnn_type_t s2[4] = {0};  \
   cnn_type_t s3[4] = {0};  \
   cnn_type_t s4[4] = {0};  \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x20[p + 0]) * (wloc[p + 24]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x20[p + 4]) * (wloc[p + 28]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x20[p + 0]) * (wloc[p + 24]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x20[p + 4]) * (wloc[p + 28]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x20[p + 0]) * (wloc[p + 24]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x20[p + 4]) * (wloc[p + 28]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x20[p + 0]) * (wloc[p + 24]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x20[p + 4]) * (wloc[p + 28]);}; \
   \
   wloc += 36;  \
   \
   cnn_type_t s[4];              \
   s[0] = s1[0] + s1[1] + s1[2] + s1[3]; \
   s[1] = s2[0] + s2[1] + s2[2] + s2[3]; \
   s[2] = s3[0] + s3[1] + s3[2] + s3[3]; \
   s[3] = s4[0] + s4[1] + s4[2] + s4[3]; \
   \
   for (int p = 0; p < 4; p++){s[p] = (s[p] - uloc[p]) * sloc[p] + bloc[p];};                  \
   for (int p = 0; p < 4; p++){y[p] = (s[p] < 0) ? (0) : ((s[p] > maxd) ? (maxd) : (s[p]));};  \
   \
   y += 4;  uloc += 4; sloc += 4; bloc += 4;  \
}

#define macpool_bnrelu_inp_rr_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_rr_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_rr_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_rr_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_rr_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_rr_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_rr_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_rr_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_rr_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_rr_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_rr_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_rr_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_rr_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_rr_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_rr_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_rr_256(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_rr_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_rr_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

void macpool_bnrelu_inp_rr(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);

   cnn_type_t x00[12], x10[12], x20[12];

   for (int p = 0; p < 8; p++){x00[p] = _px[              p];};
   for (int p = 0; p < 8; p++){x10[p] = _px[       __Ln + p];};
   for (int p = 0; p < 8; p++){x20[p] = _px[__Ln + __Ln + p];};

   macpool_bnrelu_inp_rr_16 (x00, x10, x20, _pw, _pu, _ps, _pb, maxd, _py);
   macpool_bnrelu_inp_rr_32 (x00, x10, x20, _pw, _pu, _ps, _pb, maxd, _py);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define macpool_bnrelu_inp_bb_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   cnn_type_t s1[4] = {0};  \
   cnn_type_t s2[4] = {0};  \
   cnn_type_t s3[4] = {0};  \
   cnn_type_t s4[4] = {0};  \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 8]) * (wloc[p +  8]);}; \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 8]) * (wloc[p + 20]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 8]) * (wloc[p +  8]);}; \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 8]) * (wloc[p + 20]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 8]) * (wloc[p +  8]);}; \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 8]) * (wloc[p + 20]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 8]) * (wloc[p +  8]);}; \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 8]) * (wloc[p + 20]);}; \
   \
  wloc += 36;  \
   \
   cnn_type_t s[4];              \
   s[0] = s1[0] + s1[1] + s1[2] + s1[3]; \
   s[1] = s2[0] + s2[1] + s2[2] + s2[3]; \
   s[2] = s3[0] + s3[1] + s3[2] + s3[3]; \
   s[3] = s4[0] + s4[1] + s4[2] + s4[3]; \
   \
   for (int p = 0; p < 4; p++){s[p] = (s[p] - uloc[p]) * sloc[p] + bloc[p];};                  \
   for (int p = 0; p < 4; p++){y[p] = (s[p] < 0) ? (0) : ((s[p] > maxd) ? (maxd) : (s[p]));};  \
   \
   y += 4;  uloc += 4; sloc += 4; bloc += 4;  \
}

#define macpool_bnrelu_inp_bb_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_bb_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_bb_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_bb_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_bb_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_bb_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_bb_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_bb_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_bb_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_bb_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_bb_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_bb_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_bb_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_bb_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_bb_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_bb_256(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_bb_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_bb_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

void macpool_bnrelu_inp_bb(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);

   cnn_type_t x00[12], x10[12];

   for (int p = 0; p < 12; p++){x00[p] = _px[              p];};
   for (int p = 0; p < 12; p++){x10[p] = _px[       __Ln + p];};

   macpool_bnrelu_inp_bb_16 (x00, x10, x20, _pw, _pu, _ps, _pb, maxd, _py);
   macpool_bnrelu_inp_bb_32 (x00, x10, x20, _pw, _pu, _ps, _pb, maxd, _py);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define macpool_bnrelu_inp_br_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   cnn_type_t s1[4] = {0};  \
   cnn_type_t s2[4] = {0};  \
   cnn_type_t s3[4] = {0};  \
   cnn_type_t s4[4] = {0};  \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s1[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s2[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s3[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   \
   wloc += 36;  \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 0]) * (wloc[p +  0]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x00[p + 4]) * (wloc[p +  4]);}; \
   \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 0]) * (wloc[p + 12]);}; \
   for (int p = 0; p < 4; p++){s4[p] += (x10[p + 4]) * (wloc[p + 16]);}; \
   \
   wloc += 36;  \
   \
   cnn_type_t s[4];              \
   s[0] = s1[0] + s1[1] + s1[2] + s1[3]; \
   s[1] = s2[0] + s2[1] + s2[2] + s2[3]; \
   s[2] = s3[0] + s3[1] + s3[2] + s3[3]; \
   s[3] = s4[0] + s4[1] + s4[2] + s4[3]; \
   \
   for (int p = 0; p < 4; p++){s[p] = (s[p] - uloc[p]) * sloc[p] + bloc[p];};                  \
   for (int p = 0; p < 4; p++){y[p] = (s[p] < 0) ? (0) : ((s[p] > maxd) ? (maxd) : (s[p]));};  \
   \
   y += 4;  uloc += 4; sloc += 4; bloc += 4;  \
}

#define macpool_bnrelu_inp_br_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_br_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_br_04(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_br_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_br_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_br_08(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_br_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_br_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_br_16(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_br_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_br_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_br_32(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_br_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_br_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_br_64(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

#define macpool_bnrelu_inp_br_256(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y) \
{  \
   macpool_bnrelu_inp_br_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
   macpool_bnrelu_inp_br_128(x00, x10, x20, wloc, uloc, sloc, bloc, maxd, y);  \
}

void macpool_bnrelu_inp_br(
   const int                   __Ln,
   const int                   __Lx,
   const int                   __Ly,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);

   cnn_type_t x00[12], x10[12];

   for (int p = 0; p < 8; p++){x00[p] = _px[              p];};
   for (int p = 0; p < 8; p++){x10[p] = _px[       __Ln + p];};

   macpool_bnrelu_inp_br_16 (x00, x10, x20, _pw, _pu, _ps, _pb, maxd, _py);
   macpool_bnrelu_inp_br_32 (x00, x10, x20, _pw, _pu, _ps, _pb, maxd, _py);
}


/* conv3x3 (stride = 2, same), input layer */
void conv3x3_p1s2_bnrelu_inp(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);

   int __Ln = C1 * 1 * N;

   tf_assert((C2 & 15) == 0);

   cnn_type_t * py = _py;

   for (int m = 1; m < (M-1); m += 2)
   {
      cnn_type_t * px = _px + (m-1) * N * C1; // - C1;

      /* CC */
      for (int n = 1; n < (N-1); n += 2) 
      {
         macpool_bnrelu_inp_core(__Ln, C1, C2, px, _pw, _pu, _ps, _pb, maxd, py);
         px += (C1 + C1);
         py += (C2     );
      }

      /* RR */
      {
         macpool_bnrelu_inp_rr(__Ln, C1, C2, px, _pw, _pu, _ps, _pb, maxd, py);
         px += (C1 + C1);
         py += (C2     );
      }
   }

   /* M - 1 */
   {
      int m = (M-1);

      cnn_type_t * px = _px + (m-1) * N * C1;

      /* BB */
      for (int n = 1; n < (N-1); n += 2) 
      {
         macpool_bnrelu_inp_bb(__Ln, C1, C2, px, _pw, _pu, _ps, _pb, maxd, py);
         px += (C1 + C1);
         py += (C2     );
      }

      /* BR */
      {
         macpool_bnrelu_inp_br(__Ln, C1, C2, px, _pw, _pu, _ps, _pb, maxd, py);
         px += (C1 + C1);
         py += (C2     );
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define conv1x1_p0s1_bn_core_08 \
{  \
   cnn_type_t yc[8];  \
   \
   macpool4(C1, C1, _px, _pw, yc + 0);  \
   _pw += (C1 * 4);                    \
   \
   macpool4(C1, C1, _px, _pw, yc + 4);  \
   _pw += (C1 * 4);                    \
   \
   for (int p = 0; p < 8; p++){_py[p] = (yc[p] - _pu[p]) * _ps[p] + _pb[p];};  \
   \
   _pu += 8; _ps += 8; _pb += 8; _py += 8;  \
}

#define conv1x1_p0s1_bn_core_16 \
{  \
   conv1x1_p0s1_bn_core_08;  \
   conv1x1_p0s1_bn_core_08;  \
}

#define conv1x1_p0s1_bn_core_32 \
{  \
   conv1x1_p0s1_bn_core_16;  \
   conv1x1_p0s1_bn_core_16;  \
}

#define conv1x1_p0s1_bn_core_64 \
{  \
   conv1x1_p0s1_bn_core_32;  \
   conv1x1_p0s1_bn_core_32;  \
}

#define conv1x1_p0s1_bn_core_128 \
{  \
   conv1x1_p0s1_bn_core_64;  \
   conv1x1_p0s1_bn_core_64;  \
}

#define conv1x1_p0s1_bn_core_256 \
{  \
   conv1x1_p0s1_bn_core_128;  \
   conv1x1_p0s1_bn_core_128;  \
}

#define conv1x1_p0s1_bn_core_512 \
{  \
   conv1x1_p0s1_bn_core_256;  \
   conv1x1_p0s1_bn_core_256;  \
}

#define conv1x1_p0s1_bn_vardecl  \
   \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \

#define conv1x1_p0s1_bn_paramdecl \
   \
   const int                     C1,  \
   const cnn_type_t * restrict __px,  \
   const cnn_type_t * restrict __pw,  \
   const cnn_type_t * restrict __pu,  \
   const cnn_type_t * restrict __ps,  \
   const cnn_type_t * restrict __pb,  \
         cnn_type_t * restrict __py   \

static void _conv1x1_p0s1_bn_core_08(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;
   conv1x1_p0s1_bn_core_08;
}

static void _conv1x1_p0s1_bn_core_16(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;
   conv1x1_p0s1_bn_core_16;
}

static void _conv1x1_p0s1_bn_core_24(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_08;
   conv1x1_p0s1_bn_core_16;
}

static void _conv1x1_p0s1_bn_core_32(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;
   conv1x1_p0s1_bn_core_32;
}

static void _conv1x1_p0s1_bn_core_40(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;
   conv1x1_p0s1_bn_core_08;
   conv1x1_p0s1_bn_core_32;
}

static void _conv1x1_p0s1_bn_core_48(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;
   conv1x1_p0s1_bn_core_16;
   conv1x1_p0s1_bn_core_32;
}

static void _conv1x1_p0s1_bn_core_56(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_08;
   conv1x1_p0s1_bn_core_16;
   conv1x1_p0s1_bn_core_32;
}

static void _conv1x1_p0s1_bn_core_64(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_16;
   conv1x1_p0s1_bn_core_16;
   conv1x1_p0s1_bn_core_32;
}

static void _conv1x1_p0s1_bn_core_128(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_64;
   conv1x1_p0s1_bn_core_64;
}

static void _conv1x1_p0s1_bn_core_192(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_64;
   conv1x1_p0s1_bn_core_64;
   conv1x1_p0s1_bn_core_64;
}

static void _conv1x1_p0s1_bn_core_256(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_64;
   conv1x1_p0s1_bn_core_64;
   conv1x1_p0s1_bn_core_128;
}

static void _conv1x1_p0s1_bn_core_320(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_64;
   conv1x1_p0s1_bn_core_128;
   conv1x1_p0s1_bn_core_128;
}

static void _conv1x1_p0s1_bn_core_384(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_128;
   conv1x1_p0s1_bn_core_128;
   conv1x1_p0s1_bn_core_128;
}

static void _conv1x1_p0s1_bn_core_448(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_64;
   conv1x1_p0s1_bn_core_128;
   conv1x1_p0s1_bn_core_256;
}

static void _conv1x1_p0s1_bn_core_512(
   conv1x1_p0s1_bn_paramdecl
)
{
   conv1x1_p0s1_bn_vardecl;

   conv1x1_p0s1_bn_core_128;
   conv1x1_p0s1_bn_core_128;
   conv1x1_p0s1_bn_core_256;
}

static void _conv1x1_p0s1_bn_core_null(
   conv1x1_p0s1_bn_paramdecl
)
{
}


typedef void (*conv1x1_p0s1_bn_core_func)(
   conv1x1_p0s1_bn_paramdecl
);

static conv1x1_p0s1_bn_core_func conv1x1_p0s1_bn_core_func_08[] = {
   _conv1x1_p0s1_bn_core_null,
   _conv1x1_p0s1_bn_core_08,
   _conv1x1_p0s1_bn_core_16,
   _conv1x1_p0s1_bn_core_24,
   _conv1x1_p0s1_bn_core_32,
   _conv1x1_p0s1_bn_core_40,
   _conv1x1_p0s1_bn_core_48,
   _conv1x1_p0s1_bn_core_56,
};

static conv1x1_p0s1_bn_core_func conv1x1_p0s1_bn_core_func_64[] = {
   _conv1x1_p0s1_bn_core_null,
   _conv1x1_p0s1_bn_core_64,
   _conv1x1_p0s1_bn_core_128,
   _conv1x1_p0s1_bn_core_192,
   _conv1x1_p0s1_bn_core_256,
   _conv1x1_p0s1_bn_core_320,
   _conv1x1_p0s1_bn_core_384,
   _conv1x1_p0s1_bn_core_448,
   _conv1x1_p0s1_bn_core_512,
};

void conv1x1_p0s1_bn_core(
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
         cnn_type_t * restrict __py
)
{
   int __ixxx = (C2 >> 9) ;
   int __i064 = (C2 >> 6) & 7;
   int __i008 = (C2 >> 3) & 7;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      conv1x1_p0s1_bn_core_func_64[8](
         C1,
       __px,
       __pw + (__i << 9) * C1,
       __pu + (__i << 9) ,
       __ps + (__i << 9) ,
       __pb + (__i << 9) ,
       __py + (__i << 9)
       );
   }

   conv1x1_p0s1_bn_core_func_64[__i064](
      C1,
      __px,
      __pw + (__ixxx << 9) * C1,
      __pu + (__ixxx << 9) ,
      __ps + (__ixxx << 9) ,
      __pb + (__ixxx << 9) ,
      __py + (__ixxx << 9)
      );

   conv1x1_p0s1_bn_core_func_08[__i008](
      C1,
      __px,
      __pw + ((__i064 << 6) + (__ixxx << 9)) * C1,
      __pu + ((__i064 << 6) + (__ixxx << 9)) ,
      __ps + ((__i064 << 6) + (__ixxx << 9)) ,
      __pb + ((__i064 << 6) + (__ixxx << 9)) ,
      __py + ((__i064 << 6) + (__ixxx << 9))
      );

}

void conv1x1_p0s1_bn(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);
   tf_assert((C2 & 7) == 0);

   for (int i = 0; i < M*N; i++)
   {
      conv1x1_p0s1_bn_core(C1, C2, _px, __pw, __pu, __ps, __pb, _py);
      _px += C1; _py += C2;
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define conv1x1_p0s1_bnrelu_core_08 \
{  \
   cnn_type_t yc[8], yd[8];  \
   \
   macpool4(C1, C1, _px, _pw, yc + 0);  \
   _pw += (C1 * 4);                    \
   \
   macpool4(C1, C1, _px, _pw, yc + 4);  \
   _pw += (C1 * 4);                    \
   \
   for (int p = 0; p < 8; p++){ yd[p] = (yc[p] - _pu[p]) * _ps[p] + _pb[p];};  \
   for (int p = 0; p < 8; p++){_py[p] = (yd[p] < 0) ? (0) : ((yd[p] > maxd) ? (maxd) : (yd[p]));};  \
   \
   _pu += 8; _ps += 8; _pb += 8; _py += 8;  \
}

#define conv1x1_p0s1_bnrelu_core_16 \
{  \
   conv1x1_p0s1_bnrelu_core_08;  \
   conv1x1_p0s1_bnrelu_core_08;  \
}

#define conv1x1_p0s1_bnrelu_core_32 \
{  \
   conv1x1_p0s1_bnrelu_core_16;  \
   conv1x1_p0s1_bnrelu_core_16;  \
}

#define conv1x1_p0s1_bnrelu_core_64 \
{  \
   conv1x1_p0s1_bnrelu_core_32;  \
   conv1x1_p0s1_bnrelu_core_32;  \
}

#define conv1x1_p0s1_bnrelu_core_128 \
{  \
   conv1x1_p0s1_bnrelu_core_64;  \
   conv1x1_p0s1_bnrelu_core_64;  \
}

#define conv1x1_p0s1_bnrelu_core_256 \
{  \
   conv1x1_p0s1_bnrelu_core_128;  \
   conv1x1_p0s1_bnrelu_core_128;  \
}

#define conv1x1_p0s1_bnrelu_core_512 \
{  \
   conv1x1_p0s1_bnrelu_core_256;  \
   conv1x1_p0s1_bnrelu_core_256;  \
}

#define conv1x1_p0s1_bnrelu_vardecl  \
   \
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);  \
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);  \
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);  \
   cnn_type_t * _pu = __builtin_assume_aligned(__pu, 16);  \
   cnn_type_t * _ps = __builtin_assume_aligned(__ps, 16);  \
   cnn_type_t * _pb = __builtin_assume_aligned(__pb, 16);  \

#define conv1x1_p0s1_bnrelu_paramdecl \
   \
   const int                     C1,  \
   const cnn_type_t * restrict __px,  \
   const cnn_type_t * restrict __pw,  \
   const cnn_type_t * restrict __pu,  \
   const cnn_type_t * restrict __ps,  \
   const cnn_type_t * restrict __pb,  \
   const cnn_type_t            maxd,  \
         cnn_type_t * restrict __py   \

static void _conv1x1_p0s1_bnrelu_core_08(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;
   conv1x1_p0s1_bnrelu_core_08;
}

static void _conv1x1_p0s1_bnrelu_core_16(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;
   conv1x1_p0s1_bnrelu_core_16;
}

static void _conv1x1_p0s1_bnrelu_core_24(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_08;
   conv1x1_p0s1_bnrelu_core_16;
}

static void _conv1x1_p0s1_bnrelu_core_32(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;
   conv1x1_p0s1_bnrelu_core_32;
}

static void _conv1x1_p0s1_bnrelu_core_40(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;
   conv1x1_p0s1_bnrelu_core_08;
   conv1x1_p0s1_bnrelu_core_32;
}

static void _conv1x1_p0s1_bnrelu_core_48(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;
   conv1x1_p0s1_bnrelu_core_16;
   conv1x1_p0s1_bnrelu_core_32;
}

static void _conv1x1_p0s1_bnrelu_core_56(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_08;
   conv1x1_p0s1_bnrelu_core_16;
   conv1x1_p0s1_bnrelu_core_32;
}

static void _conv1x1_p0s1_bnrelu_core_64(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_16;
   conv1x1_p0s1_bnrelu_core_16;
   conv1x1_p0s1_bnrelu_core_32;
}

static void _conv1x1_p0s1_bnrelu_core_128(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_64;
   conv1x1_p0s1_bnrelu_core_64;
}

static void _conv1x1_p0s1_bnrelu_core_192(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_64;
   conv1x1_p0s1_bnrelu_core_64;
   conv1x1_p0s1_bnrelu_core_64;
}

static void _conv1x1_p0s1_bnrelu_core_256(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_64;
   conv1x1_p0s1_bnrelu_core_64;
   conv1x1_p0s1_bnrelu_core_128;
}

static void _conv1x1_p0s1_bnrelu_core_320(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_64;
   conv1x1_p0s1_bnrelu_core_128;
   conv1x1_p0s1_bnrelu_core_128;
}

static void _conv1x1_p0s1_bnrelu_core_384(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_128;
   conv1x1_p0s1_bnrelu_core_128;
   conv1x1_p0s1_bnrelu_core_128;
}

static void _conv1x1_p0s1_bnrelu_core_448(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_64;
   conv1x1_p0s1_bnrelu_core_128;
   conv1x1_p0s1_bnrelu_core_256;
}

static void _conv1x1_p0s1_bnrelu_core_512(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
   conv1x1_p0s1_bnrelu_vardecl;

   conv1x1_p0s1_bnrelu_core_128;
   conv1x1_p0s1_bnrelu_core_128;
   conv1x1_p0s1_bnrelu_core_256;
}

static void _conv1x1_p0s1_bnrelu_core_null(
   conv1x1_p0s1_bnrelu_paramdecl
)
{
}


typedef void (*conv1x1_p0s1_bnrelu_core_func)(
   conv1x1_p0s1_bnrelu_paramdecl
);

static conv1x1_p0s1_bnrelu_core_func conv1x1_p0s1_bnrelu_core_func_08[] = {
   _conv1x1_p0s1_bnrelu_core_null,
   _conv1x1_p0s1_bnrelu_core_08,
   _conv1x1_p0s1_bnrelu_core_16,
   _conv1x1_p0s1_bnrelu_core_24,
   _conv1x1_p0s1_bnrelu_core_32,
   _conv1x1_p0s1_bnrelu_core_40,
   _conv1x1_p0s1_bnrelu_core_48,
   _conv1x1_p0s1_bnrelu_core_56,
};

static conv1x1_p0s1_bnrelu_core_func conv1x1_p0s1_bnrelu_core_func_64[] = {
   _conv1x1_p0s1_bnrelu_core_null,
   _conv1x1_p0s1_bnrelu_core_64,
   _conv1x1_p0s1_bnrelu_core_128,
   _conv1x1_p0s1_bnrelu_core_192,
   _conv1x1_p0s1_bnrelu_core_256,
   _conv1x1_p0s1_bnrelu_core_320,
   _conv1x1_p0s1_bnrelu_core_384,
   _conv1x1_p0s1_bnrelu_core_448,
   _conv1x1_p0s1_bnrelu_core_512,
};

void conv1x1_p0s1_bnrelu_core(
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   int __ixxx = (C2 >> 9) ;
   int __i064 = (C2 >> 6) & 7;
   int __i008 = (C2 >> 3) & 7;

   for (int __i = 0; __i < __ixxx; __i++)
   {
      conv1x1_p0s1_bnrelu_core_func_64[8](
         C1,
       __px,
       __pw + (__i << 9) * C1,
       __pu + (__i << 9) ,
       __ps + (__i << 9) ,
       __pb + (__i << 9) ,
       maxd,
       __py + (__i << 9)
       );
   }

   conv1x1_p0s1_bnrelu_core_func_64[__i064](
      C1,
      __px,
      __pw + (__ixxx << 9) * C1,
      __pu + (__ixxx << 9) ,
      __ps + (__ixxx << 9) ,
      __pb + (__ixxx << 9) ,
      maxd,
      __py + (__ixxx << 9)
      );

   conv1x1_p0s1_bnrelu_core_func_08[__i008](
      C1,
      __px,
      __pw + ((__i064 << 6) + (__ixxx << 9)) * C1,
      __pu + ((__i064 << 6) + (__ixxx << 9)) ,
      __ps + ((__i064 << 6) + (__ixxx << 9)) ,
      __pb + ((__i064 << 6) + (__ixxx << 9)) ,
      maxd,
      __py + ((__i064 << 6) + (__ixxx << 9))
      );

}

void conv1x1_p0s1_bnrelu(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
   const cnn_type_t * restrict __pu,
   const cnn_type_t * restrict __ps,
   const cnn_type_t * restrict __pb,
   const cnn_type_t            maxd,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);
   tf_assert((C2 & 7) == 0);

   for (int m = 0; m < M * N; m++)
   {
      conv1x1_p0s1_bnrelu_core(C1, C2, _px, __pw, __pu, __ps, __pb, maxd, _py);
      _px += C1; _py += C2;
   }
}
