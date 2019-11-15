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
void dw_conv3x3_p1s1_bnrelu_core_tile2x2(
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

void dw_conv3x3_p1s1_bnrelu_core_tile2x1(
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

void dw_conv3x3_p1s1_bnrelu_ll_tile2x1(
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

void dw_conv3x3_p1s1_bnrelu_rr_tile2x1(
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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* depthwise conv3x3 (stride = 1, same) */
void dw_conv3x3_p1s1_bnrelu(
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

   int __Ln = C1 * 1 * N;

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);
   tf_assert((C2 & 15) == 0);

   {
      int m = 0;

      cnn_type_t * px = _px + (m-1) * N * C1 - C1;
      cnn_type_t * py = _py + (m-0) * N * C2;

      /* TL */
      {
         dw_conv3x3_p1s1_bnrelu_tl(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }

      /* TT */
      for (int n = 1; n < (N-1); n++) 
      {
         dw_conv3x3_p1s1_bnrelu_tt(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }

      /* TR */
      {
         dw_conv3x3_p1s1_bnrelu_tr(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }
   }

   #if (0)
   for (int m = 1; m < (M-2); m+=2)
   {
      cnn_type_t * px = _px + (m-1) * N * C1 - C1;
      cnn_type_t * py = _py + (m-0) * N * C2;

      /* LL */
      {
         dw_conv3x3_p1s1_bnrelu_ll_tile2x1(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }

      /* CC */
      for (int n = 1; n < (N-2); n+=2) 
      {
         dw_conv3x3_p1s1_bnrelu_core_tile2x2(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += (C1 + C1);
         px += (C1 + C1);
      }

      if (N & 1)
      {
         dw_conv3x3_p1s1_bnrelu_core_tile2x1(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;

      }

      /* RR */
      {
         dw_conv3x3_p1s1_bnrelu_rr_tile2x1(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }
   }
   #endif

   for (int m = 1; m < (M-1); m+=1)
   {
      // int m = M - 2;
      cnn_type_t * px = _px + (m-1) * N * C1 - C1;
      cnn_type_t * py = _py + (m-0) * N * C2;
      
      /* LL */
      {
         dw_conv3x3_p1s1_bnrelu_ll(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }

      /* CC */
      for (int n = 1; n < (N-1); n++) 
      {
         dw_conv3x3_p1s1_bnrelu_core(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }

      /* RR */
      {
         dw_conv3x3_p1s1_bnrelu_rr(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }
   }

   /* M - 1 */
   {
      int m = (M-1);

      cnn_type_t * px = _px + (m-1) * N * C1 - C1;
      cnn_type_t * py = _py + (m-0) * N * C2;

      /* BL */
      {
         dw_conv3x3_p1s1_bnrelu_bl(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }

      /* BB */
      for (int n = 1; n < (N-1); n++) 
      {
         dw_conv3x3_p1s1_bnrelu_bb(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }

      /* BR */
      {
         dw_conv3x3_p1s1_bnrelu_br(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         py += C1;
         px += C1;
      }
   }
}

/* depthwise conv3x3 (stride = 2, same) */
void dw_conv3x3_p1s2_bnrelu(
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

   int __Ln = C1 * 1 * N;

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);
   tf_assert((C2 & 15) == 0);
   tf_assert((M  & 1 ) == 0);
   tf_assert((N  & 1 ) == 0);

   cnn_type_t * py = _py;

   for (int m = 1; m < (M-1); m+=2)
   {
      cnn_type_t * px = _px + (m-1) * N * C1;

      /* CC */
      for (int n = 1; n < (N-1); n+=2) 
      {
         dw_conv3x3_p1s1_bnrelu_core(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         px += (C1 + C1);
         py += (C1     );
      }

      /* RR */
      {
         dw_conv3x3_p1s1_bnrelu_rr(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         px += (C1 + C1);
         py += (C1     );
      }
   }

   /* M - 1 */
   {
      int m = (M-1);

      cnn_type_t * px = _px + (m-1) * N * C1;

      /* BB */
      for (int n = 1; n < (N-1); n+=2) 
      {
         dw_conv3x3_p1s1_bnrelu_bb(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         px += (C1 + C1);
         py += (C1     );
      }

      {
         dw_conv3x3_p1s1_bnrelu_br(__Ln, C1, C2, px, __pw, __pu, __ps, __pb, maxd, py);

         px += (C1 + C1);
         py += (C1     );
      }
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
