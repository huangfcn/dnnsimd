#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <assert.h>

#include "cnntype.h"

int fastmcp(
         void * restrict _pdst, 
   const void * restrict _psrc, 
         int             _nb
);

int fastmzr(
         void * restrict _pdst,
         int             _nb
);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define addpool_08(xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = (xloc[p]) + (wloc[p]);};  \
   \
   xloc += CNN_BCHSIZ;  \
   wloc += CNN_BCHSIZ;  \
   yloc += CNN_BCHSIZ;  \
}

#define addpool_16(xloc, wloc, yloc) \
{  \
   addpool_08(xloc, wloc, yloc);  \
   addpool_08(xloc, wloc, yloc);  \
}

#define addpool_32(xloc, wloc, yloc) \
{  \
   addpool_16(xloc, wloc, yloc);  \
   addpool_16(xloc, wloc, yloc);  \
}

#define addpool_64(xloc, wloc, yloc) \
{  \
   addpool_32(xloc, wloc, yloc);  \
   addpool_32(xloc, wloc, yloc);  \
}

#define addpool_128(xloc, wloc, yloc) \
{  \
   addpool_64(xloc, wloc, yloc);  \
   addpool_64(xloc, wloc, yloc);  \
}

#define addpool_256(xloc, wloc, yloc) \
{  \
   addpool_128(xloc, wloc, yloc);  \
   addpool_128(xloc, wloc, yloc);  \
}

#define addpool_512(xloc, wloc, yloc) \
{  \
   addpool_256(xloc, wloc, yloc);  \
   addpool_256(xloc, wloc, yloc);  \
}

static void _addpool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

#if (0)
static void _addpool_4096(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);

   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_3584(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);

   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_3072(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);

   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_2560(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);

   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
}
#endif

static void _addpool_2048(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_512(xloc, wloc, yloc);
}

static void _addpool_448(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_64 (xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
}

static void _addpool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_128(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
}

static void _addpool_320(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_64 (xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
}

static void _addpool_256(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_128(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
}

static void _addpool_192(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_64(xloc, wloc, yloc);
   addpool_128(xloc, wloc, yloc);
}

static void _addpool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_128(xloc, wloc, yloc);
}

static void _addpool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_64(xloc, wloc, yloc);
}

static void _addpool_56(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_08(xloc, wloc, yloc);
   addpool_16(xloc, wloc, yloc);
   addpool_32(xloc, wloc, yloc);
}

static void _addpool_48(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_16(xloc, wloc, yloc);
   addpool_32(xloc, wloc, yloc);
}

static void _addpool_40(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_08(xloc, wloc, yloc);
   addpool_32(xloc, wloc, yloc);
}

static void _addpool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_32(xloc, wloc, yloc);
}

static void _addpool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_16(xloc, wloc, yloc);
   addpool_08(xloc, wloc, yloc);
}

static void _addpool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_16(xloc, wloc, yloc);
}

static void _addpool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool_08(xloc, wloc, yloc);
}

#if (0)
static void _addpool_07(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 7; i++){yloc[i] = (xloc[i] + wloc[i]);};
}

static void _addpool_06(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 6; i++){yloc[i] = (xloc[i] + wloc[i]);};
}

static void _addpool_05(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 5; i++){yloc[i] = (xloc[i] + wloc[i]);};
}

static void _addpool_04(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 4; i++){yloc[i] = (xloc[i] + wloc[i]);};
}

static void _addpool_03(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 3; i++){yloc[i] = (xloc[i] + wloc[i]);};
}

static void _addpool_02(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 2; i++){yloc[i] = (xloc[i] + wloc[i]);};
}

static void _addpool_01(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 1; i++){yloc[i] = (xloc[i] + wloc[i]);};
}
#endif

typedef void (*addpool_func_t)(const cnn_type_t *, const cnn_type_t *, cnn_type_t *);
static addpool_func_t _addpool_tab_512[] = {
   _addpool_00,
   _addpool_512,
   _addpool_1024,
   _addpool_1536,
   _addpool_2048,
   // _addpool_2560,
   // _addpool_3072,
   // _addpool_3584,   
};

static addpool_func_t _addpool_tab_064[] = {
   _addpool_00,
   _addpool_64,
   _addpool_128,
   _addpool_192,
   _addpool_256,
   _addpool_320,
   _addpool_384,
   _addpool_448,
};

static addpool_func_t _addpool_tab_008[] = {
   _addpool_00,
   _addpool_08,
   _addpool_16,
   _addpool_24,
   _addpool_32,
   _addpool_40,
   _addpool_48,
   _addpool_56,
};

/*
static addpool_func_t _addpool_tab_001[] = {
   _addpool_00,
   _addpool_01,
   _addpool_02,
   _addpool_03,
   _addpool_04,
   _addpool_05,
   _addpool_06,
   _addpool_07,
};
*/

void addpool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   int __ixxx = (_n >> 11);
   int __i512 = (_n >>  9) & (3);
   int __i064 = (_n >>  6) & (7);
   int __i008 = (_n >>  3) & (7);
   // int __i001 = (_n >>  0) & (7);

   for (int i = 0; i < __ixxx; i++)
   {
      _addpool_tab_512[4](xloc, wloc, yloc);
      xloc += 2048;
      wloc += 2048;
      yloc += 2048;
   }

   if (__i512)
   {
      _addpool_tab_512[__i512](xloc, wloc, yloc);
      xloc += (__i512 << 9);
      wloc += (__i512 << 9);
      yloc += (__i512 << 9);
   }

   if (__i064)
   {
      _addpool_tab_064[__i064](xloc, wloc, yloc);
      xloc += (__i064 << 6);
      wloc += (__i064 << 6);
      yloc += (__i064 << 6);
   }

   if (__i008)
   {
      _addpool_tab_008[__i008](xloc, wloc, yloc);
      xloc += (__i008 << 3);
      wloc += (__i008 << 3);
      yloc += (__i008 << 3);
   }

   // _addpool_tab_001[__i001](xloc, wloc, yloc);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define addpool2_08(xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = (xloc[p]) + wloc;};  \
   \
   xloc += CNN_BCHSIZ;  \
   yloc += CNN_BCHSIZ;  \
}

#define addpool2_16(xloc, wloc, yloc) \
{  \
   addpool2_08(xloc, wloc, yloc);  \
   addpool2_08(xloc, wloc, yloc);  \
}

#define addpool2_32(xloc, wloc, yloc) \
{  \
   addpool2_16(xloc, wloc, yloc);  \
   addpool2_16(xloc, wloc, yloc);  \
}

#define addpool2_64(xloc, wloc, yloc) \
{  \
   addpool2_32(xloc, wloc, yloc);  \
   addpool2_32(xloc, wloc, yloc);  \
}

#define addpool2_128(xloc, wloc, yloc) \
{  \
   addpool2_64(xloc, wloc, yloc);  \
   addpool2_64(xloc, wloc, yloc);  \
}

#define addpool2_256(xloc, wloc, yloc) \
{  \
   addpool2_128(xloc, wloc, yloc);  \
   addpool2_128(xloc, wloc, yloc);  \
}

#define addpool2_512(xloc, wloc, yloc) \
{  \
   addpool2_256(xloc, wloc, yloc);  \
   addpool2_256(xloc, wloc, yloc);  \
}

static void _addpool2_00(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _addpool2_1536(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool2_512(xloc, wloc, yloc);
   addpool2_512(xloc, wloc, yloc);
   addpool2_512(xloc, wloc, yloc);
}

static void _addpool2_1024(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool2_512(xloc, wloc, yloc);
   addpool2_512(xloc, wloc, yloc);
}

static void _addpool2_512(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool2_512(xloc, wloc, yloc);
}

static void _addpool2_384(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool2_128(xloc, wloc, yloc);
   addpool2_128(xloc, wloc, yloc);
   addpool2_128(xloc, wloc, yloc);
}

static void _addpool2_256(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   addpool2_128(xloc, wloc, yloc);
   addpool2_128(xloc, wloc, yloc);
}

static void _addpool2_128(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   addpool2_128(xloc, wloc, yloc);
}

static void _addpool2_96(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   addpool2_64(xloc, wloc, yloc);
   addpool2_32(xloc, wloc, yloc);
}

static void _addpool2_64(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   addpool2_64(xloc, wloc, yloc);
}

static void _addpool2_32(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   addpool2_32(xloc, wloc, yloc);
}

static void _addpool2_24(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   addpool2_16(xloc, wloc, yloc);
   addpool2_08(xloc, wloc, yloc);
}

static void _addpool2_16(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool2_16(xloc, wloc, yloc);
}

static void _addpool2_08(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   addpool2_08(xloc, wloc, yloc);
}

static void _addpool2_07(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 7; i++){yloc[i] = (xloc[i] + wloc);};
}

static void _addpool2_06(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 6; i++){yloc[i] = (xloc[i] + wloc);};
}

static void _addpool2_05(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 5; i++){yloc[i] = (xloc[i] + wloc);};
}

static void _addpool2_04(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 4; i++){yloc[i] = (xloc[i] + wloc);};
}

static void _addpool2_03(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 3; i++){yloc[i] = (xloc[i] + wloc);};
}

static void _addpool2_02(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 2; i++){yloc[i] = (xloc[i] + wloc);};
}

static void _addpool2_01(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 1; i++){yloc[i] = (xloc[i] + wloc);};
}


typedef void (*addpool2_func_t)(const cnn_type_t *, cnn_type_t, cnn_type_t *);
static addpool2_func_t _addpool2_tab_512[] = {
   _addpool2_00,
   _addpool2_512,
   _addpool2_1024,
   _addpool2_1536,
};

static addpool2_func_t _addpool2_tab_128[] = {
   _addpool2_00,
   _addpool2_128,
   _addpool2_256,
   _addpool2_384,
};

static addpool2_func_t _addpool2_tab_32[] = {
   _addpool2_00,
   _addpool2_32,
   _addpool2_64,
   _addpool2_96,
};

static addpool2_func_t _addpool2_tab_08[] = {
   _addpool2_00,
   _addpool2_08,
   _addpool2_16,
   _addpool2_24,
};

static addpool2_func_t _addpool2_tab_00[] = {
   _addpool2_00,
   _addpool2_01,
   _addpool2_02,
   _addpool2_03,
   _addpool2_04,
   _addpool2_05,
   _addpool2_06,
   _addpool2_07,
};

void addpool2(
         int                   _n,
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++){
         (_addpool2_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         yloc += (1 << 10);

         (_addpool2_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);
      (_addpool2_tab_512[nb & 3])(xloc, wloc, yloc);

      xloc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);
      (_addpool2_tab_128[nb])(xloc, wloc, yloc);

      xloc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);
      (_addpool2_tab_32[nb])(xloc, wloc, yloc);

      xloc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_addpool2_tab_08[nb])(xloc, wloc, yloc);

      xloc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (_addpool2_tab_00[_n])(xloc, wloc, yloc);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define subpool_08(xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = (xloc[p]) - (wloc[p]);};  \
   \
   xloc += CNN_BCHSIZ;  \
   wloc += CNN_BCHSIZ;  \
   yloc += CNN_BCHSIZ;  \
}

#define subpool_16(xloc, wloc, yloc) \
{  \
   subpool_08(xloc, wloc, yloc);  \
   subpool_08(xloc, wloc, yloc);  \
}

#define subpool_32(xloc, wloc, yloc) \
{  \
   subpool_16(xloc, wloc, yloc);  \
   subpool_16(xloc, wloc, yloc);  \
}

#define subpool_64(xloc, wloc, yloc) \
{  \
   subpool_32(xloc, wloc, yloc);  \
   subpool_32(xloc, wloc, yloc);  \
}

#define subpool_128(xloc, wloc, yloc) \
{  \
   subpool_64(xloc, wloc, yloc);  \
   subpool_64(xloc, wloc, yloc);  \
}

#define subpool_256(xloc, wloc, yloc) \
{  \
   subpool_128(xloc, wloc, yloc);  \
   subpool_128(xloc, wloc, yloc);  \
}

#define subpool_512(xloc, wloc, yloc) \
{  \
   subpool_256(xloc, wloc, yloc);  \
   subpool_256(xloc, wloc, yloc);  \
}

static void _subpool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _subpool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_512(xloc, wloc, yloc);
   subpool_512(xloc, wloc, yloc);
   subpool_512(xloc, wloc, yloc);
}

static void _subpool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_512(xloc, wloc, yloc);
   subpool_512(xloc, wloc, yloc);
}

static void _subpool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_512(xloc, wloc, yloc);
}

static void _subpool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_128(xloc, wloc, yloc);
   subpool_128(xloc, wloc, yloc);
   subpool_128(xloc, wloc, yloc);
}

static void _subpool_256(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_128(xloc, wloc, yloc);
   subpool_128(xloc, wloc, yloc);
}

static void _subpool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_128(xloc, wloc, yloc);
}

static void _subpool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_64(xloc, wloc, yloc);
   subpool_32(xloc, wloc, yloc);
}

static void _subpool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_64(xloc, wloc, yloc);
}

static void _subpool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_32(xloc, wloc, yloc);
}

static void _subpool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_16(xloc, wloc, yloc);
   subpool_08(xloc, wloc, yloc);
}

static void _subpool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_16(xloc, wloc, yloc);
}

static void _subpool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   subpool_08(xloc, wloc, yloc);
}

static void _subpool_07(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 7; i++){yloc[i] = (xloc[i] - wloc[i]);};
}

static void _subpool_06(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 6; i++){yloc[i] = (xloc[i] - wloc[i]);};
}

static void _subpool_05(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 5; i++){yloc[i] = (xloc[i] - wloc[i]);};
}

static void _subpool_04(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 4; i++){yloc[i] = (xloc[i] - wloc[i]);};
}

static void _subpool_03(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 3; i++){yloc[i] = (xloc[i] - wloc[i]);};
}

static void _subpool_02(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 2; i++){yloc[i] = (xloc[i] - wloc[i]);};
}

static void _subpool_01(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 1; i++){yloc[i] = (xloc[i] - wloc[i]);};
}

typedef void (*subpool_func_t)(const cnn_type_t *, const cnn_type_t *, cnn_type_t *);
static subpool_func_t _subpool_tab_512[] = {
   _subpool_00,
   _subpool_512,
   _subpool_1024,
   _subpool_1536,
};

static subpool_func_t _subpool_tab_128[] = {
   _subpool_00,
   _subpool_128,
   _subpool_256,
   _subpool_384,
};

static subpool_func_t _subpool_tab_32[] = {
   _subpool_00,
   _subpool_32,
   _subpool_64,
   _subpool_96,
};

static subpool_func_t _subpool_tab_08[] = {
   _subpool_00,
   _subpool_08,
   _subpool_16,
   _subpool_24,
};

static subpool_func_t _subpool_tab_00[] = {
   _subpool_00,
   _subpool_01,
   _subpool_02,
   _subpool_03,
   _subpool_04,
   _subpool_05,
   _subpool_06,
   _subpool_07,
};

void subpool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++){
         (_subpool_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         wloc += (1 << 10);
         yloc += (1 << 10);

         (_subpool_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         wloc += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);
      (_subpool_tab_512[nb & 3])(xloc, wloc, yloc);

      xloc += (nb << 9);
      wloc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);
      (_subpool_tab_128[nb])(xloc, wloc, yloc);

      xloc += (nb << 7);
      wloc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);
      (_subpool_tab_32[nb])(xloc, wloc, yloc);

      xloc += (nb << 5);
      wloc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_subpool_tab_08[nb])(xloc, wloc, yloc);

      xloc += (nb << 3);
      wloc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (_subpool_tab_00[_n])(xloc, wloc, yloc);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define mulpool_08(xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = (xloc[p]) * (wloc[p]);};  \
   \
   xloc += CNN_BCHSIZ;  \
   wloc += CNN_BCHSIZ;  \
   yloc += CNN_BCHSIZ;  \
}

#define mulpool_16(xloc, wloc, yloc) \
{  \
   mulpool_08(xloc, wloc, yloc);  \
   mulpool_08(xloc, wloc, yloc);  \
}

#define mulpool_32(xloc, wloc, yloc) \
{  \
   mulpool_16(xloc, wloc, yloc);  \
   mulpool_16(xloc, wloc, yloc);  \
}

#define mulpool_64(xloc, wloc, yloc) \
{  \
   mulpool_32(xloc, wloc, yloc);  \
   mulpool_32(xloc, wloc, yloc);  \
}

#define mulpool_128(xloc, wloc, yloc) \
{  \
   mulpool_64(xloc, wloc, yloc);  \
   mulpool_64(xloc, wloc, yloc);  \
}

#define mulpool_256(xloc, wloc, yloc) \
{  \
   mulpool_128(xloc, wloc, yloc);  \
   mulpool_128(xloc, wloc, yloc);  \
}

#define mulpool_512(xloc, wloc, yloc) \
{  \
   mulpool_256(xloc, wloc, yloc);  \
   mulpool_256(xloc, wloc, yloc);  \
}

static void _mulpool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _mulpool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_512(xloc, wloc, yloc);
   mulpool_512(xloc, wloc, yloc);
   mulpool_512(xloc, wloc, yloc);
}

static void _mulpool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_512(xloc, wloc, yloc);
   mulpool_512(xloc, wloc, yloc);
}

static void _mulpool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_512(xloc, wloc, yloc);
}

static void _mulpool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_128(xloc, wloc, yloc);
   mulpool_128(xloc, wloc, yloc);
   mulpool_128(xloc, wloc, yloc);
}

static void _mulpool_256(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_128(xloc, wloc, yloc);
   mulpool_128(xloc, wloc, yloc);
}

static void _mulpool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_128(xloc, wloc, yloc);
}

static void _mulpool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_64(xloc, wloc, yloc);
   mulpool_32(xloc, wloc, yloc);
}

static void _mulpool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_64(xloc, wloc, yloc);
}

static void _mulpool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_32(xloc, wloc, yloc);
}

static void _mulpool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_16(xloc, wloc, yloc);
   mulpool_08(xloc, wloc, yloc);
}

static void _mulpool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_16(xloc, wloc, yloc);
}

static void _mulpool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool_08(xloc, wloc, yloc);
}

static void _mulpool_07(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 7; i++){yloc[i] = (xloc[i] * wloc[i]);};
}

static void _mulpool_06(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 6; i++){yloc[i] = (xloc[i] * wloc[i]);};
}

static void _mulpool_05(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 5; i++){yloc[i] = (xloc[i] * wloc[i]);};
}

static void _mulpool_04(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 4; i++){yloc[i] = (xloc[i] * wloc[i]);};
}

static void _mulpool_03(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 3; i++){yloc[i] = (xloc[i] * wloc[i]);};
}

static void _mulpool_02(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 2; i++){yloc[i] = (xloc[i] * wloc[i]);};
}

static void _mulpool_01(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 1; i++){yloc[i] = (xloc[i] * wloc[i]);};
}


typedef void (*mulpool_func_t)(const cnn_type_t *, const cnn_type_t *, cnn_type_t *);
static mulpool_func_t _mulpool_tab_512[] = {
   _mulpool_00,
   _mulpool_512,
   _mulpool_1024,
   _mulpool_1536,
};

static mulpool_func_t _mulpool_tab_128[] = {
   _mulpool_00,
   _mulpool_128,
   _mulpool_256,
   _mulpool_384,
};

static mulpool_func_t _mulpool_tab_32[] = {
   _mulpool_00,
   _mulpool_32,
   _mulpool_64,
   _mulpool_96,
};

static mulpool_func_t _mulpool_tab_08[] = {
   _mulpool_00,
   _mulpool_08,
   _mulpool_16,
   _mulpool_24,
};

static mulpool_func_t _mulpool_tab_00[] = {
   _mulpool_00,
   _mulpool_01,
   _mulpool_02,
   _mulpool_03,
   _mulpool_04,
   _mulpool_05,
   _mulpool_06,
   _mulpool_07,
};

void mulpool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++){
         (_mulpool_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         wloc += (1 << 10);
         yloc += (1 << 10);

         (_mulpool_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         wloc += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);
      (_mulpool_tab_512[nb & 3])(xloc, wloc, yloc);

      xloc += (nb << 9);
      wloc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);
      (_mulpool_tab_128[nb])(xloc, wloc, yloc);

      xloc += (nb << 7);
      wloc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);
      (_mulpool_tab_32[nb])(xloc, wloc, yloc);

      xloc += (nb << 5);
      wloc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_mulpool_tab_08[nb])(xloc, wloc, yloc);

      xloc += (nb << 3);
      wloc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (_mulpool_tab_00[_n])(xloc, wloc, yloc);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define mulpool2_08(xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = (xloc[p]) * wloc;};  \
   \
   xloc += CNN_BCHSIZ;  \
   yloc += CNN_BCHSIZ;  \
}

#define mulpool2_16(xloc, wloc, yloc) \
{  \
   mulpool2_08(xloc, wloc, yloc);  \
   mulpool2_08(xloc, wloc, yloc);  \
}

#define mulpool2_32(xloc, wloc, yloc) \
{  \
   mulpool2_16(xloc, wloc, yloc);  \
   mulpool2_16(xloc, wloc, yloc);  \
}

#define mulpool2_64(xloc, wloc, yloc) \
{  \
   mulpool2_32(xloc, wloc, yloc);  \
   mulpool2_32(xloc, wloc, yloc);  \
}

#define mulpool2_128(xloc, wloc, yloc) \
{  \
   mulpool2_64(xloc, wloc, yloc);  \
   mulpool2_64(xloc, wloc, yloc);  \
}

#define mulpool2_256(xloc, wloc, yloc) \
{  \
   mulpool2_128(xloc, wloc, yloc);  \
   mulpool2_128(xloc, wloc, yloc);  \
}

#define mulpool2_512(xloc, wloc, yloc) \
{  \
   mulpool2_256(xloc, wloc, yloc);  \
   mulpool2_256(xloc, wloc, yloc);  \
}

static void _mulpool2_00(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _mulpool2_1536(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_512(xloc, wloc, yloc);
   mulpool2_512(xloc, wloc, yloc);
   mulpool2_512(xloc, wloc, yloc);
}

static void _mulpool2_1024(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_512(xloc, wloc, yloc);
   mulpool2_512(xloc, wloc, yloc);
}

static void _mulpool2_512(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_512(xloc, wloc, yloc);
}

static void _mulpool2_384(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_128(xloc, wloc, yloc);
   mulpool2_128(xloc, wloc, yloc);
   mulpool2_128(xloc, wloc, yloc);
}

static void _mulpool2_256(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_128(xloc, wloc, yloc);
   mulpool2_128(xloc, wloc, yloc);
}

static void _mulpool2_128(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_128(xloc, wloc, yloc);
}

static void _mulpool2_96(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_64(xloc, wloc, yloc);
   mulpool2_32(xloc, wloc, yloc);
}

static void _mulpool2_64(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_64(xloc, wloc, yloc);
}

static void _mulpool2_32(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_32(xloc, wloc, yloc);
}

static void _mulpool2_24(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   mulpool2_16(xloc, wloc, yloc);
   mulpool2_08(xloc, wloc, yloc);
}

static void _mulpool2_16(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_16(xloc, wloc, yloc);
}

static void _mulpool2_08(
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   mulpool2_08(xloc, wloc, yloc);
}

static void _mulpool2_07(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 7; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_06(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 6; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_05(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 5; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_04(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 4; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_03(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 3; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_02(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 2; i++){yloc[i] = (xloc[i] * wloc);};
}

static void _mulpool2_01(
   const cnn_type_t * restrict xloc,
         cnn_type_t            wloc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 1; i++){yloc[i] = (xloc[i] * wloc);};
}


typedef void (*mulpool2_func_t)(const cnn_type_t *, cnn_type_t, cnn_type_t *);
static mulpool2_func_t _mulpool2_tab_512[] = {
   _mulpool2_00,
   _mulpool2_512,
   _mulpool2_1024,
   _mulpool2_1536,
};

static mulpool2_func_t _mulpool2_tab_128[] = {
   _mulpool2_00,
   _mulpool2_128,
   _mulpool2_256,
   _mulpool2_384,
};

static mulpool2_func_t _mulpool2_tab_32[] = {
   _mulpool2_00,
   _mulpool2_32,
   _mulpool2_64,
   _mulpool2_96,
};

static mulpool2_func_t _mulpool2_tab_08[] = {
   _mulpool2_00,
   _mulpool2_08,
   _mulpool2_16,
   _mulpool2_24,
};

static mulpool2_func_t _mulpool2_tab_00[] = {
   _mulpool2_00,
   _mulpool2_01,
   _mulpool2_02,
   _mulpool2_03,
   _mulpool2_04,
   _mulpool2_05,
   _mulpool2_06,
   _mulpool2_07,
};

void mulpool2(
         int                   _n,
   const cnn_type_t * restrict _xloc,
         cnn_type_t            _wloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t   wloc = _wloc;
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++){
         (_mulpool2_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         yloc += (1 << 10);

         (_mulpool2_tab_512[2])(xloc, wloc, yloc);
         xloc += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);
      (_mulpool2_tab_512[nb & 3])(xloc, wloc, yloc);

      xloc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);
      (_mulpool2_tab_128[nb])(xloc, wloc, yloc);

      xloc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);
      (_mulpool2_tab_32[nb])(xloc, wloc, yloc);

      xloc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_mulpool2_tab_08[nb])(xloc, wloc, yloc);

      xloc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (_mulpool2_tab_00[_n])(xloc, wloc, yloc);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define macpool_08(xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] += (xloc[p]) * (wloc[p]);};  \
   \
   xloc += CNN_BCHSIZ;  \
   wloc += CNN_BCHSIZ;  \
}

#define macpool_16(xloc, wloc, yloc) \
{  \
   macpool_08(xloc, wloc, yloc);  \
   macpool_08(xloc, wloc, yloc);  \
}

#define macpool_32(xloc, wloc, yloc) \
{  \
   macpool_16(xloc, wloc, yloc);  \
   macpool_16(xloc, wloc, yloc);  \
}

#define macpool_64(xloc, wloc, yloc) \
{  \
   macpool_32(xloc, wloc, yloc);  \
   macpool_32(xloc, wloc, yloc);  \
}

#define macpool_128(xloc, wloc, yloc) \
{  \
   macpool_64(xloc, wloc, yloc);  \
   macpool_64(xloc, wloc, yloc);  \
}

#define macpool_256(xloc, wloc, yloc) \
{  \
   macpool_128(xloc, wloc, yloc);  \
   macpool_128(xloc, wloc, yloc);  \
}

#define macpool_512(xloc, wloc, yloc) \
{  \
   macpool_256(xloc, wloc, yloc);  \
   macpool_256(xloc, wloc, yloc);  \
}

static void _macpool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   ;
}

#if (0)
static void _macpool_4096(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);

   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_3584(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);

   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_3072(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);

   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_2560(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);

   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
}
#endif

static void _macpool_2048(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_512(xloc, wloc, yloc);
}

static void _macpool_448(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_64 (xloc, wloc, yloc);

   macpool_128(xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
}

static void _macpool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_128(xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
}

static void _macpool_320(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool_64 (xloc, wloc, yloc);

   macpool_128(xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
}

static void _macpool_256(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_128(xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
}

static void _macpool_192(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool_64 (xloc, wloc, yloc);
   macpool_128(xloc, wloc, yloc);
}

static void _macpool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_128(xloc, wloc, yloc);
}

static void _macpool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_64(xloc, wloc, yloc);
}

static void _macpool_56(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool_08(xloc, wloc, yloc);
   macpool_16(xloc, wloc, yloc);
   macpool_32(xloc, wloc, yloc);
}

static void _macpool_48(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool_16(xloc, wloc, yloc);
   macpool_32(xloc, wloc, yloc);
}

static void _macpool_40(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool_08(xloc, wloc, yloc);
   macpool_32(xloc, wloc, yloc);
}

static void _macpool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_32(xloc, wloc, yloc);
}

static void _macpool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_16(xloc, wloc, yloc);
   macpool_08(xloc, wloc, yloc);
}

static void _macpool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_16(xloc, wloc, yloc);
}

static void _macpool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool_08(xloc, wloc, yloc);
}

#if (0)
static void _macpool_07(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 7; p++){yloc[p] += (xloc[p]) * (wloc[p]);};
}

static void _macpool_06(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 6; p++){yloc[p] += (xloc[p]) * (wloc[p]);};
}

static void _macpool_05(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 5; p++){yloc[p] += (xloc[p]) * (wloc[p]);};
}

static void _macpool_04(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 4; p++){yloc[p] += (xloc[p]) * (wloc[p]);};
}

static void _macpool_03(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 3; p++){yloc[p] += (xloc[p]) * (wloc[p]);};
}

static void _macpool_02(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 2; p++){yloc[p] += (xloc[p]) * (wloc[p]);};
}

static void _macpool_01(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 1; p++){yloc[p] += (xloc[p]) * (wloc[p]);};
}
#endif

typedef void (*macpool_func_t)(const cnn_type_t *, const cnn_type_t *, acc_type_t *);
static macpool_func_t _macpool_tab_512[] = {
   _macpool_00,
   _macpool_512,
   _macpool_1024,
   _macpool_1536,
   _macpool_2048,
   // _macpool_2560,
   // _macpool_3072,
   // _macpool_3584,
};

static macpool_func_t _macpool_tab_064[] = {
   _macpool_00,
   _macpool_64,
   _macpool_128,
   _macpool_192,
   _macpool_256,
   _macpool_320,
   _macpool_384,
   _macpool_448,
};

static macpool_func_t _macpool_tab_008[] = {
   _macpool_00,
   _macpool_08,
   _macpool_16,
   _macpool_24,
   _macpool_32,
   _macpool_40,
   _macpool_48,
   _macpool_56,
};

/*
static macpool_func_t _macpool_tab_001[] = {
   _macpool_00,
   _macpool_01,
   _macpool_02,
   _macpool_03,
   _macpool_04,
   _macpool_05,
   _macpool_06,
   _macpool_07,
};
*/

cnn_type_t macpool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);

   acc_type_t yloc[CNN_BCHSIZ] = {0};

   int __ixxx = (_n >> 11);
   int __i512 = (_n >>  9) & (3);
   int __i064 = (_n >>  6) & (7);
   int __i008 = (_n >>  3) & (7);
   // int __i001 = (_n >>  0) & (7);

   for (int i = 0; i < __ixxx; i++)
   {
      _macpool_tab_512[4](xloc, wloc, yloc);
      xloc += 2048;
      wloc += 2048;
   }

   if (__i512)
   {
      _macpool_tab_512[__i512](xloc, wloc, yloc);
      xloc += (__i512 << 9);
      wloc += (__i512 << 9);
   }

   if (__i064)
   {
      _macpool_tab_064[__i064](xloc, wloc, yloc);
      xloc += (__i064 << 6);
      wloc += (__i064 << 6);
   }

   if (__i008)
   {
      _macpool_tab_008[__i008](xloc, wloc, yloc);
      xloc += (__i008 << 3);
      wloc += (__i008 << 3);
   }

   // _macpool_tab_001[__i001](xloc, wloc, yloc);

   {
      for (int p = 0; p < (CNN_BCHSIZ/ 2); p++){yloc[p] += yloc[p + (CNN_BCHSIZ / 2)];};
      for (int p = 0; p < (CNN_BCHSIZ/ 4); p++){yloc[p] += yloc[p + (CNN_BCHSIZ / 4)];};
      for (int p = 0; p < (CNN_BCHSIZ/ 8); p++){yloc[p] += yloc[p + (CNN_BCHSIZ / 8)];};
   }

   return (yloc[0]);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define macpool4_08(_wsiz, xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p                                        ] += (xloc[p]) * (wloc[p                       ]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p + CNN_BCHSIZ                           ] += (xloc[p]) * (wloc[p + _wsiz               ]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p + CNN_BCHSIZ +  CNN_BCHSIZ             ] += (xloc[p]) * (wloc[p + _wsiz + _wsiz       ]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p + CNN_BCHSIZ +  CNN_BCHSIZ + CNN_BCHSIZ] += (xloc[p]) * (wloc[p + _wsiz + _wsiz + _wsiz]);};  \
   \
   xloc += CNN_BCHSIZ;  \
   wloc += CNN_BCHSIZ;  \
}

#define macpool4_16(_wsiz, xloc, wloc, yloc) \
{  \
   macpool4_08(_wsiz, xloc, wloc, yloc);  \
   macpool4_08(_wsiz, xloc, wloc, yloc);  \
}

#define macpool4_32(_wsiz, xloc, wloc, yloc) \
{  \
   macpool4_16(_wsiz, xloc, wloc, yloc);  \
   macpool4_16(_wsiz, xloc, wloc, yloc);  \
}

#define macpool4_64(_wsiz, xloc, wloc, yloc) \
{  \
   macpool4_32(_wsiz, xloc, wloc, yloc);  \
   macpool4_32(_wsiz, xloc, wloc, yloc);  \
}

#define macpool4_128(_wsiz, xloc, wloc, yloc) \
{  \
   macpool4_64(_wsiz, xloc, wloc, yloc);  \
   macpool4_64(_wsiz, xloc, wloc, yloc);  \
}

#define macpool4_256(_wsiz, xloc, wloc, yloc) \
{  \
   macpool4_128(_wsiz, xloc, wloc, yloc);  \
   macpool4_128(_wsiz, xloc, wloc, yloc);  \
}

#define macpool4_512(_wsiz, xloc, wloc, yloc) \
{  \
   macpool4_256(_wsiz, xloc, wloc, yloc);  \
   macpool4_256(_wsiz, xloc, wloc, yloc);  \
}

static void _macpool4_00(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   ;
}

#if (0)
static void _macpool4_4096(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);

   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_3584(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);

   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_3072(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);

   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_2560(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_512(_wsiz, xloc, wloc, yloc);

   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
}
#endif

static void _macpool4_2048(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_1536(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_1024(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_512(_wsiz, xloc, wloc, yloc);
   macpool4_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_512(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_448(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_64 (_wsiz, xloc, wloc, yloc);

   macpool4_128(_wsiz, xloc, wloc, yloc);
   macpool4_128(_wsiz, xloc, wloc, yloc);
   macpool4_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_384(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_128(_wsiz, xloc, wloc, yloc);
   macpool4_128(_wsiz, xloc, wloc, yloc);
   macpool4_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_320(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool4_64 (_wsiz, xloc, wloc, yloc);

   macpool4_128(_wsiz, xloc, wloc, yloc);
   macpool4_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_256(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_128(_wsiz, xloc, wloc, yloc);
   macpool4_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_192(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool4_64 (_wsiz, xloc, wloc, yloc);
   macpool4_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_128(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_64(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_64(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_56(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool4_08(_wsiz, xloc, wloc, yloc);
   macpool4_16(_wsiz, xloc, wloc, yloc);
   macpool4_32(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_48(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool4_16(_wsiz, xloc, wloc, yloc);
   macpool4_32(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_40(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool4_08(_wsiz, xloc, wloc, yloc);
   macpool4_32(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_32(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_32(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_24(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_16(_wsiz, xloc, wloc, yloc);
   macpool4_08(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_16(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_16(_wsiz, xloc, wloc, yloc);
}

static void _macpool4_08(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool4_08(_wsiz, xloc, wloc, yloc);
}

typedef void (*macpool4_func_t)(const int, const cnn_type_t *, const cnn_type_t *, acc_type_t *);
static macpool4_func_t _macpool4_tab_512[] = {
   _macpool4_00,
   _macpool4_512,
   _macpool4_1024,
   _macpool4_1536,
   _macpool4_2048,
   // _macpool4_2560,
   // _macpool4_3072,
   // _macpool4_3584,
};

static macpool4_func_t _macpool4_tab_064[] = {
   _macpool4_00,
   _macpool4_64,
   _macpool4_128,
   _macpool4_192,
   _macpool4_256,
   _macpool4_320,
   _macpool4_384,
   _macpool4_448,
};

static macpool4_func_t _macpool4_tab_008[] = {
   _macpool4_00,
   _macpool4_08,
   _macpool4_16,
   _macpool4_24,
   _macpool4_32,
   _macpool4_40,
   _macpool4_48,
   _macpool4_56,
};

void macpool4(
   const int                   _n,
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _zloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * zloc = __builtin_assume_aligned(_zloc, 16);

   acc_type_t yloc[CNN_BCHSIZ * 4] = {0};

   int __ixxx = (_n >> 11);
   int __i512 = (_n >>  9) & (3);
   int __i064 = (_n >>  6) & (7);
   int __i008 = (_n >>  3) & (7);

   for (int i = 0; i < __ixxx; i++)
   {
      _macpool4_tab_512[4](_wsiz, xloc, wloc, yloc);
      xloc += 2048;
      wloc += 2048;
   }

   if (__i512)
   {
      _macpool4_tab_512[__i512](_wsiz, xloc, wloc, yloc);
      xloc += (__i512 << 9);
      wloc += (__i512 << 9);
   }

   if (__i064)
   {
      _macpool4_tab_064[__i064](_wsiz, xloc, wloc, yloc);
      xloc += (__i064 << 6);
      wloc += (__i064 << 6);
   }

   if (__i008)
   {
      _macpool4_tab_008[__i008](_wsiz, xloc, wloc, yloc);
      xloc += (__i008 << 3);
      wloc += (__i008 << 3);
   }

   // _macpool_tab_001[__i001](xloc, wloc, yloc);

   {
      zloc[0] = (yloc[0] + 
                 yloc[1] + 
                 yloc[2] + 
                 yloc[3] + 
                 yloc[4] + 
                 yloc[5] + 
                 yloc[6] + 
                 yloc[7] );

      zloc[1] = (yloc[CNN_BCHSIZ + 0] + 
                 yloc[CNN_BCHSIZ + 1] + 
                 yloc[CNN_BCHSIZ + 2] + 
                 yloc[CNN_BCHSIZ + 3] + 
                 yloc[CNN_BCHSIZ + 4] + 
                 yloc[CNN_BCHSIZ + 5] + 
                 yloc[CNN_BCHSIZ + 6] + 
                 yloc[CNN_BCHSIZ + 7] );

      zloc[2] = (yloc[CNN_BCHSIZ + CNN_BCHSIZ + 0] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 1] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 2] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 3] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 4] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 5] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 6] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 7] );

      zloc[3] = (yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 0] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 1] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 2] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 3] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 4] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 5] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 6] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 7] );
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define macpool8_08(_wsiz, xloc, wloc, yloc)  \
{  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p                                                   ] += (xloc[p]) * (wloc[p                                       ]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p + CNN_BCHSIZ                                      ] += (xloc[p]) * (wloc[p + _wsiz                               ]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p + CNN_BCHSIZ +  CNN_BCHSIZ                        ] += (xloc[p]) * (wloc[p + _wsiz + _wsiz                       ]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p + CNN_BCHSIZ +  CNN_BCHSIZ + CNN_BCHSIZ           ] += (xloc[p]) * (wloc[p + _wsiz + _wsiz + _wsiz               ]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p                                         + BCHSIZX4] += (xloc[p]) * (wloc[p                         + (_wsiz << 2)]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p + CNN_BCHSIZ                            + BCHSIZX4] += (xloc[p]) * (wloc[p + _wsiz                 + (_wsiz << 2)]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p + CNN_BCHSIZ +  CNN_BCHSIZ              + BCHSIZX4] += (xloc[p]) * (wloc[p + _wsiz + _wsiz         + (_wsiz << 2)]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p + CNN_BCHSIZ +  CNN_BCHSIZ + CNN_BCHSIZ + BCHSIZX4] += (xloc[p]) * (wloc[p + _wsiz + _wsiz + _wsiz + (_wsiz << 2)]);};  \
   \
   xloc += CNN_BCHSIZ;  \
   wloc += CNN_BCHSIZ;  \
}

#define macpool8_16(_wsiz, xloc, wloc, yloc) \
{  \
   macpool8_08(_wsiz, xloc, wloc, yloc);  \
   macpool8_08(_wsiz, xloc, wloc, yloc);  \
}

#define macpool8_32(_wsiz, xloc, wloc, yloc) \
{  \
   macpool8_16(_wsiz, xloc, wloc, yloc);  \
   macpool8_16(_wsiz, xloc, wloc, yloc);  \
}

#define macpool8_64(_wsiz, xloc, wloc, yloc) \
{  \
   macpool8_32(_wsiz, xloc, wloc, yloc);  \
   macpool8_32(_wsiz, xloc, wloc, yloc);  \
}

#define macpool8_128(_wsiz, xloc, wloc, yloc) \
{  \
   macpool8_64(_wsiz, xloc, wloc, yloc);  \
   macpool8_64(_wsiz, xloc, wloc, yloc);  \
}

#define macpool8_256(_wsiz, xloc, wloc, yloc) \
{  \
   macpool8_128(_wsiz, xloc, wloc, yloc);  \
   macpool8_128(_wsiz, xloc, wloc, yloc);  \
}

#define macpool8_512(_wsiz, xloc, wloc, yloc) \
{  \
   macpool8_256(_wsiz, xloc, wloc, yloc);  \
   macpool8_256(_wsiz, xloc, wloc, yloc);  \
}

static void _macpool8_00(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   ;
}

#if (0)
static void _macpool8_4096(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);

   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_3584(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);

   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_3072(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);

   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_2560(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_512(_wsiz, xloc, wloc, yloc);

   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
}
#endif

static void _macpool8_2048(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_1536(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_1024(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_512(_wsiz, xloc, wloc, yloc);
   macpool8_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_512(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_512(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_448(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_64 (_wsiz, xloc, wloc, yloc);

   macpool8_128(_wsiz, xloc, wloc, yloc);
   macpool8_128(_wsiz, xloc, wloc, yloc);
   macpool8_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_384(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_128(_wsiz, xloc, wloc, yloc);
   macpool8_128(_wsiz, xloc, wloc, yloc);
   macpool8_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_320(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool8_64 (_wsiz, xloc, wloc, yloc);

   macpool8_128(_wsiz, xloc, wloc, yloc);
   macpool8_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_256(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_128(_wsiz, xloc, wloc, yloc);
   macpool8_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_192(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool8_64 (_wsiz, xloc, wloc, yloc);
   macpool8_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_128(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_128(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_64(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_64(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_56(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool8_08(_wsiz, xloc, wloc, yloc);
   macpool8_16(_wsiz, xloc, wloc, yloc);
   macpool8_32(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_48(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool8_16(_wsiz, xloc, wloc, yloc);
   macpool8_32(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_40(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   macpool8_08(_wsiz, xloc, wloc, yloc);
   macpool8_32(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_32(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_32(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_24(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_16(_wsiz, xloc, wloc, yloc);
   macpool8_08(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_16(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_16(_wsiz, xloc, wloc, yloc);
}

static void _macpool8_08(
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         acc_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   acc_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   macpool8_08(_wsiz, xloc, wloc, yloc);
}

typedef void (*macpool8_func_t)(const int, const cnn_type_t *, const cnn_type_t *, acc_type_t *);
static macpool8_func_t _macpool8_tab_512[] = {
   _macpool8_00,
   _macpool8_512,
   _macpool8_1024,
   _macpool8_1536,
   _macpool8_2048,
   // _macpool8_2560,
   // _macpool8_3072,
   // _macpool8_3584,
};

static macpool8_func_t _macpool8_tab_064[] = {
   _macpool8_00,
   _macpool8_64,
   _macpool8_128,
   _macpool8_192,
   _macpool8_256,
   _macpool8_320,
   _macpool8_384,
   _macpool8_448,
};

static macpool8_func_t _macpool8_tab_008[] = {
   _macpool8_00,
   _macpool8_08,
   _macpool8_16,
   _macpool8_24,
   _macpool8_32,
   _macpool8_40,
   _macpool8_48,
   _macpool8_56,
};

void macpool8(
   const int                   _n,
   const int                   _wsiz,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _zloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * zloc = __builtin_assume_aligned(_zloc, 16);

   acc_type_t yloc[CNN_BCHSIZ * 8] = {0};

   int __ixxx = (_n >> 11);
   int __i512 = (_n >>  9) & (3);
   int __i064 = (_n >>  6) & (7);
   int __i008 = (_n >>  3) & (7);

   for (int i = 0; i < __ixxx; i++)
   {
      _macpool8_tab_512[4](_wsiz, xloc, wloc, yloc);
      xloc += 2048;
      wloc += 2048;
   }

   if (__i512)
   {
      _macpool8_tab_512[__i512](_wsiz, xloc, wloc, yloc);
      xloc += (__i512 << 9);
      wloc += (__i512 << 9);
   }

   if (__i064)
   {
      _macpool8_tab_064[__i064](_wsiz, xloc, wloc, yloc);
      xloc += (__i064 << 6);
      wloc += (__i064 << 6);
   }

   if (__i008)
   {
      _macpool8_tab_008[__i008](_wsiz, xloc, wloc, yloc);
      xloc += (__i008 << 3);
      wloc += (__i008 << 3);
   }

   // _macpool_tab_001[__i001](xloc, wloc, yloc);

   {
      int B4 = CNN_BCHSIZ * 4;

      zloc[0] = (yloc[0] + 
                 yloc[1] + 
                 yloc[2] + 
                 yloc[3] + 
                 yloc[4] + 
                 yloc[5] + 
                 yloc[6] + 
                 yloc[7] );

      zloc[1] = (yloc[CNN_BCHSIZ + 0] + 
                 yloc[CNN_BCHSIZ + 1] + 
                 yloc[CNN_BCHSIZ + 2] + 
                 yloc[CNN_BCHSIZ + 3] + 
                 yloc[CNN_BCHSIZ + 4] + 
                 yloc[CNN_BCHSIZ + 5] + 
                 yloc[CNN_BCHSIZ + 6] + 
                 yloc[CNN_BCHSIZ + 7] );

      zloc[2] = (yloc[CNN_BCHSIZ + CNN_BCHSIZ + 0] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 1] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 2] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 3] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 4] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 5] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 6] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + 7] );

      zloc[3] = (yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 0] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 1] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 2] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 3] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 4] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 5] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 6] + 
                 yloc[CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 7] );

      zloc[4] = (yloc[B4 + 0] + 
                 yloc[B4 + 1] + 
                 yloc[B4 + 2] + 
                 yloc[B4 + 3] + 
                 yloc[B4 + 4] + 
                 yloc[B4 + 5] + 
                 yloc[B4 + 6] + 
                 yloc[B4 + 7] );

      zloc[5] = (yloc[B4 + CNN_BCHSIZ + 0] + 
                 yloc[B4 + CNN_BCHSIZ + 1] + 
                 yloc[B4 + CNN_BCHSIZ + 2] + 
                 yloc[B4 + CNN_BCHSIZ + 3] + 
                 yloc[B4 + CNN_BCHSIZ + 4] + 
                 yloc[B4 + CNN_BCHSIZ + 5] + 
                 yloc[B4 + CNN_BCHSIZ + 6] + 
                 yloc[B4 + CNN_BCHSIZ + 7] );

      zloc[6] = (yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + 0] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + 1] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + 2] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + 3] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + 4] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + 5] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + 6] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + 7] );

      zloc[7] = (yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 0] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 1] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 2] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 3] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 4] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 5] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 6] + 
                 yloc[B4 + CNN_BCHSIZ + CNN_BCHSIZ + CNN_BCHSIZ + 7] );
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* conv3x3 - (stride = 1, valid) */
void conv3x3(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   int _Ln_0 = C1 * 0 * N;
   int _Ln_1 = C1 * 1 * N;
   int _Ln_2 = C1 * 2 * N;

   int C1_0  = C1 * 0;
   int C1_3  = C1 * 3;
   int C1_6  = C1 * 6;
   int C1_9  = C1 * 9;

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);

   for (int m = 0; m < (M-3+1); m++)
   {
      cnn_type_t * px = _px + m * (N-0) * C1;
      cnn_type_t * py = _py + m * (N-2) * C2;

      for (int n = 0; n < (N-3+1); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1, wloc + C1_3);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2, wloc + C1_6);

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }
   }
}

/* conv3x3 - (stride = 2, valid) */
void conv3x3_s2(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   int _Ln_0 = C1 * 0 * N;
   int _Ln_1 = C1 * 1 * N;
   int _Ln_2 = C1 * 2 * N;

   int C1_0  = C1 * 0;
   int C1_3  = C1 * 3;
   int C1_6  = C1 * 6;
   int C1_9  = C1 * 9;

   int N2 = ((N - 3) / 2) + 1;

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);

   for (int m = 0; m < M - 3 + 1; m += 2)
   {
      cnn_type_t * px = _px + (m/1) * (N ) * C1;
      cnn_type_t * py = _py + (m/2) * (N2) * C2;

      for (int n = 0; n < N - 3 + 1; n += 2) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1, wloc + C1_3);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2, wloc + C1_6);

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += (C1 + C1);
      }
   }
}

/* conv3x3 (stride = 1, same) */
void conv3x3_p1(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   int _Ln_0 = C1 * 0 * N;
   int _Ln_1 = C1 * 1 * N;
   int _Ln_2 = C1 * 2 * N;

   int _Lx_0 = C1 * 0;
   int _Lx_1 = C1 * 1;
   // int _Lx_2 = C1 * 2;

   int C1_0  = C1 * 0;
   int C1_1  = C1 * 1;
   int C1_2  = C1 * 2;
   int C1_3  = C1 * 3;
   int C1_4  = C1 * 4;
   int C1_6  = C1 * 6;
   int C1_7  = C1 * 7;
   int C1_9  = C1 * 9;

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);

   {
      int m = 0;

      cnn_type_t * px = _px + (m-1) * N * C1 - C1;
      cnn_type_t * py = _py + (m-0) * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = macpool(C1_2, px + _Ln_1 + _Lx_1, wloc + C1_4);
            cnn_type_t yc = macpool(C1_2, px + _Ln_2 + _Lx_1, wloc + C1_7);

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }

      for (int n = 1; n < (N-1); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = macpool(C1_3, px + _Ln_1 + _Lx_0, wloc + C1_3);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2 + _Lx_0, wloc + C1_6);

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = macpool(C1_2, px + _Ln_1 + _Lx_0, wloc + C1_3);
            cnn_type_t yc = macpool(C1_2, px + _Ln_2 + _Lx_0, wloc + C1_6);

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }
   }

   for (int m = 1; m < (M-1); m++)
   {
      cnn_type_t * px = _px + (m-1) * N * C1 - C1;
      cnn_type_t * py = _py + (m-0) * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_2, px + _Ln_0 + _Lx_1, wloc + C1_1);
            cnn_type_t yb = macpool(C1_2, px + _Ln_1 + _Lx_1, wloc + C1_4);
            cnn_type_t yc = macpool(C1_2, px + _Ln_2 + _Lx_1, wloc + C1_7);

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }

      for (int n = 1; n < (N-1); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0 + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1 + _Lx_0, wloc + C1_3);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2 + _Lx_0, wloc + C1_6);

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_2, px + _Ln_0 + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_2, px + _Ln_1 + _Lx_0, wloc + C1_3);
            cnn_type_t yc = macpool(C1_2, px + _Ln_2 + _Lx_0, wloc + C1_6);

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }
   }

   /* M - 1 */
   {
      int m = (M-1);

      cnn_type_t * px = _px + (m-1) * N * C1 - C1;
      cnn_type_t * py = _py + (m-0) * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_2, px + _Ln_0 + _Lx_1, wloc + C1_1);
            cnn_type_t yb = macpool(C1_2, px + _Ln_1 + _Lx_1, wloc + C1_4);
            cnn_type_t yc = 0;

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }

      for (int n = 1; n < (N-1); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0 + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1 + _Lx_0, wloc + C1_3);
            cnn_type_t yc = 0;

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_2, px + _Ln_0 + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_2, px + _Ln_1 + _Lx_0, wloc + C1_3);
            cnn_type_t yc = 0;

            *py++ = (ya + yb + yc); wloc += C1_9;
         }

         px += C1;
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* conv5x5 (stride = 1, same) */
void conv5x5_p2(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   int _Ln_0 = C1 * 0 * N;
   int _Ln_1 = C1 * 1 * N;
   int _Ln_2 = C1 * 2 * N;
   int _Ln_3 = C1 * 3 * N;
   int _Ln_4 = C1 * 4 * N;

   // int _Lx_0 = C1 * 0;
   int _Lx_1 = C1 * 1;
   int _Lx_2 = C1 * 2;

   int C1_0  = C1 * 0;
   int C1_1  = C1 * 1;
   int C1_2  = C1_1 + C1;
   int C1_3  = C1_2 + C1;
   int C1_4  = C1_3 + C1;
   int C1_5  = C1_4 + C1;
   int C1_6  = C1_5 + C1;
   int C1_7  = C1_6 + C1;

   int C1_10 = C1 * 10;
   int C1_15 = C1 * 15;
   int C1_20 = C1 * 20;

   int C1_12 = C1 * 12;
   int C1_17 = C1 * 17;
   int C1_22 = C1 * 22;

   int C1_11 = C1 * 11;
   int C1_16 = C1 * 16;
   int C1_21 = C1 * 21;

   int C1_25 = C1 * 25;

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);
#if 1
   {
      int m = 0;

      cnn_type_t * px = _px + (m-2) * N * C1 - C1 * 2;
      cnn_type_t * py = _py + (m-0) * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = 0;
            cnn_type_t yc = macpool(C1_3, px + _Ln_2 + _Lx_2, wloc + C1_12);
            cnn_type_t yd = macpool(C1_3, px + _Ln_3 + _Lx_2, wloc + C1_17);
            cnn_type_t ye = macpool(C1_3, px + _Ln_4 + _Lx_2, wloc + C1_22);

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = 0;
            cnn_type_t yc = macpool(C1_4, px + _Ln_2 + _Lx_1, wloc + C1_11);
            cnn_type_t yd = macpool(C1_4, px + _Ln_3 + _Lx_1, wloc + C1_16);
            cnn_type_t ye = macpool(C1_4, px + _Ln_4 + _Lx_1, wloc + C1_21);

            *py++ = (ya + yb + yc + yd + ye);
            wloc += C1_25;
         }

         px += C1;
      }

      for (int n = 2; n < (N-2); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = 0;
            cnn_type_t yc = macpool(C1_5, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_5, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = macpool(C1_5, px + _Ln_4, wloc + C1_20);

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = 0;
            cnn_type_t yc = macpool(C1_4, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_4, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = macpool(C1_4, px + _Ln_4, wloc + C1_20);

            *py++ = (ya + yb + yc + yd + ye);
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = 0;
            cnn_type_t yc = macpool(C1_3, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_3, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = macpool(C1_3, px + _Ln_4, wloc + C1_20);

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }
   }

   {
      int m = 1;

      cnn_type_t * px = _px + (m-2) * N * C1 - C1 * 2;
      cnn_type_t * py = _py + (m-0) * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = macpool(C1_3, px + _Ln_1 + _Lx_2, wloc + C1_7);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2 + _Lx_2, wloc + C1_12);
            cnn_type_t yd = macpool(C1_3, px + _Ln_3 + _Lx_2, wloc + C1_17);
            cnn_type_t ye = macpool(C1_3, px + _Ln_4 + _Lx_2, wloc + C1_22);

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = macpool(C1_4, px + _Ln_1 + _Lx_1, wloc + C1_6);
            cnn_type_t yc = macpool(C1_4, px + _Ln_2 + _Lx_1, wloc + C1_11);
            cnn_type_t yd = macpool(C1_4, px + _Ln_3 + _Lx_1, wloc + C1_16);
            cnn_type_t ye = macpool(C1_4, px + _Ln_4 + _Lx_1, wloc + C1_21);

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      for (int n = 2; n < (N-2); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = macpool(C1_5, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_5, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_5, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = macpool(C1_5, px + _Ln_4, wloc + C1_20);

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = macpool(C1_4, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_4, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_4, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = macpool(C1_4, px + _Ln_4, wloc + C1_20);

            *py++ = (ya + yb + yc + yd + ye);
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = 0;
            cnn_type_t yb = macpool(C1_3, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_3, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = macpool(C1_3, px + _Ln_4, wloc + C1_20);

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }
   }
#endif
   for (int m = 2; m < (M-2); m++)
   {
      cnn_type_t * px = _px + (m-2) * N * C1 - C1 * 2;
      cnn_type_t * py = _py + (m-0) * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0 + _Lx_2, wloc + C1_2);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1 + _Lx_2, wloc + C1_7);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2 + _Lx_2, wloc + C1_12);
            cnn_type_t yd = macpool(C1_3, px + _Ln_3 + _Lx_2, wloc + C1_17);
            cnn_type_t ye = macpool(C1_3, px + _Ln_4 + _Lx_2, wloc + C1_22);

            *py++ = (ya + yb + yc + yd + ye); wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_4, px + _Ln_0 + _Lx_1, wloc + C1_1);
            cnn_type_t yb = macpool(C1_4, px + _Ln_1 + _Lx_1, wloc + C1_6);
            cnn_type_t yc = macpool(C1_4, px + _Ln_2 + _Lx_1, wloc + C1_11);
            cnn_type_t yd = macpool(C1_4, px + _Ln_3 + _Lx_1, wloc + C1_16);
            cnn_type_t ye = macpool(C1_4, px + _Ln_4 + _Lx_1, wloc + C1_21);

            *py++ = (ya + yb + yc + yd + ye); wloc += C1_25;
         }

         px += C1;
      }

      for (int n = 2; n < (N-2); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_5, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_5, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_5, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_5, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = macpool(C1_5, px + _Ln_4, wloc + C1_20);

            *py++ = (ya + yb + yc + yd + ye); wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_4, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_4, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_4, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_4, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = macpool(C1_4, px + _Ln_4, wloc + C1_20);

            *py++ = (ya + yb + yc + yd + ye); wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_3, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = macpool(C1_3, px + _Ln_4, wloc + C1_20);

            *py++ = (ya + yb + yc + yd + ye); wloc += C1_25;
         }

         px += C1;
      }
   }
#if 1
   {
      int m = M - 2;

      cnn_type_t * px = _px + (m-2) * N * C1 - C1 * 2;
      cnn_type_t * py = _py + (m-0) * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0 + _Lx_2, wloc + C1_2);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1 + _Lx_2, wloc + C1_7);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2 + _Lx_2, wloc + C1_12);
            cnn_type_t yd = macpool(C1_3, px + _Ln_3 + _Lx_2, wloc + C1_17);
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye);
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_4, px + _Ln_0 + _Lx_1, wloc + C1_1);
            cnn_type_t yb = macpool(C1_4, px + _Ln_1 + _Lx_1, wloc + C1_6);
            cnn_type_t yc = macpool(C1_4, px + _Ln_2 + _Lx_1, wloc + C1_11);
            cnn_type_t yd = macpool(C1_4, px + _Ln_3 + _Lx_1, wloc + C1_16);
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye);
            wloc += C1_25;
         }

         px += C1;
      }

      for (int n = 2; n < (N-2); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_5, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_5, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_5, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_5, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_4, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_4, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_4, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_4, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = macpool(C1_3, px + _Ln_3, wloc + C1_15);
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }
   }

   {
      int m = M - 1;

      cnn_type_t * px = _px + (m-2) * N * C1 - C1 * 2;
      cnn_type_t * py = _py + (m-0) * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0 + _Lx_2, wloc + C1_2);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1 + _Lx_2, wloc + C1_7);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2 + _Lx_2, wloc + C1_12);
            cnn_type_t yd = 0;
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_4, px + _Ln_0 + _Lx_1, wloc + C1_1);
            cnn_type_t yb = macpool(C1_4, px + _Ln_1 + _Lx_1, wloc + C1_6);
            cnn_type_t yc = macpool(C1_4, px + _Ln_2 + _Lx_1, wloc + C1_11);
            cnn_type_t yd = 0;
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      for (int n = 2; n < (N-2); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_5, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_5, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_5, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = 0;
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_4, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_4, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_4, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = 0;
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye); 
            wloc += C1_25;
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1_3, px + _Ln_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1_3, px + _Ln_1, wloc + C1_5);
            cnn_type_t yc = macpool(C1_3, px + _Ln_2, wloc + C1_10);
            cnn_type_t yd = 0;
            cnn_type_t ye = 0;

            *py++ = (ya + yb + yc + yd + ye);
            wloc += C1_25;
         }

         px += C1;
      }
   }
   #endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* conv1x3 padding = 1 */
void conv1x3_p1(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);

   for (int m = 0; m < M; m++)
   {
      cnn_type_t * px = _px + m * N * C1;
      cnn_type_t * py = _py + m * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 2 * C1, px, wloc + C1); wloc += (1 * 3 * C1);
         }
      }

      for (int n = 1; n < (N-1); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 3 * C1, px, wloc); wloc += (1 * 3 * C1);
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 2 * C1, px, wloc); wloc += (1 * 3 * C1);
         }

         px += C1;
      }
   }
}

/* conv3x1 padding = 1 */
void conv3x1_p1(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   int _Lx_0 = C1 * 0 * N;
   int _Lx_1 = C1 * 1 * N;
   int _Lx_2 = C1 * 2 * N;
   // int _Lx_3 = C1 * 3 * N;
   // int _Lx_4 = C1 * 4 * N;
   // int _Lx_5 = C1 * 5 * N;
   // int _Lx_6 = C1 * 6 * N;

   int C1_0  = C1 * 0;
   int C1_1  = C1 * 1;
   int C1_2  = C1 * 2;
   int C1_3  = C1 * 3;
   // int C1_4  = C1 * 4;
   // int C1_5  = C1 * 5;
   // int C1_6  = C1 * 6;
   // int C1_7  = C1 * 7;

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);

   {
      int m = 0;

      cnn_type_t * px = _px + 0 * N * C1;
      cnn_type_t * py = _py + m * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_1);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_2);

            *py++ = (ya + yb); wloc += C1_3;
         }

         px += C1;
      }
   }

   for (int m = 1; m < (M - 1); m++)
   {
      cnn_type_t * px = _px + (m - 1) * N * C1;
      cnn_type_t * py = _py + (m - 0) * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_1);
            cnn_type_t yc = macpool(C1, px + _Lx_2, wloc + C1_2);

            *py++ = (ya + yb + yc); wloc += C1_3;
         }

         px += C1;
      }
   }

   {
      int m = M - 1;

      cnn_type_t * px = _px + (m - 1) * N * C1;
      cnn_type_t * py = _py + (m - 0) * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_1);

            *py++ = (ya + yb); wloc += C1_3;
         }

         px += C1;
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* conv1x7 padding = 3 */
void conv1x7_p3(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);

   for (int m = 0; m < M; m++)
   {
      cnn_type_t * px = _px + m * N * C1;
      cnn_type_t * py = _py + m * N * C2;

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 4 * C1, px, wloc + (3 * C1)); wloc += (1 * 7 * C1);
         }
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 5 * C1, px, wloc + (2 * C1)); wloc += (1 * 7 * C1);
         }
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 6 * C1, px, wloc + (1 * C1)); wloc += (1 * 7 * C1);
         }
      }

      for (int n = 3; n < (N-3); n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 7 * C1, px, wloc); wloc += (1 * 7 * C1);
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 6 * C1, px, wloc); wloc += (1 * 7 * C1);
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 5 * C1, px, wloc); wloc += (1 * 7 * C1);
         }

         px += C1;
      }

      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            *py++ = macpool(1 * 4 * C1, px, wloc); wloc += (1 * 7 * C1);
         }

         px += C1;
      }
   }
}

/* conv7x1 padding = 3 */
void conv7x1_p3(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   int _Lx_0 = C1 * 0 * N;
   int _Lx_1 = C1 * 1 * N;
   int _Lx_2 = C1 * 2 * N;
   int _Lx_3 = C1 * 3 * N;
   int _Lx_4 = C1 * 4 * N;
   int _Lx_5 = C1 * 5 * N;
   int _Lx_6 = C1 * 6 * N;

   int C1_0  = C1 * 0;
   int C1_1  = C1 * 1;
   int C1_2  = C1 * 2;
   int C1_3  = C1 * 3;
   int C1_4  = C1 * 4;
   int C1_5  = C1 * 5;
   int C1_6  = C1 * 6;
   int C1_7  = C1 * 7;

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);

   {
      int m = 0;

      cnn_type_t * px = _px + 0 * N * C1;
      cnn_type_t * py = _py + m * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_3);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_4);
            cnn_type_t yc = macpool(C1, px + _Lx_2, wloc + C1_5);
            cnn_type_t yd = macpool(C1, px + _Lx_3, wloc + C1_6);

            *py++ = (ya + yb + yc + yd); wloc += C1_7;
         }

         px += C1;
      }
   }

   {
      int m = 1;

      cnn_type_t * px = _px + 0 * N * C1;
      cnn_type_t * py = _py + m * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_2);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_3);
            cnn_type_t yc = macpool(C1, px + _Lx_2, wloc + C1_4);
            cnn_type_t yd = macpool(C1, px + _Lx_3, wloc + C1_5);
            cnn_type_t ye = macpool(C1, px + _Lx_4, wloc + C1_6);

            *py++ = (ya + yb + yc + yd + ye); wloc += C1_7;
         }

         px += C1;
      }
   }

   {
      int m = 2;

      cnn_type_t * px = _px + 0 * N * C1;
      cnn_type_t * py = _py + m * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_1);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_2);
            cnn_type_t yc = macpool(C1, px + _Lx_2, wloc + C1_3);
            cnn_type_t yd = macpool(C1, px + _Lx_3, wloc + C1_4);
            cnn_type_t ye = macpool(C1, px + _Lx_4, wloc + C1_5);
            cnn_type_t yf = macpool(C1, px + _Lx_5, wloc + C1_6);

            *py++ = (ya + yb + yc + yd + ye + yf); wloc += C1_7;
         }

         px += C1;
      }
   }

   for (int m = 3; m < (M - 3); m++)
   {
      cnn_type_t * px = _px + (m - 3) * N * C1;
      cnn_type_t * py = _py + (m - 0) * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_1);
            cnn_type_t yc = macpool(C1, px + _Lx_2, wloc + C1_2);
            cnn_type_t yd = macpool(C1, px + _Lx_3, wloc + C1_3);
            cnn_type_t ye = macpool(C1, px + _Lx_4, wloc + C1_4);
            cnn_type_t yf = macpool(C1, px + _Lx_5, wloc + C1_5);
            cnn_type_t yg = macpool(C1, px + _Lx_6, wloc + C1_6);

            *py++ = (ya + yb + yc + yd + ye + yf + yg); wloc += C1_7;
         }

         px += C1;
      }
   }

   {
      int m = M - 3;

      cnn_type_t * px = _px + (m - 3) * N * C1;
      cnn_type_t * py = _py + (m - 0) * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_1);
            cnn_type_t yc = macpool(C1, px + _Lx_2, wloc + C1_2);
            cnn_type_t yd = macpool(C1, px + _Lx_3, wloc + C1_3);
            cnn_type_t ye = macpool(C1, px + _Lx_4, wloc + C1_4);
            cnn_type_t yf = macpool(C1, px + _Lx_5, wloc + C1_5);

            *py++ = (ya + yb + yc + yd + ye + yf); wloc += C1_7;
         }

         px += C1;
      }
   }

   {
      int m = M - 2;

      cnn_type_t * px = _px + (m - 3) * N * C1;
      cnn_type_t * py = _py + (m - 0) * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_1);
            cnn_type_t yc = macpool(C1, px + _Lx_2, wloc + C1_2);
            cnn_type_t yd = macpool(C1, px + _Lx_3, wloc + C1_3);
            cnn_type_t ye = macpool(C1, px + _Lx_4, wloc + C1_4);

            *py++ = (ya + yb + yc + yd + ye); wloc += C1_7;
         }

         px += C1;
      }
   }

   {
      int m = M - 1;

      cnn_type_t * px = _px + (m - 3) * N * C1;
      cnn_type_t * py = _py + (m - 0) * N * C2;

      for (int n = 0; n < N; n++) 
      {
         cnn_type_t * wloc = _pw;

         for (int c = 0; c < C2; c++)
         {
            cnn_type_t ya = macpool(C1, px + _Lx_0, wloc + C1_0);
            cnn_type_t yb = macpool(C1, px + _Lx_1, wloc + C1_1);
            cnn_type_t yc = macpool(C1, px + _Lx_2, wloc + C1_2);
            cnn_type_t yd = macpool(C1, px + _Lx_3, wloc + C1_3);

            *py++ = (ya + yb + yc + yd); wloc += C1_7;
         }

         px += C1;
      }
   }
}

void conv1x1(
         int  M,
         int  N,
         int  C1,
         int  C2,
   const cnn_type_t * restrict __px,
   const cnn_type_t * restrict __pw,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);
   cnn_type_t * _pw = __builtin_assume_aligned(__pw, 16);

   tf_assert((C1 & (CNN_BCHSIZ - 1)) == 0);

   for (int m = 0; m < M - 1 + 1; m++)
   {
      cnn_type_t * px = _px + m * (N-0) * C1;
      cnn_type_t * py = _py + m * (N-0) * C2;

      for (int n = 0; n < (N-1+1); n++)
      {
         {
            cnn_type_t * wloc = _pw;
            for (int c = 0; c < C2; c++)
            {
               *py++ = macpool(C1, px, wloc); wloc += (C1);
            }
         }

         px += C1;
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void fullnet(
         int                   M,
         int                   N,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _wloc,
         cnn_type_t * restrict _yloc   
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * wloc = __builtin_assume_aligned(_wloc, 16);
   cnn_type_t * yloc = _yloc;

   for (int m = 0; m < M; m++)
   {
      *yloc++ = macpool(N, xloc, wloc);

      wloc += N;
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define bnormpool_08(xloc, uloc, varn, beta, yloc)   \
{  \
   cnn_type_t zloc[CNN_BCHSIZ];          \
   cnn_type_t dloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      zloc[p] = xloc[p] - uloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      dloc[p] = zloc[p] * varn[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = dloc[p] + beta[p];       \
   }                                     \
                                         \
   xloc += CNN_BCHSIZ;                   \
   uloc += CNN_BCHSIZ;                   \
   varn += CNN_BCHSIZ;                   \
   beta += CNN_BCHSIZ;                   \
   yloc += CNN_BCHSIZ;                   \
}

#define bnormpool_16(xloc, uloc, varn, beta, yloc) \
{  \
   bnormpool_08(xloc, uloc, varn, beta, yloc);  \
   bnormpool_08(xloc, uloc, varn, beta, yloc);  \
}

#define bnormpool_32(xloc, uloc, varn, beta, yloc) \
{  \
   bnormpool_16(xloc, uloc, varn, beta, yloc);  \
   bnormpool_16(xloc, uloc, varn, beta, yloc);  \
}

#define bnormpool_64(xloc, uloc, varn, beta, yloc) \
{  \
   bnormpool_32(xloc, uloc, varn, beta, yloc);  \
   bnormpool_32(xloc, uloc, varn, beta, yloc);  \
}

#define bnormpool_128(xloc, uloc, varn, beta, yloc) \
{  \
   bnormpool_64(xloc, uloc, varn, beta, yloc);  \
   bnormpool_64(xloc, uloc, varn, beta, yloc);  \
}

#define bnormpool_256(xloc, uloc, varn, beta, yloc) \
{  \
   bnormpool_128(xloc, uloc, varn, beta, yloc);  \
   bnormpool_128(xloc, uloc, varn, beta, yloc);  \
}

#define bnormpool_512(xloc, uloc, varn, beta, yloc) \
{  \
   bnormpool_256(xloc, uloc, varn, beta, yloc);  \
   bnormpool_256(xloc, uloc, varn, beta, yloc);  \
}

static void _bnormpool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _bnormpool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   bnormpool_512(xloc, uloc, varn, beta, yloc);
   bnormpool_512(xloc, uloc, varn, beta, yloc);
   bnormpool_512(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_512(xloc, uloc, varn, beta, yloc);
   bnormpool_512(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_512(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_128(xloc, uloc, varn, beta, yloc);
   bnormpool_128(xloc, uloc, varn, beta, yloc);
   bnormpool_128(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_256(  
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_128(xloc, uloc, varn, beta, yloc);
   bnormpool_128(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_128(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_64(xloc, uloc, varn, beta, yloc);
   bnormpool_32(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_64(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_32(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_16(xloc, uloc, varn, beta, yloc);
   bnormpool_08(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_16(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnormpool_08(xloc, uloc, varn, beta, yloc);
}

static void _bnormpool_07(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 7; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
   }
}

static void _bnormpool_06(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 6; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
   }
}

static void _bnormpool_05(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 5; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
   }
}

static void _bnormpool_04(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 4; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
   }
}

static void _bnormpool_03(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 3; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
   }
}

static void _bnormpool_02(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 2; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
   }
}

static void _bnormpool_01(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 1; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
   }
}

typedef void (*bnormpool_func_t)(
   const cnn_type_t *, 
   const cnn_type_t *, 
   const cnn_type_t *, 
   const cnn_type_t *, 
         cnn_type_t *
);
static bnormpool_func_t _bnormpool_tab_512[] = {
   _bnormpool_00,
   _bnormpool_512,
   _bnormpool_1024,
   _bnormpool_1536,
};

static bnormpool_func_t _bnormpool_tab_128[] = {
   _bnormpool_00,
   _bnormpool_128,
   _bnormpool_256,
   _bnormpool_384,
};

static bnormpool_func_t _bnormpool_tab_32[] = {
   _bnormpool_00,
   _bnormpool_32,
   _bnormpool_64,
   _bnormpool_96,
};

static bnormpool_func_t _bnormpool_tab_08[] = {
   _bnormpool_00,
   _bnormpool_08,
   _bnormpool_16,
   _bnormpool_24,
};

static bnormpool_func_t _bnormpool_tab_00[] = {
   _bnormpool_00,
   _bnormpool_01,
   _bnormpool_02,
   _bnormpool_03,
   _bnormpool_04,
   _bnormpool_05,
   _bnormpool_06,
   _bnormpool_07,
};

void bnormpool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_bnormpool_tab_512[2])(xloc, uloc, varn, beta, yloc);
         xloc += (1 << 10);
         uloc += (1 << 10);
         varn += (1 << 10);
         beta += (1 << 10);
         yloc += (1 << 10);

         (_bnormpool_tab_512[2])(xloc, uloc, varn, beta, yloc);
         xloc += (1 << 10);
         uloc += (1 << 10);
         varn += (1 << 10);
         beta += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_bnormpool_tab_512[nb])(xloc, uloc, varn, beta, yloc);
      xloc += (nb << 9);
      uloc += (nb << 9);
      varn += (nb << 9);
      beta += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_bnormpool_tab_128[nb])(xloc, uloc, varn, beta, yloc);
      xloc += (nb << 7);
      uloc += (nb << 7);
      varn += (nb << 7);
      beta += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_bnormpool_tab_32[nb])(xloc, uloc, varn, beta, yloc);
      xloc += (nb << 5);
      uloc += (nb << 5);
      varn += (nb << 5);
      beta += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_bnormpool_tab_08[nb])(xloc, uloc, varn, beta, yloc);

      xloc += (nb << 3);
      uloc += (nb << 3);
      varn += (nb << 3);
      beta += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   /* 0-7 */
   {
      _bnormpool_tab_00[_n](xloc, uloc, varn, beta, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define bnrelupool_08(xloc, uloc, varn, beta, yloc)   \
{  \
   cnn_type_t zloc[CNN_BCHSIZ];          \
   cnn_type_t dloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      zloc[p] = xloc[p] - uloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      dloc[p] = zloc[p] * varn[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = dloc[p] + beta[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = (yloc[p] > 0) ? (yloc[p]) : 0.0; \
   }                                     \
                                         \
   xloc += CNN_BCHSIZ;                   \
   uloc += CNN_BCHSIZ;                   \
   varn += CNN_BCHSIZ;                   \
   beta += CNN_BCHSIZ;                   \
   yloc += CNN_BCHSIZ;                   \
}

#define bnrelupool_16(xloc, uloc, varn, beta, yloc) \
{  \
   bnrelupool_08(xloc, uloc, varn, beta, yloc);  \
   bnrelupool_08(xloc, uloc, varn, beta, yloc);  \
}

#define bnrelupool_32(xloc, uloc, varn, beta, yloc) \
{  \
   bnrelupool_16(xloc, uloc, varn, beta, yloc);  \
   bnrelupool_16(xloc, uloc, varn, beta, yloc);  \
}

#define bnrelupool_64(xloc, uloc, varn, beta, yloc) \
{  \
   bnrelupool_32(xloc, uloc, varn, beta, yloc);  \
   bnrelupool_32(xloc, uloc, varn, beta, yloc);  \
}

#define bnrelupool_128(xloc, uloc, varn, beta, yloc) \
{  \
   bnrelupool_64(xloc, uloc, varn, beta, yloc);  \
   bnrelupool_64(xloc, uloc, varn, beta, yloc);  \
}

#define bnrelupool_256(xloc, uloc, varn, beta, yloc) \
{  \
   bnrelupool_128(xloc, uloc, varn, beta, yloc);  \
   bnrelupool_128(xloc, uloc, varn, beta, yloc);  \
}

#define bnrelupool_512(xloc, uloc, varn, beta, yloc) \
{  \
   bnrelupool_256(xloc, uloc, varn, beta, yloc);  \
   bnrelupool_256(xloc, uloc, varn, beta, yloc);  \
}

static void _bnrelupool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _bnrelupool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   bnrelupool_512(xloc, uloc, varn, beta, yloc);
   bnrelupool_512(xloc, uloc, varn, beta, yloc);
   bnrelupool_512(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_512(xloc, uloc, varn, beta, yloc);
   bnrelupool_512(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_512(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_128(xloc, uloc, varn, beta, yloc);
   bnrelupool_128(xloc, uloc, varn, beta, yloc);
   bnrelupool_128(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_256(  
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_128(xloc, uloc, varn, beta, yloc);
   bnrelupool_128(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_128(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_64(xloc, uloc, varn, beta, yloc);
   bnrelupool_32(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_64(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_32(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_16(xloc, uloc, varn, beta, yloc);
   bnrelupool_08(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_16(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   bnrelupool_08(xloc, uloc, varn, beta, yloc);
}

static void _bnrelupool_07(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 7; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _bnrelupool_06(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 6; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _bnrelupool_05(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 5; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _bnrelupool_04(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 4; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _bnrelupool_03(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 3; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _bnrelupool_02(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 2; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _bnrelupool_01(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 1; i++)
   {
      _yloc[i] = (_xloc[i] - _uloc[i]) * _varn[i] + _beta[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

typedef void (*bnrelupool_func_t)(
   const cnn_type_t *, 
   const cnn_type_t *, 
   const cnn_type_t *, 
   const cnn_type_t *, 
         cnn_type_t *
);
static bnrelupool_func_t _bnrelupool_tab_512[] = {
   _bnrelupool_00,
   _bnrelupool_512,
   _bnrelupool_1024,
   _bnrelupool_1536,
};

static bnrelupool_func_t _bnrelupool_tab_128[] = {
   _bnrelupool_00,
   _bnrelupool_128,
   _bnrelupool_256,
   _bnrelupool_384,
};

static bnrelupool_func_t _bnrelupool_tab_32[] = {
   _bnrelupool_00,
   _bnrelupool_32,
   _bnrelupool_64,
   _bnrelupool_96,
};

static bnrelupool_func_t _bnrelupool_tab_08[] = {
   _bnrelupool_00,
   _bnrelupool_08,
   _bnrelupool_16,
   _bnrelupool_24,
};

static bnrelupool_func_t _bnrelupool_tab_00[] = {
   _bnrelupool_00,
   _bnrelupool_01,
   _bnrelupool_02,
   _bnrelupool_03,
   _bnrelupool_04,
   _bnrelupool_05,
   _bnrelupool_06,
   _bnrelupool_07,
};

void bnrelupool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _uloc,
   const cnn_type_t * restrict _varn,
   const cnn_type_t * restrict _beta,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * varn = __builtin_assume_aligned(_varn, 16);
   cnn_type_t * beta = __builtin_assume_aligned(_beta, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_bnrelupool_tab_512[2])(xloc, uloc, varn, beta, yloc);
         xloc += (1 << 10);
         uloc += (1 << 10);
         varn += (1 << 10);
         beta += (1 << 10);
         yloc += (1 << 10);

         (_bnrelupool_tab_512[2])(xloc, uloc, varn, beta, yloc);
         xloc += (1 << 10);
         uloc += (1 << 10);
         varn += (1 << 10);
         beta += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_bnrelupool_tab_512[nb])(xloc, uloc, varn, beta, yloc);
      xloc += (nb << 9);
      uloc += (nb << 9);
      varn += (nb << 9);
      beta += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_bnrelupool_tab_128[nb])(xloc, uloc, varn, beta, yloc);
      xloc += (nb << 7);
      uloc += (nb << 7);
      varn += (nb << 7);
      beta += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_bnrelupool_tab_32[nb])(xloc, uloc, varn, beta, yloc);
      xloc += (nb << 5);
      uloc += (nb << 5);
      varn += (nb << 5);
      beta += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_bnrelupool_tab_08[nb])(xloc, uloc, varn, beta, yloc);

      xloc += (nb << 3);
      uloc += (nb << 3);
      varn += (nb << 3);
      beta += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   /* 0-7 */
   {
      _bnrelupool_tab_00[_n](xloc, uloc, varn, beta, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define b35respool_08(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc)   \
{  \
   cnn_type_t ploc[CNN_BCHSIZ];          \
   cnn_type_t qloc[CNN_BCHSIZ];          \
   cnn_type_t uloc[CNN_BCHSIZ];          \
   cnn_type_t dloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      ploc[p] = z0loc[p] + z1loc[p];     \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      qloc[p] = z2loc[p] + bloc[p];      \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = ploc[p] + qloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      dloc[p] = uloc[p] * scl;           \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = dloc[p] + xloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = (yloc[p] > 0) ? (yloc[p]) : 0.0; \
   }                                     \
                                         \
   z0loc += CNN_BCHSIZ;                  \
   z1loc += CNN_BCHSIZ;                  \
   z2loc += CNN_BCHSIZ;                  \
   bloc  += CNN_BCHSIZ;                  \
   xloc  += CNN_BCHSIZ;                  \
   yloc  += CNN_BCHSIZ;                  \
}

#define b35respool_16(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc) \
{  \
   b35respool_08(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
   b35respool_08(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
}

#define b35respool_32(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc) \
{  \
   b35respool_16(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
   b35respool_16(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
}

#define b35respool_64(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc) \
{  \
   b35respool_32(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
   b35respool_32(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
}

#define b35respool_128(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc) \
{  \
   b35respool_64(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
   b35respool_64(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
}

#define b35respool_256(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc) \
{  \
   b35respool_128(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
   b35respool_128(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
}

#define b35respool_512(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc) \
{  \
   b35respool_256(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
   b35respool_256(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);  \
}

static void _b35respool_00(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _b35respool_1536(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   b35respool_512(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
   b35respool_512(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
   b35respool_512(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_1024(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_512(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
   b35respool_512(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_512(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_512(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_384(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_128(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
   b35respool_128(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
   b35respool_128(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_256(  
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_128(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
   b35respool_128(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_128(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_128(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_96(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_64(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
   b35respool_32(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_64(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_64(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_32(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_32(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_24(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_16(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
   b35respool_08(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_16(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_16(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_08(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b35respool_08(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
}

static void _b35respool_07(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 7; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _z2loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b35respool_06(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 6; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _z2loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b35respool_05(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 5; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _z2loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b35respool_04(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 4; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _z2loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b35respool_03(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 3; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _z2loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b35respool_02(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 2; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _z2loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b35respool_01(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 1; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _z2loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

typedef void (*b35respool_func_t)(
   const cnn_type_t *, 
   const cnn_type_t *, 
   const cnn_type_t *, 
   const cnn_type_t *, 
         cnn_type_t  , 
   const cnn_type_t *, 
         cnn_type_t *
);

static b35respool_func_t _b35respool_tab_512[] = {
   _b35respool_00,
   _b35respool_512,
   _b35respool_1024,
   _b35respool_1536,
};

static b35respool_func_t _b35respool_tab_128[] = {
   _b35respool_00,
   _b35respool_128,
   _b35respool_256,
   _b35respool_384,
};

static b35respool_func_t _b35respool_tab_32[] = {
   _b35respool_00,
   _b35respool_32,
   _b35respool_64,
   _b35respool_96,
};

static b35respool_func_t _b35respool_tab_08[] = {
   _b35respool_00,
   _b35respool_08,
   _b35respool_16,
   _b35respool_24,
};

static b35respool_func_t _b35respool_tab_00[] = {
   _b35respool_00,
   _b35respool_01,
   _b35respool_02,
   _b35respool_03,
   _b35respool_04,
   _b35respool_05,
   _b35respool_06,
   _b35respool_07,
};

void b35respool(
         int                   _n,
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _z2loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * z2loc = __builtin_assume_aligned(_z2loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_b35respool_tab_512[2])(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
         z0loc += (1 << 10);
         z1loc += (1 << 10);
         z2loc += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);

         (_b35respool_tab_512[2])(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
         z0loc += (1 << 10);
         z1loc += (1 << 10);
         z2loc += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_b35respool_tab_512[nb])(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 9);
      z1loc += (nb << 9);
      z2loc += (nb << 9);
      bloc  += (nb << 9);
      xloc  += (nb << 9);
      yloc  += (nb << 9);
      _n    -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_b35respool_tab_128[nb])(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 7);
      z1loc += (nb << 7);
      z2loc += (nb << 7);
      bloc  += (nb << 7);
      xloc  += (nb << 7);
      yloc  += (nb << 7);
      _n    -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_b35respool_tab_32[nb])(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 5);
      z1loc += (nb << 5);
      z2loc += (nb << 5);
      bloc  += (nb << 5);
      xloc  += (nb << 5);
      yloc  += (nb << 5);
      _n    -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_b35respool_tab_08[nb])(z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 3);
      z1loc += (nb << 3);
      z2loc += (nb << 3);
      bloc  += (nb << 3);
      xloc  += (nb << 3);
      yloc  += (nb << 3);
      _n    -= (nb << 3);
   }

   /* 0-7 */
   {
      _b35respool_tab_00[_n](z0loc, z1loc, z2loc, bloc, scl, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define b17respool_08(z0loc, z1loc, bloc, scl, xloc, yloc)   \
{  \
   cnn_type_t ploc[CNN_BCHSIZ];          \
   cnn_type_t uloc[CNN_BCHSIZ];          \
   cnn_type_t dloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      ploc[p] = z0loc[p] + z1loc[p];     \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = ploc[p] + bloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      dloc[p] = uloc[p] * scl;           \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = dloc[p] + xloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = (yloc[p] > 0) ? (yloc[p]) : 0.0; \
   }                                     \
                                         \
   z0loc += CNN_BCHSIZ;                  \
   z1loc += CNN_BCHSIZ;                  \
   bloc  += CNN_BCHSIZ;                  \
   xloc  += CNN_BCHSIZ;                  \
   yloc  += CNN_BCHSIZ;                  \
}

#define b17respool_16(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool_08(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool_08(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool_32(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool_16(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool_16(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool_64(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool_32(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool_32(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool_128(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool_64(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool_64(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool_256(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool_128(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool_128(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool_512(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool_256(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool_256(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

static void _b17respool_00(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _b17respool_1536(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   b17respool_512(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool_512(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool_512(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_1024(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_512(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool_512(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_512(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_512(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_384(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_128(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool_128(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool_128(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_256(  
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_128(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool_128(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_128(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_128(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_96(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_64(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool_32(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_64(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_64(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_32(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_32(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_24(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_16(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool_08(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_16(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_16(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_08(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool_08(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool_07(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 7; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b17respool_06(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 6; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b17respool_05(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 5; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b17respool_04(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 4; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b17respool_03(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 3; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b17respool_02(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 2; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b17respool_01(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 1; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

typedef void (*b17respool_func_t)(
   const cnn_type_t *, 
   const cnn_type_t *, 
   const cnn_type_t *, 
         cnn_type_t  , 
   const cnn_type_t *, 
         cnn_type_t *
);

static b17respool_func_t _b17respool_tab_512[] = {
   _b17respool_00,
   _b17respool_512,
   _b17respool_1024,
   _b17respool_1536,
};

static b17respool_func_t _b17respool_tab_128[] = {
   _b17respool_00,
   _b17respool_128,
   _b17respool_256,
   _b17respool_384,
};

static b17respool_func_t _b17respool_tab_32[] = {
   _b17respool_00,
   _b17respool_32,
   _b17respool_64,
   _b17respool_96,
};

static b17respool_func_t _b17respool_tab_08[] = {
   _b17respool_00,
   _b17respool_08,
   _b17respool_16,
   _b17respool_24,
};

static b17respool_func_t _b17respool_tab_00[] = {
   _b17respool_00,
   _b17respool_01,
   _b17respool_02,
   _b17respool_03,
   _b17respool_04,
   _b17respool_05,
   _b17respool_06,
   _b17respool_07,
};

void b17respool(
         int                   _n,
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_b17respool_tab_512[2])(z0loc, z1loc, bloc, scl, xloc, yloc);
         z0loc += (1 << 10);
         z1loc += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);

         (_b17respool_tab_512[2])(z0loc, z1loc, bloc, scl, xloc, yloc);
         z0loc += (1 << 10);
         z1loc += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_b17respool_tab_512[nb])(z0loc, z1loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 9);
      z1loc += (nb << 9);
      bloc  += (nb << 9);
      xloc  += (nb << 9);
      yloc  += (nb << 9);
      _n    -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_b17respool_tab_128[nb])(z0loc, z1loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 7);
      z1loc += (nb << 7);
      bloc  += (nb << 7);
      xloc  += (nb << 7);
      yloc  += (nb << 7);
      _n    -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_b17respool_tab_32[nb])(z0loc, z1loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 5);
      z1loc += (nb << 5);
      bloc  += (nb << 5);
      xloc  += (nb << 5);
      yloc  += (nb << 5);
      _n    -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_b17respool_tab_08[nb])(z0loc, z1loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 3);
      z1loc += (nb << 3);
      bloc  += (nb << 3);
      xloc  += (nb << 3);
      yloc  += (nb << 3);
      _n    -= (nb << 3);
   }

   /* 0-7 */
   {
      _b17respool_tab_00[_n](z0loc, z1loc, bloc, scl, xloc, yloc);
   }
}

void b08respool(
         int                   _n,
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   return b17respool(
      _n,
      _z0loc, _z1loc, 
      _bloc,  _scl,
      _xloc,  _yloc
   );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define b17respool2_08(z0loc, z1loc, bloc, scl, xloc, yloc)   \
{  \
   cnn_type_t ploc[CNN_BCHSIZ];          \
   cnn_type_t uloc[CNN_BCHSIZ];          \
   cnn_type_t dloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      ploc[p] = z0loc[p] + z1loc[p];     \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = ploc[p] + bloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      dloc[p] = uloc[p] * scl;           \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = dloc[p] + xloc[p];       \
   }                                     \
                                         \
   z0loc += CNN_BCHSIZ;                  \
   z1loc += CNN_BCHSIZ;                  \
   bloc  += CNN_BCHSIZ;                  \
   xloc  += CNN_BCHSIZ;                  \
   yloc  += CNN_BCHSIZ;                  \
}

#define b17respool2_16(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool2_08(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool2_08(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool2_32(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool2_16(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool2_16(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool2_64(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool2_32(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool2_32(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool2_128(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool2_64(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool2_64(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool2_256(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool2_128(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool2_128(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

#define b17respool2_512(z0loc, z1loc, bloc, scl, xloc, yloc) \
{  \
   b17respool2_256(z0loc, z1loc, bloc, scl, xloc, yloc);  \
   b17respool2_256(z0loc, z1loc, bloc, scl, xloc, yloc);  \
}

static void _b17respool2_00(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _b17respool2_1536(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   b17respool2_512(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool2_512(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool2_512(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_1024(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_512(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool2_512(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_512(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_512(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_384(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_128(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool2_128(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool2_128(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_256(  
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_128(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool2_128(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_128(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_128(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_96(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_64(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool2_32(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_64(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_64(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_32(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_32(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_24(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_16(z0loc, z1loc, bloc, scl, xloc, yloc);
   b17respool2_08(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_16(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_16(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_08(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b17respool2_08(z0loc, z1loc, bloc, scl, xloc, yloc);
}

static void _b17respool2_07(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 7; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b17respool2_06(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 6; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b17respool2_05(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 5; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b17respool2_04(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 4; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b17respool2_03(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 3; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b17respool2_02(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 2; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b17respool2_01(
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 1; i++)
   {
      _yloc[i] = (_z0loc[i] + _z1loc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

typedef void (*b17respool2_func_t)(
   const cnn_type_t *, 
   const cnn_type_t *, 
   const cnn_type_t *, 
         cnn_type_t  , 
   const cnn_type_t *, 
         cnn_type_t *
);

static b17respool2_func_t _b17respool2_tab_512[] = {
   _b17respool2_00,
   _b17respool2_512,
   _b17respool2_1024,
   _b17respool2_1536,
};

static b17respool2_func_t _b17respool2_tab_128[] = {
   _b17respool2_00,
   _b17respool2_128,
   _b17respool2_256,
   _b17respool2_384,
};

static b17respool2_func_t _b17respool2_tab_32[] = {
   _b17respool2_00,
   _b17respool2_32,
   _b17respool2_64,
   _b17respool2_96,
};

static b17respool2_func_t _b17respool2_tab_08[] = {
   _b17respool2_00,
   _b17respool2_08,
   _b17respool2_16,
   _b17respool2_24,
};

static b17respool2_func_t _b17respool2_tab_00[] = {
   _b17respool2_00,
   _b17respool2_01,
   _b17respool2_02,
   _b17respool2_03,
   _b17respool2_04,
   _b17respool2_05,
   _b17respool2_06,
   _b17respool2_07,
};

void b17respool2(
         int                   _n,
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * z0loc = __builtin_assume_aligned(_z0loc, 16); 
   cnn_type_t * z1loc = __builtin_assume_aligned(_z1loc, 16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_b17respool2_tab_512[2])(z0loc, z1loc, bloc, scl, xloc, yloc);
         z0loc += (1 << 10);
         z1loc += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);

         (_b17respool2_tab_512[2])(z0loc, z1loc, bloc, scl, xloc, yloc);
         z0loc += (1 << 10);
         z1loc += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_b17respool2_tab_512[nb])(z0loc, z1loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 9);
      z1loc += (nb << 9);
      bloc  += (nb << 9);
      xloc  += (nb << 9);
      yloc  += (nb << 9);
      _n    -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_b17respool2_tab_128[nb])(z0loc, z1loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 7);
      z1loc += (nb << 7);
      bloc  += (nb << 7);
      xloc  += (nb << 7);
      yloc  += (nb << 7);
      _n    -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_b17respool2_tab_32[nb])(z0loc, z1loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 5);
      z1loc += (nb << 5);
      bloc  += (nb << 5);
      xloc  += (nb << 5);
      yloc  += (nb << 5);
      _n    -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_b17respool2_tab_08[nb])(z0loc, z1loc, bloc, scl, xloc, yloc);
      z0loc += (nb << 3);
      z1loc += (nb << 3);
      bloc  += (nb << 3);
      xloc  += (nb << 3);
      yloc  += (nb << 3);
      _n    -= (nb << 3);
   }

   /* 0-7 */
   {
      _b17respool2_tab_00[_n](z0loc, z1loc, bloc, scl, xloc, yloc);
   }
}

void b08respool2(
         int                   _n,
   const cnn_type_t * restrict _z0loc,
   const cnn_type_t * restrict _z1loc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   return b17respool2(
      _n,
      _z0loc, _z1loc, 
      _bloc,  _scl,
      _xloc,  _yloc
   );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define b01respool_08(zloc, bloc, scl, xloc, yloc)   \
{  \
   cnn_type_t uloc[CNN_BCHSIZ];          \
   cnn_type_t dloc[CNN_BCHSIZ];          \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = zloc[p] + bloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      dloc[p] = uloc[p] * scl;           \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = dloc[p] + xloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = (yloc[p] > 0) ? (yloc[p]) : 0.0; \
   }                                     \
                                         \
   zloc  += CNN_BCHSIZ;                  \
   bloc  += CNN_BCHSIZ;                  \
   xloc  += CNN_BCHSIZ;                  \
   yloc  += CNN_BCHSIZ;                  \
}

#define b01respool_16(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool_08(zloc, bloc, scl, xloc, yloc);  \
   b01respool_08(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool_32(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool_16(zloc, bloc, scl, xloc, yloc);  \
   b01respool_16(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool_64(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool_32(zloc, bloc, scl, xloc, yloc);  \
   b01respool_32(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool_128(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool_64(zloc, bloc, scl, xloc, yloc);  \
   b01respool_64(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool_256(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool_128(zloc, bloc, scl, xloc, yloc);  \
   b01respool_128(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool_512(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool_256(zloc, bloc, scl, xloc, yloc);  \
   b01respool_256(zloc, bloc, scl, xloc, yloc);  \
}

static void _b01respool_00(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _b01respool_1536(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   b01respool_512(zloc, bloc, scl, xloc, yloc);
   b01respool_512(zloc, bloc, scl, xloc, yloc);
   b01respool_512(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_1024(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_512(zloc, bloc, scl, xloc, yloc);
   b01respool_512(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_512(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_512(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_384(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_128(zloc, bloc, scl, xloc, yloc);
   b01respool_128(zloc, bloc, scl, xloc, yloc);
   b01respool_128(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_256(  
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_128(zloc, bloc, scl, xloc, yloc);
   b01respool_128(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_128(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_128(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_96(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_64(zloc, bloc, scl, xloc, yloc);
   b01respool_32(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_64(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_64(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_32(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_32(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_24(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_16(zloc, bloc, scl, xloc, yloc);
   b01respool_08(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_16(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_16(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_08(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool_08(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool_07(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 7; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b01respool_06(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 6; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b01respool_05(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 5; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b01respool_04(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 4; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b01respool_03(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 3; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b01respool_02(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 2; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

static void _b01respool_01(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 1; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
      _yloc[i] = (_yloc[i] > 0.0) ? (_yloc[i]) : (0.0);
   }
}

typedef void (*b01respool_func_t)(
   const cnn_type_t *, 
   const cnn_type_t *, 
         cnn_type_t  , 
   const cnn_type_t *, 
         cnn_type_t *
);

static b01respool_func_t _b01respool_tab_512[] = {
   _b01respool_00,
   _b01respool_512,
   _b01respool_1024,
   _b01respool_1536,
};

static b01respool_func_t _b01respool_tab_128[] = {
   _b01respool_00,
   _b01respool_128,
   _b01respool_256,
   _b01respool_384,
};

static b01respool_func_t _b01respool_tab_32[] = {
   _b01respool_00,
   _b01respool_32,
   _b01respool_64,
   _b01respool_96,
};

static b01respool_func_t _b01respool_tab_08[] = {
   _b01respool_00,
   _b01respool_08,
   _b01respool_16,
   _b01respool_24,
};

static b01respool_func_t _b01respool_tab_00[] = {
   _b01respool_00,
   _b01respool_01,
   _b01respool_02,
   _b01respool_03,
   _b01respool_04,
   _b01respool_05,
   _b01respool_06,
   _b01respool_07,
};

void b01respool(
         int                   _n,
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_b01respool_tab_512[2])(zloc, bloc, scl, xloc, yloc);
         zloc  += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);

         (_b01respool_tab_512[2])(zloc, bloc, scl, xloc, yloc);
         zloc  += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_b01respool_tab_512[nb])(zloc, bloc, scl, xloc, yloc);
      zloc  += (nb << 9);
      bloc  += (nb << 9);
      xloc  += (nb << 9);
      yloc  += (nb << 9);
      _n    -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_b01respool_tab_128[nb])(zloc, bloc, scl, xloc, yloc);
      zloc  += (nb << 7);
      bloc  += (nb << 7);
      xloc  += (nb << 7);
      yloc  += (nb << 7);
      _n    -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_b01respool_tab_32[nb])(zloc, bloc, scl, xloc, yloc);
      zloc  += (nb << 5);
      bloc  += (nb << 5);
      xloc  += (nb << 5);
      yloc  += (nb << 5);
      _n    -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_b01respool_tab_08[nb])(zloc, bloc, scl, xloc, yloc);
      zloc  += (nb << 3);
      bloc  += (nb << 3);
      xloc  += (nb << 3);
      yloc  += (nb << 3);
      _n    -= (nb << 3);
   }

   /* 0-7 */
   {
      _b01respool_tab_00[_n](zloc, bloc, scl, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define b01respool2_08(zloc, bloc, scl, xloc, yloc)   \
{  \
   cnn_type_t uloc[CNN_BCHSIZ];          \
   cnn_type_t dloc[CNN_BCHSIZ];          \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = zloc[p] + bloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      dloc[p] = uloc[p] * scl;           \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = dloc[p] + xloc[p];       \
   }                                     \
                                         \
   zloc  += CNN_BCHSIZ;                  \
   bloc  += CNN_BCHSIZ;                  \
   xloc  += CNN_BCHSIZ;                  \
   yloc  += CNN_BCHSIZ;                  \
}

#define b01respool2_16(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool2_08(zloc, bloc, scl, xloc, yloc);  \
   b01respool2_08(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool2_32(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool2_16(zloc, bloc, scl, xloc, yloc);  \
   b01respool2_16(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool2_64(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool2_32(zloc, bloc, scl, xloc, yloc);  \
   b01respool2_32(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool2_128(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool2_64(zloc, bloc, scl, xloc, yloc);  \
   b01respool2_64(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool2_256(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool2_128(zloc, bloc, scl, xloc, yloc);  \
   b01respool2_128(zloc, bloc, scl, xloc, yloc);  \
}

#define b01respool2_512(zloc, bloc, scl, xloc, yloc) \
{  \
   b01respool2_256(zloc, bloc, scl, xloc, yloc);  \
   b01respool2_256(zloc, bloc, scl, xloc, yloc);  \
}

static void _b01respool2_00(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _b01respool2_1536(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   b01respool2_512(zloc, bloc, scl, xloc, yloc);
   b01respool2_512(zloc, bloc, scl, xloc, yloc);
   b01respool2_512(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_1024(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_512(zloc, bloc, scl, xloc, yloc);
   b01respool2_512(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_512(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_512(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_384(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_128(zloc, bloc, scl, xloc, yloc);
   b01respool2_128(zloc, bloc, scl, xloc, yloc);
   b01respool2_128(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_256(  
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_128(zloc, bloc, scl, xloc, yloc);
   b01respool2_128(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_128(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_128(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_96(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_64(zloc, bloc, scl, xloc, yloc);
   b01respool2_32(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_64(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_64(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_32(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_32(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_24(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_16(zloc, bloc, scl, xloc, yloc);
   b01respool2_08(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_16(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_16(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_08(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);
   
   b01respool2_08(zloc, bloc, scl, xloc, yloc);
}

static void _b01respool2_07(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 7; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b01respool2_06(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 6; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b01respool2_05(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 5; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b01respool2_04(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 4; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b01respool2_03(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 3; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b01respool2_02(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 2; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

static void _b01respool2_01(
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   for (int i = 0; i < 1; i++)
   {
      _yloc[i] = (_zloc[i] + _bloc[i]) * _scl + _xloc[i];
   }
}

typedef void (*b01respool2_func_t)(
   const cnn_type_t *, 
   const cnn_type_t *, 
         cnn_type_t  , 
   const cnn_type_t *, 
         cnn_type_t *
);

static b01respool2_func_t _b01respool2_tab_512[] = {
   _b01respool2_00,
   _b01respool2_512,
   _b01respool2_1024,
   _b01respool2_1536,
};

static b01respool2_func_t _b01respool2_tab_128[] = {
   _b01respool2_00,
   _b01respool2_128,
   _b01respool2_256,
   _b01respool2_384,
};

static b01respool2_func_t _b01respool2_tab_32[] = {
   _b01respool2_00,
   _b01respool2_32,
   _b01respool2_64,
   _b01respool2_96,
};

static b01respool2_func_t _b01respool2_tab_08[] = {
   _b01respool2_00,
   _b01respool2_08,
   _b01respool2_16,
   _b01respool2_24,
};

static b01respool2_func_t _b01respool2_tab_00[] = {
   _b01respool2_00,
   _b01respool2_01,
   _b01respool2_02,
   _b01respool2_03,
   _b01respool2_04,
   _b01respool2_05,
   _b01respool2_06,
   _b01respool2_07,
};

void b01respool2(
         int                   _n,
   const cnn_type_t * restrict _zloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t            _scl,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * zloc  = __builtin_assume_aligned(_zloc,  16); 
   cnn_type_t * bloc  = __builtin_assume_aligned(_bloc,  16);
   cnn_type_t   scl   = _scl;
   cnn_type_t * xloc  = __builtin_assume_aligned(_xloc,  16); 
   cnn_type_t * yloc  = __builtin_assume_aligned(_yloc,  16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_b01respool2_tab_512[2])(zloc, bloc, scl, xloc, yloc);
         zloc  += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);

         (_b01respool2_tab_512[2])(zloc, bloc, scl, xloc, yloc);
         zloc  += (1 << 10);
         bloc  += (1 << 10);
         xloc  += (1 << 10);
         yloc  += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_b01respool2_tab_512[nb])(zloc, bloc, scl, xloc, yloc);
      zloc  += (nb << 9);
      bloc  += (nb << 9);
      xloc  += (nb << 9);
      yloc  += (nb << 9);
      _n    -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_b01respool2_tab_128[nb])(zloc, bloc, scl, xloc, yloc);
      zloc  += (nb << 7);
      bloc  += (nb << 7);
      xloc  += (nb << 7);
      yloc  += (nb << 7);
      _n    -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_b01respool2_tab_32[nb])(zloc, bloc, scl, xloc, yloc);
      zloc  += (nb << 5);
      bloc  += (nb << 5);
      xloc  += (nb << 5);
      yloc  += (nb << 5);
      _n    -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_b01respool2_tab_08[nb])(zloc, bloc, scl, xloc, yloc);
      zloc  += (nb << 3);
      bloc  += (nb << 3);
      xloc  += (nb << 3);
      yloc  += (nb << 3);
      _n    -= (nb << 3);
   }

   /* 0-7 */
   {
      _b01respool2_tab_00[_n](zloc, bloc, scl, xloc, yloc);
   }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define prelupool_08(xloc, bloc, ploc, yloc)   \
{  \
   cnn_type_t uloc[CNN_BCHSIZ];          \
   acc_type_t wloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = xloc[p] + bloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      wloc[p] = uloc[p] * ploc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = (uloc[p] < 0) ? (wloc[p]) : (uloc[p]);  \
   }                                     \
                                         \
   xloc += CNN_BCHSIZ;                   \
   bloc += CNN_BCHSIZ;                   \
   ploc += CNN_BCHSIZ;                   \
   yloc += CNN_BCHSIZ;                   \
}

#define prelupool_16(xloc, bloc, ploc, yloc) \
{  \
   prelupool_08(xloc, bloc, ploc, yloc);  \
   prelupool_08(xloc, bloc, ploc, yloc);  \
}

#define prelupool_32(xloc, bloc, ploc, yloc) \
{  \
   prelupool_16(xloc, bloc, ploc, yloc);  \
   prelupool_16(xloc, bloc, ploc, yloc);  \
}

#define prelupool_64(xloc, bloc, ploc, yloc) \
{  \
   prelupool_32(xloc, bloc, ploc, yloc);  \
   prelupool_32(xloc, bloc, ploc, yloc);  \
}

#define prelupool_128(xloc, bloc, ploc, yloc) \
{  \
   prelupool_64(xloc, bloc, ploc, yloc);  \
   prelupool_64(xloc, bloc, ploc, yloc);  \
}

#define prelupool_256(xloc, bloc, ploc, yloc) \
{  \
   prelupool_128(xloc, bloc, ploc, yloc);  \
   prelupool_128(xloc, bloc, ploc, yloc);  \
}

#define prelupool_512(xloc, bloc, ploc, yloc) \
{  \
   prelupool_256(xloc, bloc, ploc, yloc);  \
   prelupool_256(xloc, bloc, ploc, yloc);  \
}

static void _prelupool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _prelupool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   prelupool_512(xloc, bloc, ploc, yloc);
   prelupool_512(xloc, bloc, ploc, yloc);
   prelupool_512(xloc, bloc, ploc, yloc);
}

static void _prelupool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_512(xloc, bloc, ploc, yloc);
   prelupool_512(xloc, bloc, ploc, yloc);
}

static void _prelupool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_512(xloc, bloc, ploc, yloc);
}

static void _prelupool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_128(xloc, bloc, ploc, yloc);
   prelupool_128(xloc, bloc, ploc, yloc);
   prelupool_128(xloc, bloc, ploc, yloc);
}

static void _prelupool_256(  
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_128(xloc, bloc, ploc, yloc);
   prelupool_128(xloc, bloc, ploc, yloc);
}

static void _prelupool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_128(xloc, bloc, ploc, yloc);
}

static void _prelupool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_64(xloc, bloc, ploc, yloc);
   prelupool_32(xloc, bloc, ploc, yloc);
}

static void _prelupool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_64(xloc, bloc, ploc, yloc);
}

static void _prelupool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_32(xloc, bloc, ploc, yloc);
}

static void _prelupool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_16(xloc, bloc, ploc, yloc);
   prelupool_08(xloc, bloc, ploc, yloc);
}

static void _prelupool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_16(xloc, bloc, ploc, yloc);
}

static void _prelupool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   prelupool_08(xloc, bloc, ploc, yloc);
}

static void _prelupool_07(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict bloc,
   const cnn_type_t * restrict ploc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 7; i++)
   {
      cnn_type_t uloc, wloc;

      uloc    = xloc[i] + bloc[i];
      wloc    = uloc    * ploc[i];
      yloc[i] = (uloc < 0) ? (wloc) : (uloc);
   }
}

static void _prelupool_06(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict bloc,
   const cnn_type_t * restrict ploc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 6; i++)
   {
      cnn_type_t uloc, wloc;

      uloc    = xloc[i] + bloc[i];
      wloc    = uloc    * ploc[i];
      yloc[i] = (uloc < 0) ? (wloc) : (uloc);
   }
}

static void _prelupool_05(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict bloc,
   const cnn_type_t * restrict ploc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 5; i++)
   {
      cnn_type_t uloc, wloc;

      uloc    = xloc[i] + bloc[i];
      wloc    = uloc    * ploc[i];
      yloc[i] = (uloc < 0) ? (wloc) : (uloc);
   }
}

static void _prelupool_04(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict bloc,
   const cnn_type_t * restrict ploc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 4; i++)
   {
      cnn_type_t uloc, wloc;

      uloc    = xloc[i] + bloc[i];
      wloc    = uloc    * ploc[i];
      yloc[i] = (uloc < 0) ? (wloc) : (uloc);
   }
}

static void _prelupool_03(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict bloc,
   const cnn_type_t * restrict ploc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 3; i++)
   {
      cnn_type_t uloc, wloc;

      uloc    = xloc[i] + bloc[i];
      wloc    = uloc    * ploc[i];
      yloc[i] = (uloc < 0) ? (wloc) : (uloc);
   }
}

static void _prelupool_02(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict bloc,
   const cnn_type_t * restrict ploc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 2; i++)
   {
      cnn_type_t uloc, wloc;

      uloc    = xloc[i] + bloc[i];
      wloc    = uloc    * ploc[i];
      yloc[i] = (uloc < 0) ? (wloc) : (uloc);
   }
}

static void _prelupool_01(
   const cnn_type_t * restrict xloc,
   const cnn_type_t * restrict bloc,
   const cnn_type_t * restrict ploc,
         cnn_type_t * restrict yloc
)
{
   for (int i = 0; i < 1; i++)
   {
      cnn_type_t uloc, wloc;

      uloc    = xloc[i] + bloc[i];
      wloc    = uloc    * ploc[i];
      yloc[i] = (uloc < 0) ? (wloc) : (uloc);
   }
}

typedef void (*prelupool_func_t)(
   const cnn_type_t *, 
   const cnn_type_t *, 
   const cnn_type_t *, 
         cnn_type_t *
);
static prelupool_func_t _prelupool_tab_512[] = {
   _prelupool_00,
   _prelupool_512,
   _prelupool_1024,
   _prelupool_1536,
};

static prelupool_func_t _prelupool_tab_128[] = {
   _prelupool_00,
   _prelupool_128,
   _prelupool_256,
   _prelupool_384,
};

static prelupool_func_t _prelupool_tab_32[] = {
   _prelupool_00,
   _prelupool_32,
   _prelupool_64,
   _prelupool_96,
};

static prelupool_func_t _prelupool_tab_08[] = {
   _prelupool_00,
   _prelupool_08,
   _prelupool_16,
   _prelupool_24,
};

static prelupool_func_t _prelupool_tab_00[] = {
   _prelupool_00,
   _prelupool_01,
   _prelupool_02,
   _prelupool_03,
   _prelupool_04,
   _prelupool_05,
   _prelupool_06,
   _prelupool_07,
};

void prelupool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
   const cnn_type_t * restrict _ploc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * ploc = __builtin_assume_aligned(_ploc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_prelupool_tab_512[2])(xloc, bloc, ploc, yloc);
         xloc += (1 << 10);
         bloc += (1 << 10);
         ploc += (1 << 10);
         yloc += (1 << 10);

         (_prelupool_tab_512[2])(xloc, bloc, ploc, yloc);
         xloc += (1 << 10);
         bloc += (1 << 10);
         ploc += (1 << 10);
         yloc += (1 << 10);
      }
      _n -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);

      (_prelupool_tab_512[nb])(xloc, bloc, ploc, yloc);
      xloc += (nb << 9);
      bloc += (nb << 9);
      ploc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);

      (_prelupool_tab_128[nb])(xloc, bloc, ploc, yloc);
      xloc += (nb << 7);
      bloc += (nb << 7);
      ploc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);

      (_prelupool_tab_32[nb])(xloc, bloc, ploc, yloc);
      xloc += (nb << 5);
      bloc += (nb << 5);
      ploc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_prelupool_tab_08[nb])(xloc, bloc, ploc, yloc);

      xloc += (nb << 3);
      bloc += (nb << 3);
      ploc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   /* 0 - 7 */
   {
      _prelupool_tab_00[_n](xloc, bloc, ploc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define relupool_08(xloc, bloc, yloc)   \
{  \
   cnn_type_t uloc[CNN_BCHSIZ];          \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      uloc[p] = xloc[p] + bloc[p];       \
   }                                     \
                                         \
   for (int p = 0; p < CNN_BCHSIZ; p++)  \
   {                                     \
      yloc[p] = (uloc[p] > 0) ? (uloc[p]) : (0.0);  \
   }                                     \
                                         \
   xloc += CNN_BCHSIZ;                   \
   bloc += CNN_BCHSIZ;                   \
   yloc += CNN_BCHSIZ;                   \
}

#define relupool_16(xloc, bloc, yloc) \
{  \
   relupool_08(xloc, bloc, yloc);  \
   relupool_08(xloc, bloc, yloc);  \
}

#define relupool_32(xloc, bloc, yloc) \
{  \
   relupool_16(xloc, bloc, yloc);  \
   relupool_16(xloc, bloc, yloc);  \
}

#define relupool_64(xloc, bloc, yloc) \
{  \
   relupool_32(xloc, bloc, yloc);  \
   relupool_32(xloc, bloc, yloc);  \
}

#define relupool_128(xloc, bloc, yloc) \
{  \
   relupool_64(xloc, bloc, yloc);  \
   relupool_64(xloc, bloc, yloc);  \
}

#define relupool_256(xloc, bloc, yloc) \
{  \
   relupool_128(xloc, bloc, yloc);  \
   relupool_128(xloc, bloc, yloc);  \
}

#define relupool_512(xloc, bloc, yloc) \
{  \
   relupool_256(xloc, bloc, yloc);  \
   relupool_256(xloc, bloc, yloc);  \
}

static void _relupool_00(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   ;
}

static void _relupool_1536(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   relupool_512(xloc, bloc, yloc);
   relupool_512(xloc, bloc, yloc);
   relupool_512(xloc, bloc, yloc);
}

static void _relupool_1024(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_512(xloc, bloc, yloc);
   relupool_512(xloc, bloc, yloc);
}

static void _relupool_512(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_512(xloc, bloc, yloc);
}

static void _relupool_384(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_128(xloc, bloc, yloc);
   relupool_128(xloc, bloc, yloc);
   relupool_128(xloc, bloc, yloc);
}

static void _relupool_256(  
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_128(xloc, bloc, yloc);
   relupool_128(xloc, bloc, yloc);
}

static void _relupool_128(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_128(xloc, bloc, yloc);
}

static void _relupool_96(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_64(xloc, bloc, yloc);
   relupool_32(xloc, bloc, yloc);
}

static void _relupool_64(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_64(xloc, bloc, yloc);
}

static void _relupool_32(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_32(xloc, bloc, yloc);
}

static void _relupool_24(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_16(xloc, bloc, yloc);
   relupool_08(xloc, bloc, yloc);
}

static void _relupool_16(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_16(xloc, bloc, yloc);
}

static void _relupool_08(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   relupool_08(xloc, bloc, yloc);
}

static void _relupool_07(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   for (int p = 0; p < 7; p++){cnn_type_t uloc; uloc = xloc[p] + bloc[p]; yloc[p] = (uloc > 0) ? (uloc) : (0.0);};
}

static void _relupool_06(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 6; p++){cnn_type_t uloc; uloc = xloc[p] + bloc[p]; yloc[p] = (uloc > 0) ? (uloc) : (0.0);};
}

static void _relupool_05(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 5; p++){cnn_type_t uloc; uloc = xloc[p] + bloc[p]; yloc[p] = (uloc > 0) ? (uloc) : (0.0);};
}

static void _relupool_04(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 4; p++){cnn_type_t uloc; uloc = xloc[p] + bloc[p]; yloc[p] = (uloc > 0) ? (uloc) : (0.0);};
}

static void _relupool_03(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 3; p++){cnn_type_t uloc; uloc = xloc[p] + bloc[p]; yloc[p] = (uloc > 0) ? (uloc) : (0.0);};
}

static void _relupool_02(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 2; p++){cnn_type_t uloc; uloc = xloc[p] + bloc[p]; yloc[p] = (uloc > 0) ? (uloc) : (0.0);};
}

static void _relupool_01(
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);
   
   for (int p = 0; p < 1; p++){cnn_type_t uloc; uloc = xloc[p] + bloc[p]; yloc[p] = (uloc > 0) ? (uloc) : (0.0);};
}

typedef void (*relupool_func_t)(const cnn_type_t *, const cnn_type_t *, cnn_type_t *);
static relupool_func_t _relupool_tab_512[] = {
   _relupool_00,
   _relupool_512,
   _relupool_1024,
   _relupool_1536,
};

static relupool_func_t _relupool_tab_128[] = {
   _relupool_00,
   _relupool_128,
   _relupool_256,
   _relupool_384,
};

static relupool_func_t _relupool_tab_32[] = {
   _relupool_00,
   _relupool_32,
   _relupool_64,
   _relupool_96,
};

static relupool_func_t _relupool_tab_08[] = {
   _relupool_00,
   _relupool_08,
   _relupool_16,
   _relupool_24,
};

static relupool_func_t _relupool_tab_00[] = {
   _relupool_00,
   _relupool_01,
   _relupool_02,
   _relupool_03,
   _relupool_04,
   _relupool_05,
   _relupool_06,
   _relupool_07,
};

void relupool(
         int                   _n,
   const cnn_type_t * restrict _xloc,
   const cnn_type_t * restrict _bloc,
         cnn_type_t * restrict _yloc
)
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16); 
   cnn_type_t * bloc = __builtin_assume_aligned(_bloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_n >= (1 << 11))
   {
      int nb = (_n >> 11);
      
      for (int i = 0; i < nb; i++)
      {
         (_relupool_tab_512[2])(xloc, bloc, yloc);
         xloc += (1 << 10);
         bloc += (1 << 10);
         yloc += (1 << 10);

         (_relupool_tab_512[2])(xloc, bloc, yloc);
         xloc += (1 << 10);
         bloc += (1 << 10);
         yloc += (1 << 10);
      }
      _n   -= (nb << 11);
   }

   if (_n >= (1 << 9))
   {
      int nb = (_n >> 9);
      (_relupool_tab_512[nb & 3])(xloc, bloc, yloc);

      xloc += (nb << 9);
      bloc += (nb << 9);
      yloc += (nb << 9);
      _n   -= (nb << 9);
   }

   if (_n >= (1 << 7))
   {
      int nb = (_n >> 7);
      (_relupool_tab_128[nb])(xloc, bloc, yloc);

      xloc += (nb << 7);
      bloc += (nb << 7);
      yloc += (nb << 7);
      _n   -= (nb << 7);
   }

   if (_n >= (1 << 5))
   {
      int nb = (_n >> 5);
      (_relupool_tab_32[nb])(xloc, bloc, yloc);

      xloc += (nb << 5);
      bloc += (nb << 5);
      yloc += (nb << 5);
      _n   -= (nb << 5);
   }

   if (_n >= (1 << 3))
   {
      int nb = (_n >> 3);
      (_relupool_tab_08[nb])(xloc, bloc, yloc);

      xloc += (nb << 3);
      bloc += (nb << 3);
      yloc += (nb << 3);
      _n   -= (nb << 3);
   }

   {
      (_relupool_tab_00[_n])(xloc, bloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define maxpool2x2_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool2x2_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + p]);};  \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
}

#define maxpool2x2_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x2_08(_Lx, _Ln, xloc, yloc) \
   maxpool2x2_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x2_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x2_16(_Lx, _Ln, xloc, yloc) \
   maxpool2x2_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x2_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x2_32(_Lx, _Ln, xloc, yloc) \
   maxpool2x2_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x2_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x2_64(_Lx, _Ln, xloc, yloc) \
   maxpool2x2_64(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x2_256(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x2_128(_Lx, _Ln, xloc, yloc) \
   maxpool2x2_128(_Lx, _Ln, xloc, yloc) \
}

static void _maxpool2x2_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _maxpool2x2_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_256(_Lx, _Ln, xloc, yloc);
   maxpool2x2_128(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x2_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_128(_Lx, _Ln, xloc, yloc);
   maxpool2x2_128(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x2_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_64(_Lx, _Ln, xloc, yloc);
   maxpool2x2_64(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x2_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_64(_Lx, _Ln, xloc, yloc);
   maxpool2x2_32(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x2_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_64(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_32(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_16(_Lx, _Ln, xloc, yloc);
   maxpool2x2_08(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x2_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_16(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_08(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x2_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x2_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool2x2_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool2x2_func_t maxpool2x2_fntab_128[] = {
   _maxpool2x2_00,
   _maxpool2x2_128,
   _maxpool2x2_256,
   _maxpool2x2_384,
};

static maxpool2x2_func_t maxpool2x2_fntab_32[] = {
   _maxpool2x2_00,
   _maxpool2x2_32,
   _maxpool2x2_64,
   _maxpool2x2_96,
};

static maxpool2x2_func_t maxpool2x2_fntab_08[] = {
   _maxpool2x2_00,
   _maxpool2x2_08,
   _maxpool2x2_16,
   _maxpool2x2_24,
};

static maxpool2x2_func_t maxpool2x2_fntab_00[] = {
   _maxpool2x2_00,
   _maxpool2x2_01,
   _maxpool2x2_02,
   _maxpool2x2_03,
   _maxpool2x2_04,
   _maxpool2x2_05,
   _maxpool2x2_06,
   _maxpool2x2_07,
};

static void _maxpool2x2_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (maxpool2x2_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (maxpool2x2_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }

   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (maxpool2x2_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool2x2_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool2x2_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (maxpool2x2_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define maxpool2x1_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + p]) ? (wloc[p]) : (xloc[_Ln + p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool2x1_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x1_08(_Lx, _Ln, xloc, yloc) \
   maxpool2x1_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x1_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x1_16(_Lx, _Ln, xloc, yloc) \
   maxpool2x1_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x1_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x1_32(_Lx, _Ln, xloc, yloc) \
   maxpool2x1_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x1_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x1_64(_Lx, _Ln, xloc, yloc) \
   maxpool2x1_64(_Lx, _Ln, xloc, yloc) \
}

static void _maxpool2x1_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _maxpool2x1_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_64(_Lx, _Ln, xloc, yloc);
   maxpool2x1_32(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x1_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_64(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x1_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_32(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x1_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_16(_Lx, _Ln, xloc, yloc);
   maxpool2x1_08(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x1_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_16(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x1_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x1_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool2x1_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool2x1_func_t maxpool2x1_fntab_32[] = {
   _maxpool2x1_00,
   _maxpool2x1_32,
   _maxpool2x1_64,
   _maxpool2x1_96,
};

static maxpool2x1_func_t maxpool2x1_fntab_08[] = {
   _maxpool2x1_00,
   _maxpool2x1_08,
   _maxpool2x1_16,
   _maxpool2x1_24,
};

static void _maxpool2x1_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool2x1_fntab_32[2])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool2x1_fntab_32[2])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool2x1_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool2x1_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      cnn_type_t wloc[8];
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + p]) ? (wloc[p]) : (xloc[_Ln + p]);};
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define maxpool1x2_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + p]) ? (wloc[p]) : (xloc[_Lx + p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool1x2_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x2_08(_Lx, _Ln, xloc, yloc) \
   maxpool1x2_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x2_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x2_16(_Lx, _Ln, xloc, yloc) \
   maxpool1x2_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x2_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x2_32(_Lx, _Ln, xloc, yloc) \
   maxpool1x2_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x2_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x2_64(_Lx, _Ln, xloc, yloc) \
   maxpool1x2_64(_Lx, _Ln, xloc, yloc) \
}

static void _maxpool1x2_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _maxpool1x2_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_64(_Lx, _Ln, xloc, yloc);
   maxpool1x2_32(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool1x2_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_64(_Lx, _Ln, xloc, yloc);
}

static void _maxpool1x2_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_32(_Lx, _Ln, xloc, yloc);
}

static void _maxpool1x2_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_16(_Lx, _Ln, xloc, yloc);
   maxpool1x2_08(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool1x2_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_16(_Lx, _Ln, xloc, yloc);
}

static void _maxpool1x2_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x2_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool1x2_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool1x2_func_t maxpool1x2_fntab_32[] = {
   _maxpool1x2_00,
   _maxpool1x2_32,
   _maxpool1x2_64,
   _maxpool1x2_96,
};

static maxpool1x2_func_t maxpool1x2_fntab_08[] = {
   _maxpool1x2_00,
   _maxpool1x2_08,
   _maxpool1x2_16,
   _maxpool1x2_24,
};

static void _maxpool1x2_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (maxpool1x2_fntab_32[2])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (maxpool1x2_fntab_32[2])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool1x2_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool1x2_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      cnn_type_t wloc[8];
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + p]) ? (wloc[p]) : (xloc[_Lx + p]);};
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};
   }
}

static void _maxpool1x1_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   for (int p = 0; p < _nn; p++){yloc[p] = xloc[p];};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* C should be even number !!!                                                                            */
////////////////////////////////////////////////////////////////////////////////////////////////////////////
void maxpool2x2(
         int  M,
         int  N,
         int  C,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   int _Lx = C;
   int _Ln = C * N;

   cnn_type_t * xloc = px;
   cnn_type_t * yloc = py;
   for (int m = 0; m < M; m+=2)
   {
      for (int n = 0; n < N; n+=2)
      {
         xloc = px + ( ((m  ) * (N  ) * C) + (n  ) * C);
         yloc = py + ( ((m/2) * (N/2) * C) + (n/2) * C);

         _maxpool2x2_nn(C, _Lx, _Ln, xloc, yloc);
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define maxpool3x3_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +             p]) ? (wloc[p]) : (xloc[_Ln +             p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln +             p]) ? (wloc[p]) : (xloc[_Ln + _Ln +             p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool3x3_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   {  \
      cnn_type_t wloc[8];             \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                              \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +             p]) ? (wloc[p]) : (xloc[_Ln +             p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Lx +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln +             p]) ? (wloc[p]) : (xloc[_Ln + _Ln +             p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx +       p]);};  \
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx + _Lx + p]);};  \
      \
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
   }  \
}

#define maxpool3x3_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x3_08(_Lx, _Ln, xloc, yloc)      \
   maxpool3x3_08(_Lx, _Ln, xloc, yloc)      \
}

#define maxpool3x3_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x3_16(_Lx, _Ln, xloc, yloc)      \
   maxpool3x3_16(_Lx, _Ln, xloc, yloc)      \
}

#define maxpool3x3_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x3_32(_Lx, _Ln, xloc, yloc)      \
   maxpool3x3_32(_Lx, _Ln, xloc, yloc)      \
}

#define maxpool3x3_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x3_64(_Lx, _Ln, xloc, yloc)       \
   maxpool3x3_64(_Lx, _Ln, xloc, yloc)       \
}

#define maxpool3x3_256(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x3_128(_Lx, _Ln, xloc, yloc)      \
   maxpool3x3_128(_Lx, _Ln, xloc, yloc)      \
}

static void _maxpool3x3_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _maxpool3x3_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_256(_Lx, _Ln, xloc, yloc); 
   maxpool3x3_128(_Lx, _Ln, xloc, yloc); 
}

static void _maxpool3x3_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_256(_Lx, _Ln, xloc, yloc); 
}

static void _maxpool3x3_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_128(_Lx, _Ln, xloc, yloc); 
}

static void _maxpool3x3_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_64(_Lx, _Ln, xloc, yloc);
   maxpool3x3_32(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool3x3_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_64(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_32(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_16(_Lx, _Ln, xloc, yloc);
   maxpool3x3_08(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool3x3_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_16(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_08(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x3_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x3_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool3x3_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool3x3_func_t maxpool3x3_fntab_128[] = {
   _maxpool3x3_00,
   _maxpool3x3_128,
   _maxpool3x3_256,
   _maxpool3x3_384,
};

static maxpool3x3_func_t maxpool3x3_fntab_32[] = {
   _maxpool3x3_00,
   _maxpool3x3_32,
   _maxpool3x3_64,
   _maxpool3x3_96,
};

static maxpool3x3_func_t maxpool3x3_fntab_08[] = {
   _maxpool3x3_00,
   _maxpool3x3_08,
   _maxpool3x3_16,
   _maxpool3x3_24,
};

static maxpool3x3_func_t maxpool3x3_fntab_00[] = {
   _maxpool3x3_00,
   _maxpool3x3_01,
   _maxpool3x3_02,
   _maxpool3x3_03,
   _maxpool3x3_04,
   _maxpool3x3_05,
   _maxpool3x3_06,
   _maxpool3x3_07,
};

static void _maxpool3x3_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (maxpool3x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (maxpool3x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }

   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (maxpool3x3_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool3x3_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool3x3_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      (maxpool3x3_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define maxpool3x2_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + p]) ? (wloc[p]) : (xloc[_Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln +       p]) ? (wloc[p]) : (xloc[_Ln + _Ln +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool3x2_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + p]) ? (wloc[p]) : (xloc[_Lx + p]);};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + p]);};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln +       p]) ? (wloc[p]) : (xloc[_Ln + _Ln +       p]);};  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + _Lx + p]);};  \
   \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
}

#define maxpool3x2_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x2_08(_Lx, _Ln, xloc, yloc) \
   maxpool3x2_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x2_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x2_16(_Lx, _Ln, xloc, yloc) \
   maxpool3x2_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x2_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x2_32(_Lx, _Ln, xloc, yloc) \
   maxpool3x2_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x2_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x2_64(_Lx, _Ln, xloc, yloc) \
   maxpool3x2_64(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x2_256(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x2_128(_Lx, _Ln, xloc, yloc) \
   maxpool3x2_128(_Lx, _Ln, xloc, yloc) \
}

static void _maxpool3x2_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _maxpool3x2_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_256(_Lx, _Ln, xloc, yloc); 
   maxpool3x2_128(_Lx, _Ln, xloc, yloc); 
}

static void _maxpool3x2_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_256(_Lx, _Ln, xloc, yloc); 
}

static void _maxpool3x2_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_128(_Lx, _Ln, xloc, yloc); 
}

static void _maxpool3x2_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_64(_Lx, _Ln, xloc, yloc);
   maxpool3x2_32(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool3x2_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_64(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_32(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_16(_Lx, _Ln, xloc, yloc);
   maxpool3x2_08(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool3x2_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_16(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_08(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x2_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x2_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool3x2_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool3x2_func_t maxpool3x2_fntab_128[] = {
   _maxpool3x2_00,
   _maxpool3x2_128,
   _maxpool3x2_256,
   _maxpool3x2_384,
};

static maxpool3x2_func_t maxpool3x2_fntab_32[] = {
   _maxpool3x2_00,
   _maxpool3x2_32,
   _maxpool3x2_64,
   _maxpool3x2_96,
};

static maxpool3x2_func_t maxpool3x2_fntab_08[] = {
   _maxpool3x2_00,
   _maxpool3x2_08,
   _maxpool3x2_16,
   _maxpool3x2_24,
};

static maxpool3x2_func_t maxpool3x2_fntab_00[] = {
   _maxpool3x2_00,
   _maxpool3x2_01,
   _maxpool3x2_02,
   _maxpool3x2_03,
   _maxpool3x2_04,
   _maxpool3x2_05,
   _maxpool3x2_06,
   _maxpool3x2_07,
};

static void _maxpool3x2_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (maxpool3x2_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (maxpool3x2_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }

   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (maxpool3x2_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool3x2_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool3x2_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (maxpool3x2_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define maxpool3x1_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool3x1_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +       p]) ? (wloc[p]) : (xloc[_Ln +       p]);};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Ln + p]) ? (wloc[p]) : (xloc[_Ln + _Ln + p]);};  \
   \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
}

#define maxpool3x1_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x1_08(_Lx, _Ln, xloc, yloc) \
   maxpool3x1_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x1_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x1_16(_Lx, _Ln, xloc, yloc) \
   maxpool3x1_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x1_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x1_32(_Lx, _Ln, xloc, yloc) \
   maxpool3x1_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x1_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x1_64(_Lx, _Ln, xloc, yloc) \
   maxpool3x1_64(_Lx, _Ln, xloc, yloc) \
}

#define maxpool3x1_256(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool3x1_128(_Lx, _Ln, xloc, yloc) \
   maxpool3x1_128(_Lx, _Ln, xloc, yloc) \
}

static void _maxpool3x1_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _maxpool3x1_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_256(_Lx, _Ln, xloc, yloc);
   maxpool3x1_128(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_256(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_128(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_64(_Lx, _Ln, xloc, yloc);
   maxpool3x1_32(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool3x1_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_64(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_32(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_16(_Lx, _Ln, xloc, yloc);
   maxpool3x1_08(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool3x1_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_16(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_08(_Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _maxpool3x1_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool3x1_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool3x1_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool3x1_func_t maxpool3x1_fntab_128[] = {
   _maxpool3x1_00,
   _maxpool3x1_128,
   _maxpool3x1_256,
   _maxpool3x1_384,
};

static maxpool3x1_func_t maxpool3x1_fntab_32[] = {
   _maxpool3x1_00,
   _maxpool3x1_32,
   _maxpool3x1_64,
   _maxpool3x1_96,
};

static maxpool3x1_func_t maxpool3x1_fntab_08[] = {
   _maxpool3x1_00,
   _maxpool3x1_08,
   _maxpool3x1_16,
   _maxpool3x1_24,
};

static maxpool3x1_func_t maxpool3x1_fntab_00[] = {
   _maxpool3x1_00,
   _maxpool3x1_01,
   _maxpool3x1_02,
   _maxpool3x1_03,
   _maxpool3x1_04,
   _maxpool3x1_05,
   _maxpool3x1_06,
   _maxpool3x1_07,
};

static void _maxpool3x1_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (maxpool3x1_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (maxpool3x1_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }
   
   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (maxpool3x1_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool3x1_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool3x1_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (maxpool3x1_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define maxpool2x3_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln +             p]) ? (wloc[p]) : (xloc[_Ln +             p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool2x3_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln +             p]) ? (wloc[p]) : (xloc[_Ln +             p]);};  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx +       p]) ? (wloc[p]) : (xloc[_Ln + _Lx +       p]);};  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + _Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Ln + _Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
}

#define maxpool2x3_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x3_08(_Lx, _Ln, xloc, yloc) \
   maxpool2x3_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x3_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x3_16(_Lx, _Ln, xloc, yloc) \
   maxpool2x3_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x3_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x3_32(_Lx, _Ln, xloc, yloc) \
   maxpool2x3_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x3_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x3_64(_Lx, _Ln, xloc, yloc) \
   maxpool2x3_64(_Lx, _Ln, xloc, yloc) \
}

#define maxpool2x3_256(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool2x3_128(_Lx, _Ln, xloc, yloc) \
   maxpool2x3_128(_Lx, _Ln, xloc, yloc) \
}

static void _maxpool2x3_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _maxpool2x3_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_256(_Lx, _Ln, xloc, yloc);
   maxpool2x3_128(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x3_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_128(_Lx, _Ln, xloc, yloc);
   maxpool2x3_128(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x3_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_64(_Lx, _Ln, xloc, yloc);
   maxpool2x3_64(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x3_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_64(_Lx, _Ln, xloc, yloc);
   maxpool2x3_32(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x3_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_64(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_32(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_16(_Lx, _Ln, xloc, yloc);
   maxpool2x3_08(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool2x3_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_16(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_08(_Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _maxpool2x3_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool2x3_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool2x3_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool2x3_func_t maxpool2x3_fntab_128[] = {
   _maxpool2x3_00,
   _maxpool2x3_128,
   _maxpool2x3_256,
   _maxpool2x3_384,
};

static maxpool2x3_func_t maxpool2x3_fntab_32[] = {
   _maxpool2x3_00,
   _maxpool2x3_32,
   _maxpool2x3_64,
   _maxpool2x3_96,
};

static maxpool2x3_func_t maxpool2x3_fntab_08[] = {
   _maxpool2x3_00,
   _maxpool2x3_08,
   _maxpool2x3_16,
   _maxpool2x3_24,
};

static maxpool2x3_func_t maxpool2x3_fntab_00[] = {
   _maxpool2x3_00,
   _maxpool2x3_01,
   _maxpool2x3_02,
   _maxpool2x3_03,
   _maxpool2x3_04,
   _maxpool2x3_05,
   _maxpool2x3_06,
   _maxpool2x3_07,
};

static void _maxpool2x3_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (maxpool2x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (maxpool2x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }
   
   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (maxpool2x3_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool2x3_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool2x3_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (maxpool2x3_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define maxpool1x3_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p];};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define maxpool1x3_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx +       p]) ? (wloc[p]) : (xloc[_Lx +       p]);};  \
   for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + _Lx + p]) ? (wloc[p]) : (xloc[_Lx + _Lx + p]);};  \
   \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};  \
}

#define maxpool1x3_16(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x3_08(_Lx, _Ln, xloc, yloc) \
   maxpool1x3_08(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x3_32(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x3_16(_Lx, _Ln, xloc, yloc) \
   maxpool1x3_16(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x3_64(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x3_32(_Lx, _Ln, xloc, yloc) \
   maxpool1x3_32(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x3_128(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x3_64(_Lx, _Ln, xloc, yloc) \
   maxpool1x3_64(_Lx, _Ln, xloc, yloc) \
}

#define maxpool1x3_256(_Lx, _Ln, xloc, yloc) \
{  \
   maxpool1x3_128(_Lx, _Ln, xloc, yloc) \
   maxpool1x3_128(_Lx, _Ln, xloc, yloc) \
}

static void _maxpool1x3_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _maxpool1x3_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_256(_Lx, _Ln, xloc, yloc);
   maxpool1x3_128(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool1x3_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_128(_Lx, _Ln, xloc, yloc);
   maxpool1x3_128(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool1x3_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_64(_Lx, _Ln, xloc, yloc);
   maxpool1x3_64(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool1x3_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_64(_Lx, _Ln, xloc, yloc);
   maxpool1x3_32(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool1x3_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_64(_Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_32(_Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_16(_Lx, _Ln, xloc, yloc);
   maxpool1x3_08(_Lx, _Ln, xloc, yloc);   
}

static void _maxpool1x3_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_16(_Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_08(_Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _maxpool1x3_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   maxpool1x3_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*maxpool1x3_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static maxpool1x3_func_t maxpool1x3_fntab_128[] = {
   _maxpool1x3_00,
   _maxpool1x3_128,
   _maxpool1x3_256,
   _maxpool1x3_384,
};

static maxpool1x3_func_t maxpool1x3_fntab_32[] = {
   _maxpool1x3_00,
   _maxpool1x3_32,
   _maxpool1x3_64,
   _maxpool1x3_96,
};

static maxpool1x3_func_t maxpool1x3_fntab_08[] = {
   _maxpool1x3_00,
   _maxpool1x3_08,
   _maxpool1x3_16,
   _maxpool1x3_24,
};

static maxpool1x3_func_t maxpool1x3_fntab_00[] = {
   _maxpool1x3_00,
   _maxpool1x3_01,
   _maxpool1x3_02,
   _maxpool1x3_03,
   _maxpool1x3_04,
   _maxpool1x3_05,
   _maxpool1x3_06,
   _maxpool1x3_07,
};

static void _maxpool1x3_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (maxpool1x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (maxpool1x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }

   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (maxpool1x3_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (maxpool1x3_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (maxpool1x3_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      (maxpool1x3_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* (S*C) % 4 = 0 !! */
void maxpool3x3(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   int _Lx = C;
   int _Ln = C * N;

   cnn_type_t * xloc = px;
   cnn_type_t * yloc = py;
   
   tf_assert( ((S*C)&3) == 0 );
   
   int m;
   for (m = 0; m < (M-3+1); m+=S)
   {
      int n;

      xloc = px + m * N * C;

      for (n = 0; n < (N-3+1); n+=S)
      {
         _maxpool3x3_nn(C, _Lx, _Ln, xloc, yloc);
         yloc = yloc + C;
         xloc = xloc + S * C;
      }

      /* skip the reset part if ((N-3)%S == 0) */
      if ((n-S) == (N-3)){continue;};

      {
         switch (N-n)
         {
            case 2:
            _maxpool3x2_nn(C, _Lx, _Ln, xloc, yloc);
            break;
			
            case 1:
            _maxpool3x1_nn(C, _Lx, _Ln, xloc, yloc);
            break;
         }
         yloc = yloc + C;
      }
   }

   /* skip the reset part if ((N-3)%S == 0) */
   if ((m-S) == (M-3)){return;};

   if ((M-m) >= 3) printf("m = %d, M = %d\n", m, M);

   switch (M-m)
   {
      case 2:
      {
         int n;

         xloc = px + m * N * C;

         for (n = 0; n < (N-3+1); n+=S)
         {
            _maxpool2x3_nn(C, _Lx, _Ln, xloc, yloc);
            yloc = yloc + C;
            xloc = xloc + S * C;
         }

         /* skip the reset part if ((N-3)%S == 0) */
         if ((N-n) == 3) printf("n = %d, N = %d\n", n, N);

         if ((n-S) != (N-3))
         {
            switch (N-n)
            {
               case 2:
               _maxpool2x2_nn(C, _Lx, _Ln, xloc, yloc);
               break;

               case 1:
               _maxpool2x1_nn(C, _Lx, _Ln, xloc, yloc);
               break;
            }
            yloc = yloc + C;
         }
      }
      break;

      case 1:
      {
         int n;

         xloc = px + m * N * C;

         for (n = 0; n < (N-3+1); n+=S)
         {
            _maxpool1x3_nn(C, _Lx, _Ln, xloc, yloc);
            yloc = yloc + C;
            xloc = xloc + S * C;
         }

         /* skip the reset part if ((N-3)%S == 0) */
         // if ((N-n) == 3) printf("n = %d, N = %d\n", n, N);

         if ((n-S) != (N-3))
         {
            switch (N-n)
            {
               case 2:
               _maxpool1x2_nn(C, _Lx, _Ln, xloc, yloc);
               break;

               case 1:
               _maxpool1x1_nn(C, _Lx, _Ln, xloc, yloc);
               break;
            }
            yloc = yloc + C;
         }
      }
      break;
   }
}

/* maxpool3x3 - valid */
void maxpool3x3_(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   int _Lx = C;
   int _Ln = C * N;

   cnn_type_t * xloc = px;
   cnn_type_t * yloc = py;
   
   tf_assert( ((S*C)&3) == 0 );
   
   for (int m = 0; m < (M-3+1); m+=S)
   {
      xloc = px + m * N * C;

      for (int n = 0; n < (N-3+1); n+=S)
      {
         _maxpool3x3_nn(C, _Lx, _Ln, xloc, yloc);
         yloc = yloc + C;
         xloc = xloc + S * C;
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define avgpool2x2_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Lx +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx + p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p] * 0.250;};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define avgpool2x2_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Lx +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx + p];};  \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p] * 0.250;};  \
}

#define avgpool2x2_16(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x2_08(_Lx, _Ln, xloc, yloc) \
   avgpool2x2_08(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x2_32(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x2_16(_Lx, _Ln, xloc, yloc) \
   avgpool2x2_16(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x2_64(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x2_32(_Lx, _Ln, xloc, yloc) \
   avgpool2x2_32(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x2_128(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x2_64(_Lx, _Ln, xloc, yloc) \
   avgpool2x2_64(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x2_256(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x2_128(_Lx, _Ln, xloc, yloc) \
   avgpool2x2_128(_Lx, _Ln, xloc, yloc) \
}

static void _avgpool2x2_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _avgpool2x2_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_256(_Lx, _Ln, xloc, yloc);
   avgpool2x2_128(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x2_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_128(_Lx, _Ln, xloc, yloc);
   avgpool2x2_128(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x2_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_64(_Lx, _Ln, xloc, yloc);
   avgpool2x2_64(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x2_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_64(_Lx, _Ln, xloc, yloc);
   avgpool2x2_32(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x2_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_64(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_32(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_16(_Lx, _Ln, xloc, yloc);
   avgpool2x2_08(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x2_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_16(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_08(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x2_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x2_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*avgpool2x2_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static avgpool2x2_func_t avgpool2x2_fntab_128[] = {
   _avgpool2x2_00,
   _avgpool2x2_128,
   _avgpool2x2_256,
   _avgpool2x2_384,
};

static avgpool2x2_func_t avgpool2x2_fntab_32[] = {
   _avgpool2x2_00,
   _avgpool2x2_32,
   _avgpool2x2_64,
   _avgpool2x2_96,
};

static avgpool2x2_func_t avgpool2x2_fntab_08[] = {
   _avgpool2x2_00,
   _avgpool2x2_08,
   _avgpool2x2_16,
   _avgpool2x2_24,
};

static avgpool2x2_func_t avgpool2x2_fntab_00[] = {
   _avgpool2x2_00,
   _avgpool2x2_01,
   _avgpool2x2_02,
   _avgpool2x2_03,
   _avgpool2x2_04,
   _avgpool2x2_05,
   _avgpool2x2_06,
   _avgpool2x2_07,
};

static void _avgpool2x2_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (avgpool2x2_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (avgpool2x2_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }

   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (avgpool2x2_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (avgpool2x2_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (avgpool2x2_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (avgpool2x2_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define avgpool2x1_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p] * 0.50;};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define avgpool2x1_16(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x1_08(_Lx, _Ln, xloc, yloc) \
   avgpool2x1_08(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x1_32(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x1_16(_Lx, _Ln, xloc, yloc) \
   avgpool2x1_16(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x1_64(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x1_32(_Lx, _Ln, xloc, yloc) \
   avgpool2x1_32(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x1_128(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x1_64(_Lx, _Ln, xloc, yloc) \
   avgpool2x1_64(_Lx, _Ln, xloc, yloc) \
}

static void _avgpool2x1_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _avgpool2x1_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x1_64(_Lx, _Ln, xloc, yloc);
   avgpool2x1_32(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x1_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x1_64(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x1_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x1_32(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x1_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x1_16(_Lx, _Ln, xloc, yloc);
   avgpool2x1_08(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x1_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x1_16(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x1_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x1_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*avgpool2x1_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static avgpool2x1_func_t avgpool2x1_fntab_32[] = {
   _avgpool2x1_00,
   _avgpool2x1_32,
   _avgpool2x1_64,
   _avgpool2x1_96,
};

static avgpool2x1_func_t avgpool2x1_fntab_08[] = {
   _avgpool2x1_00,
   _avgpool2x1_08,
   _avgpool2x1_16,
   _avgpool2x1_24,
};

void _avgpool2x1_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (avgpool2x1_fntab_32[2])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (avgpool2x1_fntab_32[2])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (avgpool2x1_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (avgpool2x1_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      cnn_type_t wloc[8];
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Ln + p]) ? (wloc[p]) : (xloc[_Ln + p]);};
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define avgpool1x2_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Lx + p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p] * 0.50;};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define avgpool1x2_16(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool1x2_08(_Lx, _Ln, xloc, yloc) \
   avgpool1x2_08(_Lx, _Ln, xloc, yloc) \
}

#define avgpool1x2_32(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool1x2_16(_Lx, _Ln, xloc, yloc) \
   avgpool1x2_16(_Lx, _Ln, xloc, yloc) \
}

#define avgpool1x2_64(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool1x2_32(_Lx, _Ln, xloc, yloc) \
   avgpool1x2_32(_Lx, _Ln, xloc, yloc) \
}

#define avgpool1x2_128(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool1x2_64(_Lx, _Ln, xloc, yloc) \
   avgpool1x2_64(_Lx, _Ln, xloc, yloc) \
}

static void _avgpool1x2_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _avgpool1x2_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x2_64(_Lx, _Ln, xloc, yloc);
   avgpool1x2_32(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool1x2_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x2_64(_Lx, _Ln, xloc, yloc);
}

static void _avgpool1x2_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x2_32(_Lx, _Ln, xloc, yloc);
}

static void _avgpool1x2_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x2_16(_Lx, _Ln, xloc, yloc);
   avgpool1x2_08(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool1x2_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x2_16(_Lx, _Ln, xloc, yloc);
}

static void _avgpool1x2_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x2_08(_Lx, _Ln, xloc, yloc);
}

typedef void (*avgpool1x2_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static avgpool1x2_func_t avgpool1x2_fntab_32[] = {
   _avgpool1x2_00,
   _avgpool1x2_32,
   _avgpool1x2_64,
   _avgpool1x2_96,
};

static avgpool1x2_func_t avgpool1x2_fntab_08[] = {
   _avgpool1x2_00,
   _avgpool1x2_08,
   _avgpool1x2_16,
   _avgpool1x2_24,
};

void _avgpool1x2_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (avgpool1x2_fntab_32[2])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;

         (avgpool1x2_fntab_32[2])(_Lx, _Ln, xloc, yloc);
         xloc += 64;
         yloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (avgpool1x2_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (avgpool1x2_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      cnn_type_t wloc[8];
      for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};
      for (int p = 0; p < _nn; p++){wloc[p] = (wloc[p] > xloc[_Lx + p]) ? (wloc[p]) : (xloc[_Lx + p]);};
      for (int p = 0; p < _nn; p++){yloc[p] = wloc[p];};
   }
}

void _avgpool1x1_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   for (int p = 0; p < _nn; p++){yloc[p] = xloc[p];};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* C should be even number !!!                                                                            */
////////////////////////////////////////////////////////////////////////////////////////////////////////////
void avgpool2x2(
         int  M,
         int  N,
         int  C,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   int _Lx = C;
   int _Ln = C * N;

   cnn_type_t * xloc = px;
   cnn_type_t * yloc = py;
   for (int m = 0; m < M; m+=2)
   {
      for (int n = 0; n < N; n+=2)
      {
         xloc = px + ( ((m  ) * (N  ) * C) + (n  ) * C);
         yloc = py + ( ((m/2) * (N/2) * C) + (n/2) * C);

         _avgpool2x2_nn(C, _Lx, _Ln, xloc, yloc);
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define avgpool3x3_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                        \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Lx +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Lx + _Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln +             p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx + _Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln +             p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln + _Lx +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln + _Lx + _Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p] / 9.0;};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define avgpool3x3_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                        \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Lx +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Lx + _Lx + p];};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln +             p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx + _Lx + p];};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln +             p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln + _Lx +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln + _Lx + _Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p] / 9.0;};  \
}

#define avgpool3x3_16(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x3_08(_Lx, _Ln, xloc, yloc)      \
   avgpool3x3_08(_Lx, _Ln, xloc, yloc)      \
}

#define avgpool3x3_32(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x3_16(_Lx, _Ln, xloc, yloc)      \
   avgpool3x3_16(_Lx, _Ln, xloc, yloc)      \
}

#define avgpool3x3_64(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x3_32(_Lx, _Ln, xloc, yloc)      \
   avgpool3x3_32(_Lx, _Ln, xloc, yloc)      \
}

#define avgpool3x3_128(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x3_64(_Lx, _Ln, xloc, yloc)       \
   avgpool3x3_64(_Lx, _Ln, xloc, yloc)       \
}

#define avgpool3x3_256(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x3_128(_Lx, _Ln, xloc, yloc)      \
   avgpool3x3_128(_Lx, _Ln, xloc, yloc)      \
}

static void _avgpool3x3_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _avgpool3x3_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_256(_Lx, _Ln, xloc, yloc); 
   avgpool3x3_128(_Lx, _Ln, xloc, yloc); 
}

static void _avgpool3x3_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_256(_Lx, _Ln, xloc, yloc); 
}

static void _avgpool3x3_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_128(_Lx, _Ln, xloc, yloc); 
}

static void _avgpool3x3_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_64(_Lx, _Ln, xloc, yloc);
   avgpool3x3_32(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool3x3_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_64(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_32(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_16(_Lx, _Ln, xloc, yloc);
   avgpool3x3_08(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool3x3_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_16(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_08(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x3_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x3_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*avgpool3x3_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static avgpool3x3_func_t avgpool3x3_fntab_128[] = {
   _avgpool3x3_00,
   _avgpool3x3_128,
   _avgpool3x3_256,
   _avgpool3x3_384,
};

static avgpool3x3_func_t avgpool3x3_fntab_32[] = {
   _avgpool3x3_00,
   _avgpool3x3_32,
   _avgpool3x3_64,
   _avgpool3x3_96,
};

static avgpool3x3_func_t avgpool3x3_fntab_08[] = {
   _avgpool3x3_00,
   _avgpool3x3_08,
   _avgpool3x3_16,
   _avgpool3x3_24,
};

static avgpool3x3_func_t avgpool3x3_fntab_00[] = {
   _avgpool3x3_00,
   _avgpool3x3_01,
   _avgpool3x3_02,
   _avgpool3x3_03,
   _avgpool3x3_04,
   _avgpool3x3_05,
   _avgpool3x3_06,
   _avgpool3x3_07,
};

static void _avgpool3x3_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (avgpool3x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (avgpool3x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }

   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (avgpool3x3_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (avgpool3x3_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (avgpool3x3_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      (avgpool3x3_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define avgpool3x2_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln + _Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p] * 0.1666666666667;};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define avgpool3x2_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Lx + p];};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx + p];};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln + _Lx + p];};  \
   \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p] * 0.1666666666667;};  \
}

#define avgpool3x2_16(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x2_08(_Lx, _Ln, xloc, yloc) \
   avgpool3x2_08(_Lx, _Ln, xloc, yloc) \
}

#define avgpool3x2_32(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x2_16(_Lx, _Ln, xloc, yloc) \
   avgpool3x2_16(_Lx, _Ln, xloc, yloc) \
}

#define avgpool3x2_64(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x2_32(_Lx, _Ln, xloc, yloc) \
   avgpool3x2_32(_Lx, _Ln, xloc, yloc) \
}

#define avgpool3x2_128(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x2_64(_Lx, _Ln, xloc, yloc) \
   avgpool3x2_64(_Lx, _Ln, xloc, yloc) \
}

#define avgpool3x2_256(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x2_128(_Lx, _Ln, xloc, yloc) \
   avgpool3x2_128(_Lx, _Ln, xloc, yloc) \
}

static void _avgpool3x2_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _avgpool3x2_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_256(_Lx, _Ln, xloc, yloc); 
   avgpool3x2_128(_Lx, _Ln, xloc, yloc); 
}

static void _avgpool3x2_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_256(_Lx, _Ln, xloc, yloc); 
}

static void _avgpool3x2_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_128(_Lx, _Ln, xloc, yloc); 
}

static void _avgpool3x2_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_64(_Lx, _Ln, xloc, yloc);
   avgpool3x2_32(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool3x2_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_64(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_32(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_16(_Lx, _Ln, xloc, yloc);
   avgpool3x2_08(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool3x2_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_16(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_08(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x2_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x2_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*avgpool3x2_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static avgpool3x2_func_t avgpool3x2_fntab_128[] = {
   _avgpool3x2_00,
   _avgpool3x2_128,
   _avgpool3x2_256,
   _avgpool3x2_384,
};

static avgpool3x2_func_t avgpool3x2_fntab_32[] = {
   _avgpool3x2_00,
   _avgpool3x2_32,
   _avgpool3x2_64,
   _avgpool3x2_96,
};

static avgpool3x2_func_t avgpool3x2_fntab_08[] = {
   _avgpool3x2_00,
   _avgpool3x2_08,
   _avgpool3x2_16,
   _avgpool3x2_24,
};

static avgpool3x2_func_t avgpool3x2_fntab_00[] = {
   _avgpool3x2_00,
   _avgpool3x2_01,
   _avgpool3x2_02,
   _avgpool3x2_03,
   _avgpool3x2_04,
   _avgpool3x2_05,
   _avgpool3x2_06,
   _avgpool3x2_07,
};

static void _avgpool3x2_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (avgpool3x2_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (avgpool3x2_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }

   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (avgpool3x2_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (avgpool3x2_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (avgpool3x2_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (avgpool3x2_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define avgpool3x1_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln +       p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p] * 0.33333333333333333333;};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define avgpool3x1_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln +       p];};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Ln + p];};  \
   \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p] * 0.33333333333333333333;};  \
}

#define avgpool3x1_16(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x1_08(_Lx, _Ln, xloc, yloc) \
   avgpool3x1_08(_Lx, _Ln, xloc, yloc) \
}

#define avgpool3x1_32(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x1_16(_Lx, _Ln, xloc, yloc) \
   avgpool3x1_16(_Lx, _Ln, xloc, yloc) \
}

#define avgpool3x1_64(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x1_32(_Lx, _Ln, xloc, yloc) \
   avgpool3x1_32(_Lx, _Ln, xloc, yloc) \
}

#define avgpool3x1_128(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x1_64(_Lx, _Ln, xloc, yloc) \
   avgpool3x1_64(_Lx, _Ln, xloc, yloc) \
}

#define avgpool3x1_256(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool3x1_128(_Lx, _Ln, xloc, yloc) \
   avgpool3x1_128(_Lx, _Ln, xloc, yloc) \
}

static void _avgpool3x1_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _avgpool3x1_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_256(_Lx, _Ln, xloc, yloc);
   avgpool3x1_128(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_256(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_128(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_64(_Lx, _Ln, xloc, yloc);
   avgpool3x1_32(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool3x1_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_64(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_32(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_16(_Lx, _Ln, xloc, yloc);
   avgpool3x1_08(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool3x1_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_16(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_08(_Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _avgpool3x1_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool3x1_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*avgpool3x1_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static avgpool3x1_func_t avgpool3x1_fntab_128[] = {
   _avgpool3x1_00,
   _avgpool3x1_128,
   _avgpool3x1_256,
   _avgpool3x1_384,
};

static avgpool3x1_func_t avgpool3x1_fntab_32[] = {
   _avgpool3x1_00,
   _avgpool3x1_32,
   _avgpool3x1_64,
   _avgpool3x1_96,
};

static avgpool3x1_func_t avgpool3x1_fntab_08[] = {
   _avgpool3x1_00,
   _avgpool3x1_08,
   _avgpool3x1_16,
   _avgpool3x1_24,
};

static avgpool3x1_func_t avgpool3x1_fntab_00[] = {
   _avgpool3x1_00,
   _avgpool3x1_01,
   _avgpool3x1_02,
   _avgpool3x1_03,
   _avgpool3x1_04,
   _avgpool3x1_05,
   _avgpool3x1_06,
   _avgpool3x1_07,
};

void _avgpool3x1_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (avgpool3x1_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (avgpool3x1_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }
   
   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (avgpool3x1_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (avgpool3x1_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (avgpool3x1_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (avgpool3x1_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define avgpool2x3_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Lx +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Lx + _Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln +             p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx + _Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p] * 0.166666666666666667;};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define avgpool2x3_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] > xloc[_Lx +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] > xloc[_Lx + _Lx + p];};  \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln +             p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Ln + _Lx + _Lx + p];};  \
   \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p] * 0.166666666666666667;};  \
}

#define avgpool2x3_16(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x3_08(_Lx, _Ln, xloc, yloc) \
   avgpool2x3_08(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x3_32(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x3_16(_Lx, _Ln, xloc, yloc) \
   avgpool2x3_16(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x3_64(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x3_32(_Lx, _Ln, xloc, yloc) \
   avgpool2x3_32(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x3_128(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x3_64(_Lx, _Ln, xloc, yloc) \
   avgpool2x3_64(_Lx, _Ln, xloc, yloc) \
}

#define avgpool2x3_256(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool2x3_128(_Lx, _Ln, xloc, yloc) \
   avgpool2x3_128(_Lx, _Ln, xloc, yloc) \
}

static void _avgpool2x3_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _avgpool2x3_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_256(_Lx, _Ln, xloc, yloc);
   avgpool2x3_128(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x3_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_128(_Lx, _Ln, xloc, yloc);
   avgpool2x3_128(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x3_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_64(_Lx, _Ln, xloc, yloc);
   avgpool2x3_64(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x3_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_64(_Lx, _Ln, xloc, yloc);
   avgpool2x3_32(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x3_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_64(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_32(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_16(_Lx, _Ln, xloc, yloc);
   avgpool2x3_08(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool2x3_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_16(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_08(_Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _avgpool2x3_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool2x3_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*avgpool2x3_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static avgpool2x3_func_t avgpool2x3_fntab_128[] = {
   _avgpool2x3_00,
   _avgpool2x3_128,
   _avgpool2x3_256,
   _avgpool2x3_384,
};

static avgpool2x3_func_t avgpool2x3_fntab_32[] = {
   _avgpool2x3_00,
   _avgpool2x3_32,
   _avgpool2x3_64,
   _avgpool2x3_96,
};

static avgpool2x3_func_t avgpool2x3_fntab_08[] = {
   _avgpool2x3_00,
   _avgpool2x3_08,
   _avgpool2x3_16,
   _avgpool2x3_24,
};

static avgpool2x3_func_t avgpool2x3_fntab_00[] = {
   _avgpool2x3_00,
   _avgpool2x3_01,
   _avgpool2x3_02,
   _avgpool2x3_03,
   _avgpool2x3_04,
   _avgpool2x3_05,
   _avgpool2x3_06,
   _avgpool2x3_07,
};

static void _avgpool2x3_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (avgpool2x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (avgpool2x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }
   
   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (avgpool2x3_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (avgpool2x3_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (avgpool2x3_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   /* 0 - 7 */
   {
      (avgpool2x3_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define avgpool1x3_08(_Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Lx +       p];};  \
   for (int p = 0; p < CNN_BCHSIZ; p++){wloc[p] = wloc[p] + xloc[_Lx + _Lx + p];};  \
   \
   for (int p = 0; p < CNN_BCHSIZ; p++){yloc[p] = wloc[p] * 0.33333333333333333333333;};  \
   xloc += CNN_BCHSIZ; \
   yloc += CNN_BCHSIZ; \
}

#define avgpool1x3_mm(_nn, _Lx, _Ln, xloc, yloc) \
{  \
   cnn_type_t wloc[CNN_BCHSIZ];             \
   \
   for (int p = 0; p < _nn; p++){wloc[p] = xloc[p];};                                                              \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Lx +       p];};  \
   for (int p = 0; p < _nn; p++){wloc[p] = wloc[p] + xloc[_Lx + _Lx + p];};  \
   \
   for (int p = 0; p < _nn; p++){yloc[p] = wloc[p] * 0.33333333333333333333333;};  \
}

#define avgpool1x3_16(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool1x3_08(_Lx, _Ln, xloc, yloc) \
   avgpool1x3_08(_Lx, _Ln, xloc, yloc) \
}

#define avgpool1x3_32(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool1x3_16(_Lx, _Ln, xloc, yloc) \
   avgpool1x3_16(_Lx, _Ln, xloc, yloc) \
}

#define avgpool1x3_64(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool1x3_32(_Lx, _Ln, xloc, yloc) \
   avgpool1x3_32(_Lx, _Ln, xloc, yloc) \
}

#define avgpool1x3_128(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool1x3_64(_Lx, _Ln, xloc, yloc) \
   avgpool1x3_64(_Lx, _Ln, xloc, yloc) \
}

#define avgpool1x3_256(_Lx, _Ln, xloc, yloc) \
{  \
   avgpool1x3_128(_Lx, _Ln, xloc, yloc) \
   avgpool1x3_128(_Lx, _Ln, xloc, yloc) \
}

static void _avgpool1x3_00(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
}

static void _avgpool1x3_384(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_256(_Lx, _Ln, xloc, yloc);
   avgpool1x3_128(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool1x3_256(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_128(_Lx, _Ln, xloc, yloc);
   avgpool1x3_128(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool1x3_128(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_64(_Lx, _Ln, xloc, yloc);
   avgpool1x3_64(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool1x3_96(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_64(_Lx, _Ln, xloc, yloc);
   avgpool1x3_32(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool1x3_64(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_64(_Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_32(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_32(_Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_24(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_16(_Lx, _Ln, xloc, yloc);
   avgpool1x3_08(_Lx, _Ln, xloc, yloc);   
}

static void _avgpool1x3_16(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_16(_Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_08(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_08(_Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_07(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_mm(7, _Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_06(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_mm(6, _Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_05(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_mm(5, _Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_04(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_mm(4, _Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_03(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_mm(3, _Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_02(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_mm(2, _Lx, _Ln, xloc, yloc);
}

static void _avgpool1x3_01(
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   avgpool1x3_mm(1, _Lx, _Ln, xloc, yloc);
}

typedef void (*avgpool1x3_func_t)(int, int, const cnn_type_t *, cnn_type_t *);
static avgpool1x3_func_t avgpool1x3_fntab_128[] = {
   _avgpool1x3_00,
   _avgpool1x3_128,
   _avgpool1x3_256,
   _avgpool1x3_384,
};

static avgpool1x3_func_t avgpool1x3_fntab_32[] = {
   _avgpool1x3_00,
   _avgpool1x3_32,
   _avgpool1x3_64,
   _avgpool1x3_96,
};

static avgpool1x3_func_t avgpool1x3_fntab_08[] = {
   _avgpool1x3_00,
   _avgpool1x3_08,
   _avgpool1x3_16,
   _avgpool1x3_24,
};

static avgpool1x3_func_t avgpool1x3_fntab_00[] = {
   _avgpool1x3_00,
   _avgpool1x3_01,
   _avgpool1x3_02,
   _avgpool1x3_03,
   _avgpool1x3_04,
   _avgpool1x3_05,
   _avgpool1x3_06,
   _avgpool1x3_07,
};

void _avgpool1x3_nn(
         int                   _nn,
         int                   _Lx,
         int                   _Ln,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _yloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * yloc = __builtin_assume_aligned(_yloc, 16);

   if (_nn >= 512)
   {
      int nb = (_nn >> 9);
      for (int i = 0; i < nb; i++)
      {
         (avgpool1x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;

         (avgpool1x3_fntab_128[2])(_Lx, _Ln, xloc, yloc);
         xloc += 256;
         yloc += 256;
      }
      _nn -= (nb << 9);
   }

   if (_nn >= 128)
   {
      int nb = _nn >> 7;
      (avgpool1x3_fntab_128[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 7);
      yloc += (nb << 7);
      _nn  -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (avgpool1x3_fntab_32[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 5);
      yloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (avgpool1x3_fntab_08[nb])(_Lx, _Ln, xloc, yloc);
      xloc += (nb << 3);
      yloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   {
      (avgpool1x3_fntab_00[_nn & 7])(_Lx, _Ln, xloc, yloc);
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* avgpool3x3 (stride = 1, same) */
void avgpool3x3_p1(
         int  M,
         int  N,
         int  C,
   const cnn_type_t * restrict __px,
         cnn_type_t * restrict __py
)
{
   cnn_type_t * _px = __builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __builtin_assume_aligned(__py, 16);

   int _Ln = C * N;
   int _Lx = C ;

   tf_assert((C & (CNN_BCHSIZ - 1)) == 0);

   {
      int m = 0;

      cnn_type_t * xloc = _px + (m-0) * N * C;
      cnn_type_t * yloc = _py + (m-0) * N * C;

      {
         _avgpool2x2_nn(C, _Lx, _Ln, xloc, yloc);

         yloc += C;
      }

      for (int n = 1; n < (N-1); n++) 
      {
         _avgpool2x3_nn(C, _Lx, _Ln, xloc, yloc); 

         xloc += C;        
         yloc += C;
      }

      {
         _avgpool2x2_nn(C, _Lx, _Ln, xloc, yloc); 

         xloc += C;        
         yloc += C;
      }
   }

   for (int m = 1; m < (M-1); m++)
   {
      cnn_type_t * xloc = _px + (m-1) * N * C;
      cnn_type_t * yloc = _py + (m-0) * N * C;

      {
         _avgpool3x2_nn(C, _Lx, _Ln, xloc, yloc);

         yloc += C;
      }

      for (int n = 1; n < (N-1); n++) 
      {
         _avgpool3x3_nn(C, _Lx, _Ln, xloc, yloc); 

         xloc += C;        
         yloc += C;
      }

      {
         _avgpool3x2_nn(C, _Lx, _Ln, xloc, yloc); 

         xloc += C;        
         yloc += C;
      }
   }

   /* M - 1 */
   {
      int m = (M-1);

      cnn_type_t * xloc = _px + (m-1) * N * C;
      cnn_type_t * yloc = _py + (m-0) * N * C;

      {
         _avgpool2x2_nn(C, _Lx, _Ln, xloc, yloc);

         yloc += C;
      }

      for (int n = 1; n < (N-1); n++) 
      {
         _avgpool2x3_nn(C, _Lx, _Ln, xloc, yloc); 

         xloc += C;        
         yloc += C;
      }

      {
         _avgpool2x2_nn(C, _Lx, _Ln, xloc, yloc); 

         xloc += C;        
         yloc += C;
      }
   }
}

/* avgpool3x3 - valid */
void avgpool3x3_(
         int  M,
         int  N,
         int  C,
         int  S,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   int _Lx = C;
   int _Ln = C * N;

   cnn_type_t * xloc = px;
   cnn_type_t * yloc = py;
   
   tf_assert( ((S*C)&3) == 0 );
   
   for (int m = 0; m < (M-3+1); m+=S)
   {
      xloc = px + m * N * C;

      for (int n = 0; n < (N-3+1); n+=S)
      {
         _avgpool3x3_nn(C, _Lx, _Ln, xloc, yloc);
         yloc = yloc + C;
         xloc = xloc + S * C;
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define softmaxa_04(xloc, uloc, s)  \
{  \
   for (int p = 0; p < 4; p++){uloc[p] = exp(xloc[p]);};  \
   for (int p = 0; p < 4; p++){s[p]   += uloc[p];     };  \
   \
   xloc += 4;  \
   uloc += 4;  \
}

#define softmaxa_08(xloc, uloc, s)  \
{  \
   softmaxa_04(xloc, uloc, s);      \
   softmaxa_04(xloc, uloc, s);      \
}

#define softmaxa_16(xloc, uloc, s)  \
{  \
   softmaxa_08(xloc, uloc, s);      \
   softmaxa_08(xloc, uloc, s);      \
}

#define softmaxa_32(xloc, uloc, s)  \
{  \
   softmaxa_16(xloc, uloc, s);      \
   softmaxa_16(xloc, uloc, s);      \
}

#define softmaxa_64(xloc, uloc, s)  \
{  \
   softmaxa_32(xloc, uloc, s);      \
   softmaxa_32(xloc, uloc, s);      \
}

void _softmaxa_00(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
}

void _softmaxa_96(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_64(xloc, uloc, sloc);
   softmaxa_32(xloc, uloc, sloc);   
}

void _softmaxa_64(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_64(xloc, uloc, sloc);
}

void _softmaxa_32(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_32(xloc, uloc, sloc);
}

void _softmaxa_24(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_16(xloc, uloc, sloc);   
   softmaxa_08(xloc, uloc, sloc);   
}

void _softmaxa_16(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_16(xloc, uloc, sloc);   
}

void _softmaxa_08(
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   softmaxa_08(xloc, uloc, sloc);
}

typedef void (*softmaxa_func_t)(const cnn_type_t *, cnn_type_t *, cnn_type_t *);
static softmaxa_func_t softmaxa_fntab_32[] = {
   _softmaxa_00,
   _softmaxa_32,
   _softmaxa_64,
   _softmaxa_96,
};

static softmaxa_func_t softmaxa_fntab_08[] = {
   _softmaxa_00,
   _softmaxa_08,
   _softmaxa_16,
   _softmaxa_24,
};

static void _softmaxa_nn(
         int                   _nn,
   const cnn_type_t * restrict _xloc,
         cnn_type_t * restrict _uloc,
         cnn_type_t * restrict _sloc
   )
{
   cnn_type_t * xloc = __builtin_assume_aligned(_xloc, 16);
   cnn_type_t * uloc = __builtin_assume_aligned(_uloc, 16);
   cnn_type_t * sloc = __builtin_assume_aligned(_sloc, 16);

   if (_nn >= 128)
   {
      int nb = (_nn >> 7);
      for (int i = 0; i < nb; i++)
      {
         (softmaxa_fntab_32[1])(xloc, uloc, sloc);
         xloc += 64;
         uloc += 64;

         (softmaxa_fntab_32[1])(xloc, uloc, sloc);
         xloc += 64;
         uloc += 64;
      }
      _nn -= (nb << 7);
   }

   if (_nn >= 32)
   {
      int nb = _nn >> 5;
      (softmaxa_fntab_32[nb])(xloc, uloc, sloc);
      xloc += (nb << 5);
      uloc += (nb << 5);
      _nn  -= (nb << 5);
   }

   if (_nn >= 8)
   {
      int nb = _nn >> 3;
      (softmaxa_fntab_08[nb])(xloc, uloc, sloc);
      xloc += (nb << 3);
      uloc += (nb << 3);
      _nn  -= (nb << 3);
   }

   for (int n = 0; n < _nn; n++)
   {
      uloc[n    ]  = exp(xloc[n]);
      sloc[n & 3] +=    (uloc[n]);
   }
}

static void _softmaxb_nn(
         int                   _nn,         
         cnn_type_t * restrict _yloc,
   const cnn_type_t * restrict _uloc,
         cnn_type_t            _ss
   )
{
   _ss = 1.0 / _ss;

   mulpool2(_nn, _uloc, _ss, _yloc);
}

void softmaxpool(
         int  M,
         int  N,
         int  C,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   cnn_type_t * xloc = px;
   cnn_type_t * yloc = py;

   cnn_type_t   ss;
   cnn_type_t * uloc = _aligned_malloc(1024 * sizeof(cnn_type_t), 16);
   cnn_type_t * sloc = uloc + 768;


   for (int m = 0; m < M; m++)
   {
      for (int n = 0; n < N; n++)
      {
         sloc[0] = sloc[1] = sloc[2] = sloc[3] = 0;

         _softmaxa_nn(C, xloc, uloc, sloc); 

         ss = sloc[0] + sloc[1] + sloc[2] + sloc[3];
         
         _softmaxb_nn(C, yloc, uloc, ss);

         xloc += C;
         yloc += C;       
      }
   }

   _aligned_free(uloc);
}

void conv2d_concat(
         int  M,
         int  N,
         int  Cx,
         int  Cy,
         int  Sy,
   const cnn_type_t * restrict _px,
         cnn_type_t * restrict _py
)
{
   cnn_type_t * px = __builtin_assume_aligned(_px, 16);
   cnn_type_t * py = __builtin_assume_aligned(_py, 16);

   py = py + Sy;

   for (int m = 0; m < M; m++)
   {
      for (int n = 0; n < N; n++)
      {
         fastmcp(py, px, sizeof(cnn_type_t) * Cx);

         py += Cy;
         px += Cx;
      }
   }
}