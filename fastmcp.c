#include <stdint.h>
#include "cnntype.h"

#define fastmcp_01b(pdst, psrc) \
{  \
   int8_t * pd8 = (int8_t *)pdst;   \
   int8_t * ps8 = (int8_t *)psrc;   \
   *pd8++ = *ps8++;                 \
}

#define fastmcp_02b(pdst, psrc) \
{  \
   int16_t * pd16 = (int16_t *)pdst;   \
   int16_t * ps16 = (int16_t *)psrc;   \
   *pd16++ = *ps16++;                  \
   \
   pdst = (int32_t *)pd16;             \
   psrc = (int32_t *)ps16;             \
}

#define fastmcp_04b(pdst, psrc) \
{  \
   *pdst++ = *psrc++; \
}

#define fastmcp_08b(pdst, psrc) \
{  \
   *pdst++ = *psrc++; \
   *pdst++ = *psrc++; \
}

#define fastmcp_16b(pdst, psrc) \
{  \
   int32_t d[4];  \
   for (int p = 0; p < 4; p++){d[p] = *psrc++;}; \
   for (int p = 0; p < 4; p++){*pdst++ = d[p];}; \
}

#define fastmcp_32b(pdst, psrc) \
{  \
   int32_t d[8];  \
   for (int p = 0; p < 8; p++){d[p] = *psrc++;}; \
   for (int p = 0; p < 8; p++){*pdst++ = d[p];}; \
}

#define fastmcp_64b(pdst, psrc) \
{  \
   fastmcp_32b(pdst, psrc);     \
   fastmcp_32b(pdst, psrc);     \
}

#define fastmcp_128b(pdst, psrc) \
{  \
   fastmcp_64b(pdst, psrc);     \
   fastmcp_64b(pdst, psrc);     \
}

#define fastmcp_256b(pdst, psrc) \
{  \
   fastmcp_128b(pdst, psrc);     \
   fastmcp_128b(pdst, psrc);     \
}

#define fastmcp_512b(pdst, psrc) \
{  \
   fastmcp_256b(pdst, psrc);     \
   fastmcp_256b(pdst, psrc);     \
}

#define fastmcp_1K(pdst, psrc) \
{  \
   fastmcp_512b(pdst, psrc);     \
   fastmcp_512b(pdst, psrc);     \
}

#define fastmcp_2K(pdst, psrc) \
{  \
   fastmcp_1K(pdst, psrc);     \
   fastmcp_1K(pdst, psrc);     \
}

#define fastmcp_4K(pdst, psrc) \
{  \
   fastmcp_2K(pdst, psrc);     \
   fastmcp_2K(pdst, psrc);     \
}

static int _fastmcp_1b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_01b(pdst, psrc);

   return (1);
}

static int _fastmcp_2b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_02b(pdst, psrc);

   return (2);
}

static int _fastmcp_3b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_02b(pdst, psrc);
   fastmcp_01b(pdst, psrc);

   return (3);
}

static int _fastmcp_4b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_04b(pdst, psrc);

   return (4);
}

static int _fastmcp_5b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_04b(pdst, psrc);
   fastmcp_01b(pdst, psrc);

   return (5);
}

static int _fastmcp_6b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_04b(pdst, psrc);
   fastmcp_02b(pdst, psrc);

   return (6);
}

static int _fastmcp_7b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_04b(pdst, psrc);
   fastmcp_02b(pdst, psrc);
   fastmcp_01b(pdst, psrc);

   return (7);
}

static int _fastmcp_8b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_08b(pdst, psrc);

   return (8);
}

static int _fastmcp_9b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_08b(pdst, psrc);
   fastmcp_01b(pdst, psrc);

   return (9);
}


static int _fastmcp_10b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_08b(pdst, psrc);
   fastmcp_02b(pdst, psrc);

   return (10);
}

static int _fastmcp_11b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_08b(pdst, psrc);
   fastmcp_02b(pdst, psrc);
   fastmcp_01b(pdst, psrc);

   return (11);
}

static int _fastmcp_12b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_08b(pdst, psrc);
   fastmcp_04b(pdst, psrc);

   return (12);
}

static int _fastmcp_13b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_08b(pdst, psrc);
   fastmcp_04b(pdst, psrc);
   fastmcp_01b(pdst, psrc);

   return (13);
}

static int _fastmcp_14b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_08b(pdst, psrc);
   fastmcp_04b(pdst, psrc);
   fastmcp_02b(pdst, psrc);

   return (14);
}

static int _fastmcp_15b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_08b(pdst, psrc);
   fastmcp_04b(pdst, psrc);
   fastmcp_02b(pdst, psrc);
   fastmcp_01b(pdst, psrc);

   return (15);
}

static int _fastmcp_0b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   return (0);
}

BEGIN_NOINLINE static int DECLA_NOINLINE _fastmcp_4K(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_4K(pdst, psrc);

   return (0);
}

/*
static int _fastmcp_32K(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   return (0);
}

static int _fastmcp_28K(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   return (0);
}

static int _fastmcp_24K(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   return (0);
}

static int _fastmcp_20K(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   fastmcp_4K(pdst, psrc);

   return (0);
}

static int _fastmcp_16K(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   return (0);
}

static int _fastmcp_12K(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   return (0);
}

static int _fastmcp_8K(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_4K(pdst, psrc);
   fastmcp_4K(pdst, psrc);

   return (0);
}

static int _fastmcp_4K(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_4K(pdst, psrc);

   return (0);
}

*/

static int _fastmcp_3840b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_3584b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_3328b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_3072b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_2816b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_2560b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_2304b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_2048b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_1792b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_1536b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_1280b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_1024b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_768b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_512b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);
   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_256b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_256b(pdst, psrc);

   return (0);
}

static int _fastmcp_240b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_128b(pdst, psrc);
   fastmcp_64b(pdst, psrc);
   fastmcp_32b(pdst, psrc);
   fastmcp_16b(pdst, psrc);

   return (0);
}

static int _fastmcp_224b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_128b(pdst, psrc);
   fastmcp_64b(pdst, psrc);
   fastmcp_32b(pdst, psrc);

   return (0);
}

static int _fastmcp_208b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_128b(pdst, psrc);
   fastmcp_64b(pdst, psrc);
   fastmcp_16b(pdst, psrc);

   return (0);
}

static int _fastmcp_192b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_128b(pdst, psrc);
   fastmcp_64b(pdst, psrc);

   return (0);
}

static int _fastmcp_176b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_128b(pdst, psrc);
   fastmcp_32b(pdst, psrc);
   fastmcp_16b(pdst, psrc);

   return (0);
}

static int _fastmcp_160b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_128b(pdst, psrc);
   fastmcp_32b(pdst, psrc);

   return (0);
}

static int _fastmcp_144b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_128b(pdst, psrc);
   fastmcp_16b(pdst, psrc);

   return (0);
}

static int _fastmcp_128b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_128b(pdst, psrc);

   return (0);
}

static int _fastmcp_112b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_64b(pdst, psrc);
   fastmcp_32b(pdst, psrc);
   fastmcp_16b(pdst, psrc);

   return (0);
}

static int _fastmcp_96b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_64b(pdst, psrc);
   fastmcp_32b(pdst, psrc);

   return (0);
}

static int _fastmcp_80b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_64b(pdst, psrc);
   fastmcp_16b(pdst, psrc);

   return (0);
}

static int _fastmcp_64b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_64b(pdst, psrc);

   return (0);
}

static int _fastmcp_48b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_32b(pdst, psrc);
   fastmcp_16b(pdst, psrc);

   return (0);
}

static int _fastmcp_32b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_32b(pdst, psrc);

   return (0);
}

static int _fastmcp_16b(
         void * restrict _pdst, 
   const void * restrict _psrc
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16); 
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmcp_16b(pdst, psrc);

   return (0);
}

typedef int (*fastmcp_func_t)(void *, const void *);

static fastmcp_func_t _fastmcp_tab[] = {
   _fastmcp_0b,
   _fastmcp_1b,
   _fastmcp_2b,
   _fastmcp_3b,
   _fastmcp_4b,
   _fastmcp_5b,
   _fastmcp_6b,
   _fastmcp_7b,
   _fastmcp_8b,
   _fastmcp_9b,
   _fastmcp_10b,
   _fastmcp_11b,
   _fastmcp_12b,
   _fastmcp_13b,
   _fastmcp_14b,
   _fastmcp_15b,
};

static fastmcp_func_t _fastmcp_tab_16[] = {
   _fastmcp_0b,
   _fastmcp_16b,
   _fastmcp_32b,
   _fastmcp_48b,
   _fastmcp_64b,
   _fastmcp_80b,
   _fastmcp_96b,
   _fastmcp_112b,
   _fastmcp_128b,
   _fastmcp_144b,
   _fastmcp_160b,
   _fastmcp_176b,
   _fastmcp_192b,
   _fastmcp_208b,
   _fastmcp_224b,
   _fastmcp_240b,
   _fastmcp_256b,
};

static fastmcp_func_t _fastmcp_tab_256[] = {
   _fastmcp_0b,
   _fastmcp_256b,
   _fastmcp_512b,
   _fastmcp_768b,
   _fastmcp_1024b,
   _fastmcp_1280b,
   _fastmcp_1536b,
   _fastmcp_1792b,
   _fastmcp_2048b,
   _fastmcp_2304b,
   _fastmcp_2560b,
   _fastmcp_2816b,
   _fastmcp_3072b,
   _fastmcp_3328b,
   _fastmcp_3584b,
   _fastmcp_3840b,
};

/* fast aligned memory copy */
int fastmcp(
         void * restrict _pdst, 
   const void * restrict _psrc, 
         int             _nb
)
{
   int32_t * psrc = __builtin_assume_aligned(_psrc, 16);
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   /* 4K block */
   {
      int nb = (_nb >> 12);
      for (int i = 0; i < nb; i++)
      {
         _fastmcp_4K(pdst, psrc);
         pdst += (1 << 10);
         psrc += (1 << 10);
      }
      _nb -= (nb << 12);
   }

   /* 256 bytes */
   {
      int nb = (_nb >> 8);
      (_fastmcp_tab_256[nb])(pdst, psrc);
      pdst += (nb << 6);
      psrc += (nb << 6);
      _nb  -= (nb << 8);
   }

   /* 16 bytes */
   {
      int nb = (_nb >> 4);
      (_fastmcp_tab_16[nb])(pdst, psrc);
      pdst += (nb << 2);
      psrc += (nb << 2);
      _nb  -= (nb << 4);
   }

   /* rest at most 15 bytes */
   (_fastmcp_tab[_nb & 15])(pdst, psrc);

   return (_nb);
}

#define fastmzr_01b(pdst) \
{  \
   int8_t * pd8 = (int8_t *)pdst;   \
   *pd8++ = 0;                 \
}

#define fastmzr_02b(pdst) \
{  \
   int16_t * pd16 = (int16_t *)pdst;   \
   *pd16++ = 0;                        \
   \
   pdst = (int32_t *)pd16;             \
}

#define fastmzr_04b(pdst) \
{  \
   *pdst++ = 0; \
}

#define fastmzr_08b(pdst) \
{  \
   *pdst++ = 0; \
   *pdst++ = 0; \
}

#define fastmzr_16b(pdst) \
{  \
   for (int p = 0; p < 4; p++){*pdst++ = 0;}; \
}

#define fastmzr_32b(pdst) \
{  \
   for (int p = 0; p < 8; p++){*pdst++ = 0;}; \
}

#define fastmzr_64b(pdst) \
{  \
   fastmzr_32b(pdst);     \
   fastmzr_32b(pdst);     \
}

#define fastmzr_128b(pdst) \
{  \
   fastmzr_64b(pdst);     \
   fastmzr_64b(pdst);     \
}

#define fastmzr_256b(pdst) \
{  \
   fastmzr_128b(pdst);     \
   fastmzr_128b(pdst);     \
}

#define fastmzr_512b(pdst) \
{  \
   fastmzr_256b(pdst);     \
   fastmzr_256b(pdst);     \
}

#define fastmzr_1K(pdst) \
{  \
   fastmzr_512b(pdst);     \
   fastmzr_512b(pdst);     \
}

#define fastmzr_2K(pdst) \
{  \
   fastmzr_1K(pdst);     \
   fastmzr_1K(pdst);     \
}

#define fastmzr_4K(pdst) \
{  \
   fastmzr_2K(pdst);     \
   fastmzr_2K(pdst);     \
}

static int _fastmzr_1b(
         void * restrict _pdst
)
{
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_01b(pdst);

   return (1);
}

static int _fastmzr_2b(
         void * restrict _pdst
)
{
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_02b(pdst);

   return (2);
}

static int _fastmzr_3b(
         void * restrict _pdst
)
{
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_02b(pdst);
   fastmzr_01b(pdst);

   return (3);
}

static int _fastmzr_4b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_04b(pdst);

   return (4);
}

static int _fastmzr_5b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_04b(pdst);
   fastmzr_01b(pdst);

   return (5);
}

static int _fastmzr_6b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_04b(pdst);
   fastmzr_02b(pdst);

   return (6);
}

static int _fastmzr_7b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_04b(pdst);
   fastmzr_02b(pdst);
   fastmzr_01b(pdst);

   return (7);
}

static int _fastmzr_8b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_08b(pdst);

   return (8);
}

static int _fastmzr_9b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_08b(pdst);
   fastmzr_01b(pdst);

   return (9);
}


static int _fastmzr_10b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_08b(pdst);
   fastmzr_02b(pdst);

   return (10);
}

static int _fastmzr_11b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_08b(pdst);
   fastmzr_02b(pdst);
   fastmzr_01b(pdst);

   return (11);
}

static int _fastmzr_12b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_08b(pdst);
   fastmzr_04b(pdst);

   return (12);
}

static int _fastmzr_13b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_08b(pdst);
   fastmzr_04b(pdst);
   fastmzr_01b(pdst);

   return (13);
}

static int _fastmzr_14b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_08b(pdst);
   fastmzr_04b(pdst);
   fastmzr_02b(pdst);

   return (14);
}

static int _fastmzr_15b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_08b(pdst);
   fastmzr_04b(pdst);
   fastmzr_02b(pdst);
   fastmzr_01b(pdst);

   return (15);
}

static int _fastmzr_0b(
         void * restrict _pdst
)
{
   return (0);
}

BEGIN_NOINLINE static int DECLA_NOINLINE _fastmzr_4K(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_4K(pdst);

   return (0);
}

/*
static int _fastmzr_32K(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   return (0);
}

static int _fastmzr_28K(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   return (0);
}

static int _fastmzr_24K(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   return (0);
}

static int _fastmzr_20K(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   fastmzr_4K(pdst);

   return (0);
}

static int _fastmzr_16K(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   return (0);
}

static int _fastmzr_12K(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   return (0);
}

static int _fastmzr_8K(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_4K(pdst);
   fastmzr_4K(pdst);

   return (0);
}

static int _fastmzr_4K(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_4K(pdst);

   return (0);
}

*/

static int _fastmzr_3840b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_3584b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_3328b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_3072b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_2816b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_2560b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_2304b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_2048b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_1792b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_1536b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_1280b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_1024b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_768b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_512b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);
   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_256b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_256b(pdst);

   return (0);
}

static int _fastmzr_240b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_128b(pdst);
   fastmzr_64b(pdst);
   fastmzr_32b(pdst);
   fastmzr_16b(pdst);

   return (0);
}

static int _fastmzr_224b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_128b(pdst);
   fastmzr_64b(pdst);
   fastmzr_32b(pdst);

   return (0);
}

static int _fastmzr_208b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_128b(pdst);
   fastmzr_64b(pdst);
   fastmzr_16b(pdst);

   return (0);
}

static int _fastmzr_192b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_128b(pdst);
   fastmzr_64b(pdst);

   return (0);
}

static int _fastmzr_176b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_128b(pdst);
   fastmzr_32b(pdst);
   fastmzr_16b(pdst);

   return (0);
}

static int _fastmzr_160b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_128b(pdst);
   fastmzr_32b(pdst);

   return (0);
}

static int _fastmzr_144b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_128b(pdst);
   fastmzr_16b(pdst);

   return (0);
}

static int _fastmzr_128b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_128b(pdst);

   return (0);
}

static int _fastmzr_112b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_64b(pdst);
   fastmzr_32b(pdst);
   fastmzr_16b(pdst);

   return (0);
}

static int _fastmzr_96b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_64b(pdst);
   fastmzr_32b(pdst);

   return (0);
}

static int _fastmzr_80b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_64b(pdst);
   fastmzr_16b(pdst);

   return (0);
}

static int _fastmzr_64b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_64b(pdst);

   return (0);
}

static int _fastmzr_48b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_32b(pdst);
   fastmzr_16b(pdst);

   return (0);
}

static int _fastmzr_32b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_32b(pdst);

   return (0);
}

static int _fastmzr_16b(
         void * restrict _pdst
)
{
   
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   fastmzr_16b(pdst);

   return (0);
}

typedef int (*fastmzr_func_t)(void *);

static fastmzr_func_t _fastmzr_tab[] = {
   _fastmzr_0b,
   _fastmzr_1b,
   _fastmzr_2b,
   _fastmzr_3b,
   _fastmzr_4b,
   _fastmzr_5b,
   _fastmzr_6b,
   _fastmzr_7b,
   _fastmzr_8b,
   _fastmzr_9b,
   _fastmzr_10b,
   _fastmzr_11b,
   _fastmzr_12b,
   _fastmzr_13b,
   _fastmzr_14b,
   _fastmzr_15b,
};

static fastmzr_func_t _fastmzr_tab_16[] = {
   _fastmzr_0b,
   _fastmzr_16b,
   _fastmzr_32b,
   _fastmzr_48b,
   _fastmzr_64b,
   _fastmzr_80b,
   _fastmzr_96b,
   _fastmzr_112b,
   _fastmzr_128b,
   _fastmzr_144b,
   _fastmzr_160b,
   _fastmzr_176b,
   _fastmzr_192b,
   _fastmzr_208b,
   _fastmzr_224b,
   _fastmzr_240b,
   _fastmzr_256b,
};

static fastmzr_func_t _fastmzr_tab_256[] = {
   _fastmzr_0b,
   _fastmzr_256b,
   _fastmzr_512b,
   _fastmzr_768b,
   _fastmzr_1024b,
   _fastmzr_1280b,
   _fastmzr_1536b,
   _fastmzr_1792b,
   _fastmzr_2048b,
   _fastmzr_2304b,
   _fastmzr_2560b,
   _fastmzr_2816b,
   _fastmzr_3072b,
   _fastmzr_3328b,
   _fastmzr_3584b,
   _fastmzr_3840b,
};

/* fast aligned memory copy */
int fastmzr(
         void * restrict _pdst, 
         int             _nb
)
{
   int32_t * pdst = __builtin_assume_aligned(_pdst, 16);

   /* 4K block */
   {
      int nb = (_nb >> 12);
      for (int i = 0; i < nb; i++)
      {
         _fastmzr_4K(pdst);
         pdst += (1 << 10);
      }
      _nb -= (nb << 12);
   }

   /* 256 bytes */
   {
      int nb = (_nb >> 8);
      (_fastmzr_tab_256[nb])(pdst);
      pdst += (nb << 6);
      _nb  -= (nb << 8);
   }

   /* 16 bytes */
   {
      int nb = (_nb >> 4);
      (_fastmzr_tab_16[nb])(pdst);
      pdst += (nb << 2);
      _nb  -= (nb << 4);
   }

   /* rest at most 15 bytes */
   (_fastmzr_tab[_nb & 15])(pdst);

   return (_nb);
}