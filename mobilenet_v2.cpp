#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <assert.h>

#include "cnntype.h"
#include "dnnsimd.h"

extern "C" {
#include "mobilenet_v2_mdl.h"
};

// #include "openclL2.h"
// #include "kernelcl.h"
// #include "dnnlibcl.h"

#include "tictoc.h"

#ifdef __MATLAB__
#include "mex.h"
#include "mexparams.h"
#endif

#define MOBILENET_1_4   (14)
#define MOBILENET_1_0   (10)

// extern "C" void * _aligned_malloc(size_t s, size_t align)
// {
//    return aligned_alloc(align, s);
// }
// 
// extern "C" void _aligned_free(void *p)
// {
//    free(p);
// }

cnn_type_t zbuf[1024 * 1024] = {0};
cnn_type_t obuf[1024 * 1024] = {0};

class MobileNetV2
{
public:

   /* dimension parameters */
   int M, N, C;

   int M1a;
   int N1a;
   int C1a;

   int M1e;
   int N1e;
   int C1e;

   int M1p;
   int N1p;
   int C1p;

   /////
   int M2e;
   int N2e;
   int C2e;

   int M2d;
   int N2d;
   int C2d;

   int M2p;
   int N2p;
   int C2p;

   /////
   int M3e;
   int N3e;
   int C3e;

   int M3d;
   int N3d;
   int C3d;

   int M3p;
   int N3p;
   int C3p;

   /////
   int M3s;
   int N3s;
   int C3s;

   /////
   int M4e;
   int N4e;
   int C4e;

   int M4d;
   int N4d;
   int C4d;

   int M4p;
   int N4p;
   int C4p;

   /////
   int M5e;
   int N5e;
   int C5e;

   int M5d;
   int N5d;
   int C5d;

   int M5p;
   int N5p;
   int C5p;

   /////
   int M5s;
   int N5s;
   int C5s;

   /////
   int M6e;
   int N6e;
   int C6e;

   int M6d;
   int N6d;
   int C6d;

   int M6p;
   int N6p;
   int C6p;

   /////
   int M6s;
   int N6s;
   int C6s;

   /////
   int M7e;
   int N7e;
   int C7e;

   int M7d;
   int N7d;
   int C7d;

   int M7p;
   int N7p;
   int C7p;

   /////
   int M8e;
   int N8e;
   int C8e;

   int M8d;
   int N8d;
   int C8d;

   int M8p;
   int N8p;
   int C8p;

   /////
   int M8s;
   int N8s;
   int C8s;

   /////
   int M9e;
   int N9e;
   int C9e;

   int M9d;
   int N9d;
   int C9d;

   int M9p;
   int N9p;
   int C9p;

   /////
   int M9s;
   int N9s;
   int C9s;

   /////
   int M10e;
   int N10e;
   int C10e;

   int M10d;
   int N10d;
   int C10d;

   int M10p;
   int N10p;
   int C10p;

   /////
   int M10s;
   int N10s;
   int C10s;

   /////
   int M11e;
   int N11e;
   int C11e;

   int M11d;
   int N11d;
   int C11d;

   int M11p;
   int N11p;
   int C11p;

   /////
   int M12e;
   int N12e;
   int C12e;

   int M12d;
   int N12d;
   int C12d;

   int M12p;
   int N12p;
   int C12p;

   /////
   int M12s;
   int N12s;
   int C12s;

   /////
   int M13e;
   int N13e;
   int C13e;

   int M13d;
   int N13d;
   int C13d;

   int M13p;
   int N13p;
   int C13p;

   /////
   int M13s;
   int N13s;
   int C13s;

   /////
   int M14e;
   int N14e;
   int C14e;

   int M14d;
   int N14d;
   int C14d;

   int M14p;
   int N14p;
   int C14p;

   /////
   int M15e;
   int N15e;
   int C15e;

   int M15d;
   int N15d;
   int C15d;

   int M15p;
   int N15p;
   int C15p;

   /////
   int M15s;
   int N15s;
   int C15s;

   /////
   int M16e;
   int N16e;
   int C16e;

   int M16d;
   int N16d;
   int C16d;

   int M16p;
   int N16p;
   int C16p;

   /////
   int M16s;
   int N16s;
   int C16s;

   /////
   int M17e;
   int N17e;
   int C17e;

   int M17d;
   int N17d;
   int C17d;

   int M17p;
   int N17p;
   int C17p;

   /////
   int M18a;
   int N18a;
   int C18a;

   cnn_type_t * Y1a;
   cnn_type_t * Y1e;
   cnn_type_t * Y1p;

   cnn_type_t * Y2e;
   cnn_type_t * Y2d;
   cnn_type_t * Y2p;

   cnn_type_t * Y3e;
   cnn_type_t * Y3d;
   cnn_type_t * Y3p;
   cnn_type_t * Y3s;

   cnn_type_t * Y4e;
   cnn_type_t * Y4d;
   cnn_type_t * Y4p;

   cnn_type_t * Y5e;
   cnn_type_t * Y5d;
   cnn_type_t * Y5p;
   cnn_type_t * Y5s;

   cnn_type_t * Y6e;
   cnn_type_t * Y6d;
   cnn_type_t * Y6p;
   cnn_type_t * Y6s;

   cnn_type_t * Y7e;
   cnn_type_t * Y7d;
   cnn_type_t * Y7p;

   cnn_type_t * Y8e;
   cnn_type_t * Y8d;
   cnn_type_t * Y8p;
   cnn_type_t * Y8s;

   cnn_type_t * Y9e;
   cnn_type_t * Y9d;
   cnn_type_t * Y9p;
   cnn_type_t * Y9s;

   cnn_type_t * Y10e;
   cnn_type_t * Y10d;
   cnn_type_t * Y10p;
   cnn_type_t * Y10s;

   cnn_type_t * Y11e;
   cnn_type_t * Y11d;
   cnn_type_t * Y11p;

   cnn_type_t * Y12e;
   cnn_type_t * Y12d;
   cnn_type_t * Y12p;
   cnn_type_t * Y12s;

   cnn_type_t * Y13e;
   cnn_type_t * Y13d;
   cnn_type_t * Y13p;
   cnn_type_t * Y13s;

   cnn_type_t * Y14e;
   cnn_type_t * Y14d;
   cnn_type_t * Y14p;

   cnn_type_t * Y15e;
   cnn_type_t * Y15d;
   cnn_type_t * Y15p;
   cnn_type_t * Y15s;

   cnn_type_t * Y16e;
   cnn_type_t * Y16d;
   cnn_type_t * Y16p;
   cnn_type_t * Y16s;

   cnn_type_t * Y17e;
   cnn_type_t * Y17d;
   cnn_type_t * Y17p;

   cnn_type_t * Y18a;

   cnn_type_t * Y19a;

public:
   int Initialize(int M, int N, int C, int mode);
   cnn_type_t * Run(const cnn_type_t * pX);
   int Deinitialize();

   void range(cnn_type_t * pX, int M, int N, int C);
};

int MobileNetV2::Initialize(int M0, int N0, int C0, int depth_mode)
{
   /* dimension parameters */
   M   = M0;
   N   = N0;
   C   = C0;

   M1a = M / 2;
   N1a = N / 2;
   C1a = (depth_mode == MOBILENET_1_4) ? ( 48) : ( 32);

   M1e = M1a;
   N1e = N1a;
   C1e = C1a;

   M1p = M1e;
   N1p = N1e;
   C1p = (depth_mode == MOBILENET_1_4) ? ( 24) : ( 16);

   /////
   M2e = M1p;
   N2e = N1p;
   C2e = (depth_mode == MOBILENET_1_4) ? (144) : ( 96);

   M2d = M2e / 2;
   N2d = N2e / 2;
   C2d = C2e;

   M2p = M2d;
   N2p = N2d;
   C2p = (depth_mode == MOBILENET_1_4) ? ( 32) : ( 24);

   /////
   M3e = M2p;
   N3e = N2p;
   C3e = (depth_mode == MOBILENET_1_4) ? (192) : (144);

   M3d = M3e;
   N3d = N3e;
   C3d = C3e;

   M3p = M3d;
   N3p = N3d;
   C3p = (depth_mode == MOBILENET_1_4) ? ( 32) : ( 24);

   /////
   M3s = M3p;
   N3s = N3p;
   C3s = C3p;

   /////
   M4e = M3p;
   N4e = N3p;
   C4e = (depth_mode == MOBILENET_1_4) ? (192) : (144);

   M4d = M4e / 2;
   N4d = N4e / 2;
   C4d = C4e;

   M4p = M4d;
   N4p = N4d;
   C4p = (depth_mode == MOBILENET_1_4) ? ( 48) : ( 32);

   /////
   M5e = M4p;
   N5e = N4p;
   C5e = (depth_mode == MOBILENET_1_4) ? (288) : (192);

   M5d = M5e;
   N5d = N5e;
   C5d = C5e;

   M5p = M5d;
   N5p = N5d;
   C5p = (depth_mode == MOBILENET_1_4) ? ( 48) : ( 32);

   /////
   M5s = M5p;
   N5s = N5p;
   C5s = C5p;

   /////
   M6e = M5p;
   N6e = N5p;
   C6e = (depth_mode == MOBILENET_1_4) ? (288) : (192);

   M6d = M6e;
   N6d = N6e;
   C6d = C6e;

   M6p = M6d;
   N6p = N6d;
   C6p = (depth_mode == MOBILENET_1_4) ? ( 48) : ( 32);

   /////
   M6s = M6p;
   N6s = N6p;
   C6s = C6p;

   /////
   M7e = M6s;
   N7e = N6s;
   C7e = (depth_mode == MOBILENET_1_4) ? (288) : (192);

   M7d = M7e / 2;
   N7d = N7e / 2;
   C7d = C7e;

   M7p = M7d;
   N7p = N7d;
   C7p = (depth_mode == MOBILENET_1_4) ? ( 88) : ( 64);

   /////
   M8e = M7p;
   N8e = N7p;
   C8e = (depth_mode == MOBILENET_1_4) ? (528) : (384);

   M8d = M8e;
   N8d = N8e;
   C8d = C8e;

   M8p = M8d;
   N8p = N8d;
   C8p = (depth_mode == MOBILENET_1_4) ? ( 88) : ( 64);

   /////
   M8s = M8p;
   N8s = N8p;
   C8s = C8p;

   /////
   M9e = M8p;
   N9e = N8p;
   C9e = (depth_mode == MOBILENET_1_4) ? (528) : (384);

   M9d = M9e;
   N9d = N9e;
   C9d = C9e;

   M9p = M9d;
   N9p = N9d;
   C9p = (depth_mode == MOBILENET_1_4) ? ( 88) : ( 64);

   /////
   M9s = M9p;
   N9s = N9p;
   C9s = C9p;

   /////
   M10e = M9p;
   N10e = N9p;
   C10e = (depth_mode == MOBILENET_1_4) ? (528) : (384);

   M10d = M10e;
   N10d = N10e;
   C10d = C10e;

   M10p = M10d;
   N10p = N10d;
   C10p = (depth_mode == MOBILENET_1_4) ? ( 88) : ( 64);

   /////
   M10s = M10p;
   N10s = N10p;
   C10s = C10p;

   /////
   M11e = M10p;
   N11e = N10p;
   C11e = (depth_mode == MOBILENET_1_4) ? ( 528) : ( 384);

   M11d = M11e;
   N11d = N11e;
   C11d = C11e;

   M11p = M11d;
   N11p = N11d;
   C11p = (depth_mode == MOBILENET_1_4) ? ( 136) : (  96);

   /////
   M12e = M11p;
   N12e = N11p;
   C12e = (depth_mode == MOBILENET_1_4) ? ( 816) : ( 576);

   M12d = M12e;
   N12d = N12e;
   C12d = C12e;

   M12p = M12d;
   N12p = N12d;
   C12p = (depth_mode == MOBILENET_1_4) ? ( 136) : (  96);

   /////
   M12s = M12p;
   N12s = N12p;
   C12s = C12p;

   /////
   M13e = M12p;
   N13e = N12p;
   C13e = (depth_mode == MOBILENET_1_4) ? ( 816) : ( 576);

   M13d = M12e;
   N13d = N12e;
   C13d = C12e;

   M13p = M13d;
   N13p = N13d;
   C13p = (depth_mode == MOBILENET_1_4) ? ( 136) : (  96);

   /////
   M13s = M13p;
   N13s = N13p;
   C13s = C13p;

   /////
   M14e = M13p;
   N14e = N13p;
   C14e = (depth_mode == MOBILENET_1_4) ? ( 816) : ( 576);

   M14d = M14e / 2;
   N14d = N14e / 2;
   C14d = C14e;

   M14p = M14d;
   N14p = N14d;
   C14p = (depth_mode == MOBILENET_1_4) ? ( 224) : ( 160);

   /////
   M15e = M14p;
   N15e = N14p;
   C15e = (depth_mode == MOBILENET_1_4) ? (1344) : ( 960);

   M15d = M15e;
   N15d = N15e;
   C15d = C15e;

   M15p = M15d;
   N15p = N15d;
   C15p = (depth_mode == MOBILENET_1_4) ? ( 224) : ( 160);

   /////
   M15s = M15p;
   N15s = N15p;
   C15s = C15p;

   /////
   M16e = M15p;
   N16e = N15p;
   C16e = (depth_mode == MOBILENET_1_4) ? (1344) : ( 960);

   M16d = M16e;
   N16d = N16e;
   C16d = C16e;

   M16p = M16d;
   N16p = N16d;
   C16p = (depth_mode == MOBILENET_1_4) ? ( 224) : ( 160);

   /////
   M16s = M16p;
   N16s = N16p;
   C16s = C16p;

   /////
   M17e = M16p;
   N17e = N16p;
   C17e = (depth_mode == MOBILENET_1_4) ? (1344) : ( 960);

   M17d = M17e;
   N17d = N17e;
   C17d = C17e;

   M17p = M17d;
   N17p = N17d;
   C17p = (depth_mode == MOBILENET_1_4) ? ( 448) : ( 320);

   /////
   M18a = M17p;
   N18a = N17p;
   C18a = (depth_mode == MOBILENET_1_4) ? (1792) : (1280);

   Y1a = (cnn_type_t *)_aligned_malloc(M1a * N1a * C1a  * sizeof(cnn_type_t), 16);
   Y1e = (cnn_type_t *)_aligned_malloc(M1e * N1e * C1e  * sizeof(cnn_type_t), 16);
   Y1p = (cnn_type_t *)_aligned_malloc(M1p * N1p * C1p  * sizeof(cnn_type_t), 16);

   Y2e = (cnn_type_t *)_aligned_malloc(M2e * N2e * C2e  * sizeof(cnn_type_t), 16);
   Y2d = (cnn_type_t *)_aligned_malloc(M2d * N2d * C2d  * sizeof(cnn_type_t), 16);
   Y2p = (cnn_type_t *)_aligned_malloc(M2p * N2p * C2p  * sizeof(cnn_type_t), 16);

   Y3e = (cnn_type_t *)_aligned_malloc(M3e * N3e * C3e  * sizeof(cnn_type_t), 16);
   Y3d = (cnn_type_t *)_aligned_malloc(M3d * N3d * C3d  * sizeof(cnn_type_t), 16);
   Y3p = (cnn_type_t *)_aligned_malloc(M3p * N3p * C3p  * sizeof(cnn_type_t), 16);
   Y3s = (cnn_type_t *)_aligned_malloc(M3s * N3s * C3s  * sizeof(cnn_type_t), 16);

   Y4e = (cnn_type_t *)_aligned_malloc(M4e * N4e * C4e  * sizeof(cnn_type_t), 16);
   Y4d = (cnn_type_t *)_aligned_malloc(M4d * N4d * C4d  * sizeof(cnn_type_t), 16);
   Y4p = (cnn_type_t *)_aligned_malloc(M4p * N4p * C4p  * sizeof(cnn_type_t), 16);

   Y5e = (cnn_type_t *)_aligned_malloc(M5e * N5e * C5e  * sizeof(cnn_type_t), 16);
   Y5d = (cnn_type_t *)_aligned_malloc(M5d * N5d * C5d  * sizeof(cnn_type_t), 16);
   Y5p = (cnn_type_t *)_aligned_malloc(M5p * N5p * C5p  * sizeof(cnn_type_t), 16);
   Y5s = (cnn_type_t *)_aligned_malloc(M5s * N5s * C5s  * sizeof(cnn_type_t), 16);

   Y6e = (cnn_type_t *)_aligned_malloc(M6e * N6e * C6e  * sizeof(cnn_type_t), 16);
   Y6d = (cnn_type_t *)_aligned_malloc(M6d * N6d * C6d  * sizeof(cnn_type_t), 16);
   Y6p = (cnn_type_t *)_aligned_malloc(M6p * N6p * C6p  * sizeof(cnn_type_t), 16);
   Y6s = (cnn_type_t *)_aligned_malloc(M6s * N6s * C6s  * sizeof(cnn_type_t), 16);

   Y7e = (cnn_type_t *)_aligned_malloc(M7e * N7e * C7e  * sizeof(cnn_type_t), 16);
   Y7d = (cnn_type_t *)_aligned_malloc(M7d * N7d * C7d  * sizeof(cnn_type_t), 16);
   Y7p = (cnn_type_t *)_aligned_malloc(M7p * N7p * C7p  * sizeof(cnn_type_t), 16);

   Y8e = (cnn_type_t *)_aligned_malloc(M8e * N8e * C8e  * sizeof(cnn_type_t), 16);
   Y8d = (cnn_type_t *)_aligned_malloc(M8d * N8d * C8d  * sizeof(cnn_type_t), 16);
   Y8p = (cnn_type_t *)_aligned_malloc(M8p * N8p * C8p  * sizeof(cnn_type_t), 16);
   Y8s = (cnn_type_t *)_aligned_malloc(M8s * N8s * C8s  * sizeof(cnn_type_t), 16);

   Y9e = (cnn_type_t *)_aligned_malloc(M9e * N9e * C9e  * sizeof(cnn_type_t), 16);
   Y9d = (cnn_type_t *)_aligned_malloc(M9d * N9d * C9d  * sizeof(cnn_type_t), 16);
   Y9p = (cnn_type_t *)_aligned_malloc(M9p * N9p * C9p  * sizeof(cnn_type_t), 16);
   Y9s = (cnn_type_t *)_aligned_malloc(M9s * N9s * C9s  * sizeof(cnn_type_t), 16);

   Y10e = (cnn_type_t *)_aligned_malloc(M10e * N10e * C10e * sizeof(cnn_type_t), 16);
   Y10d = (cnn_type_t *)_aligned_malloc(M10d * N10d * C10d * sizeof(cnn_type_t), 16);
   Y10p = (cnn_type_t *)_aligned_malloc(M10p * N10p * C10p * sizeof(cnn_type_t), 16);
   Y10s = (cnn_type_t *)_aligned_malloc(M10s * N10s * C10s * sizeof(cnn_type_t), 16);

   Y11e = (cnn_type_t *)_aligned_malloc(M11e * N11e * C11e * sizeof(cnn_type_t), 16);
   Y11d = (cnn_type_t *)_aligned_malloc(M11d * N11d * C11d * sizeof(cnn_type_t), 16);
   Y11p = (cnn_type_t *)_aligned_malloc(M11p * N11p * C11p * sizeof(cnn_type_t), 16);

   Y12e = (cnn_type_t *)_aligned_malloc(M12e * N12e * C12e * sizeof(cnn_type_t), 16);
   Y12d = (cnn_type_t *)_aligned_malloc(M12d * N12d * C12d * sizeof(cnn_type_t), 16);
   Y12p = (cnn_type_t *)_aligned_malloc(M12p * N12p * C12p * sizeof(cnn_type_t), 16);
   Y12s = (cnn_type_t *)_aligned_malloc(M12s * N12s * C12s * sizeof(cnn_type_t), 16);

   Y13e = (cnn_type_t *)_aligned_malloc(M13e * N13e * C13e * sizeof(cnn_type_t), 16);
   Y13d = (cnn_type_t *)_aligned_malloc(M13d * N13d * C13d * sizeof(cnn_type_t), 16);
   Y13p = (cnn_type_t *)_aligned_malloc(M13p * N13p * C13p * sizeof(cnn_type_t), 16);
   Y13s = (cnn_type_t *)_aligned_malloc(M13s * N13s * C13s * sizeof(cnn_type_t), 16);

   Y14e = (cnn_type_t *)_aligned_malloc(M14e * N14e * C14e * sizeof(cnn_type_t), 16);
   Y14d = (cnn_type_t *)_aligned_malloc(M14d * N14d * C14d * sizeof(cnn_type_t), 16);
   Y14p = (cnn_type_t *)_aligned_malloc(M14p * N14p * C14p * sizeof(cnn_type_t), 16);

   Y15e = (cnn_type_t *)_aligned_malloc(M15e * N15e * C15e * sizeof(cnn_type_t), 16);
   Y15d = (cnn_type_t *)_aligned_malloc(M15d * N15d * C15d * sizeof(cnn_type_t), 16);
   Y15p = (cnn_type_t *)_aligned_malloc(M15p * N15p * C15p * sizeof(cnn_type_t), 16);
   Y15s = (cnn_type_t *)_aligned_malloc(M15s * N15s * C15s * sizeof(cnn_type_t), 16);

   Y16e = (cnn_type_t *)_aligned_malloc(M16e * N16e * C16e * sizeof(cnn_type_t), 16);
   Y16d = (cnn_type_t *)_aligned_malloc(M16d * N16d * C16d * sizeof(cnn_type_t), 16);
   Y16p = (cnn_type_t *)_aligned_malloc(M16p * N16p * C16p * sizeof(cnn_type_t), 16);
   Y16s = (cnn_type_t *)_aligned_malloc(M16s * N16s * C16s * sizeof(cnn_type_t), 16);

   Y17e = (cnn_type_t *)_aligned_malloc(M17e * N17e * C17e * sizeof(cnn_type_t), 16);
   Y17d = (cnn_type_t *)_aligned_malloc(M17d * N17d * C17d * sizeof(cnn_type_t), 16);
   Y17p = (cnn_type_t *)_aligned_malloc(M17p * N17p * C17p * sizeof(cnn_type_t), 16);

   Y18a = (cnn_type_t *)_aligned_malloc(M18a * N18a * C18a * sizeof(cnn_type_t), 16);

   Y19a = (cnn_type_t *)_aligned_malloc(   1 *    1 * C18a * sizeof(cnn_type_t), 16);

   return (0);
}

int MobileNetV2::Deinitialize()
{
   _aligned_free(Y1a);
   _aligned_free(Y1e);
   _aligned_free(Y1p);

   _aligned_free(Y2e);
   _aligned_free(Y2d);
   _aligned_free(Y2p);

   _aligned_free(Y3e);
   _aligned_free(Y3d);
   _aligned_free(Y3p);
   _aligned_free(Y3s);

   _aligned_free(Y4e);
   _aligned_free(Y4d);
   _aligned_free(Y4p);

   _aligned_free(Y5e);
   _aligned_free(Y5d);
   _aligned_free(Y5p);
   _aligned_free(Y5s);

   _aligned_free(Y6e);
   _aligned_free(Y6d);
   _aligned_free(Y6p);
   _aligned_free(Y6s);

   _aligned_free(Y7e);
   _aligned_free(Y7d);
   _aligned_free(Y7p);

   _aligned_free(Y8e);
   _aligned_free(Y8d);
   _aligned_free(Y8p);
   _aligned_free(Y8s);

   _aligned_free(Y9e);
   _aligned_free(Y9d);
   _aligned_free(Y9p);
   _aligned_free(Y9s);

   _aligned_free(Y10e);
   _aligned_free(Y10d);
   _aligned_free(Y10p);
   _aligned_free(Y10s);

   _aligned_free(Y11e);
   _aligned_free(Y11d);
   _aligned_free(Y11p);

   _aligned_free(Y12e);
   _aligned_free(Y12d);
   _aligned_free(Y12p);
   _aligned_free(Y12s);

   _aligned_free(Y13e);
   _aligned_free(Y13d);
   _aligned_free(Y13p);
   _aligned_free(Y13s);

   _aligned_free(Y14e);
   _aligned_free(Y14d);
   _aligned_free(Y14p);

   _aligned_free(Y15e);
   _aligned_free(Y15d);
   _aligned_free(Y15p);
   _aligned_free(Y15s);

   _aligned_free(Y16e);
   _aligned_free(Y16d);
   _aligned_free(Y16p);
   _aligned_free(Y16s);

   _aligned_free(Y17e);
   _aligned_free(Y17d);
   _aligned_free(Y17p);

   _aligned_free(Y18a);

   _aligned_free(Y19a);

   return (0);
}

void mobilenet_avg7x7(
   int  C,
   cnn_type_t * __px,
   cnn_type_t * __py
   )
{
   cnn_type_t * _px = __px; // builtin_assume_aligned(__px, 16);
   cnn_type_t * _py = __py; // builtin_assume_aligned(__py, 16);

   /* S - 1 */   
   addpool(C, _px + C *  0, _px + C *  48, (cnn_type_t *)(_px + C *  0));
   addpool(C, _px + C *  1, _px + C *  47, (cnn_type_t *)(_px + C *  1));
   addpool(C, _px + C *  2, _px + C *  46, (cnn_type_t *)(_px + C *  2));
   addpool(C, _px + C *  3, _px + C *  45, (cnn_type_t *)(_px + C *  3));
   addpool(C, _px + C *  4, _px + C *  44, (cnn_type_t *)(_px + C *  4));
   addpool(C, _px + C *  5, _px + C *  43, (cnn_type_t *)(_px + C *  5));
   addpool(C, _px + C *  6, _px + C *  42, (cnn_type_t *)(_px + C *  6));
   addpool(C, _px + C *  7, _px + C *  41, (cnn_type_t *)(_px + C *  7));
   addpool(C, _px + C *  8, _px + C *  40, (cnn_type_t *)(_px + C *  8));
   addpool(C, _px + C *  9, _px + C *  39, (cnn_type_t *)(_px + C *  9));
   addpool(C, _px + C * 10, _px + C *  38, (cnn_type_t *)(_px + C * 10));
   addpool(C, _px + C * 11, _px + C *  37, (cnn_type_t *)(_px + C * 11));
   addpool(C, _px + C * 12, _px + C *  36, (cnn_type_t *)(_px + C * 12));
   addpool(C, _px + C * 13, _px + C *  35, (cnn_type_t *)(_px + C * 13));
   addpool(C, _px + C * 14, _px + C *  34, (cnn_type_t *)(_px + C * 14));
   addpool(C, _px + C * 15, _px + C *  33, (cnn_type_t *)(_px + C * 15));
   addpool(C, _px + C * 16, _px + C *  32, (cnn_type_t *)(_px + C * 16));
   addpool(C, _px + C * 17, _px + C *  31, (cnn_type_t *)(_px + C * 17));
   addpool(C, _px + C * 18, _px + C *  30, (cnn_type_t *)(_px + C * 18));
   addpool(C, _px + C * 19, _px + C *  29, (cnn_type_t *)(_px + C * 19));
   addpool(C, _px + C * 20, _px + C *  28, (cnn_type_t *)(_px + C * 20));
   addpool(C, _px + C * 21, _px + C *  27, (cnn_type_t *)(_px + C * 21));
   addpool(C, _px + C * 22, _px + C *  26, (cnn_type_t *)(_px + C * 22));
   addpool(C, _px + C * 23, _px + C *  25, (cnn_type_t *)(_px + C * 23));

   /* S - 2 */   
   addpool(C, _px + C *  0, _px + C *  24, (cnn_type_t *)(_px + C *  0));
   addpool(C, _px + C *  1, _px + C *  23, (cnn_type_t *)(_px + C *  1));
   addpool(C, _px + C *  2, _px + C *  22, (cnn_type_t *)(_px + C *  2));
   addpool(C, _px + C *  3, _px + C *  21, (cnn_type_t *)(_px + C *  3));
   addpool(C, _px + C *  4, _px + C *  20, (cnn_type_t *)(_px + C *  4));
   addpool(C, _px + C *  5, _px + C *  19, (cnn_type_t *)(_px + C *  5));
   addpool(C, _px + C *  6, _px + C *  18, (cnn_type_t *)(_px + C *  6));
   addpool(C, _px + C *  7, _px + C *  17, (cnn_type_t *)(_px + C *  7));
   addpool(C, _px + C *  8, _px + C *  16, (cnn_type_t *)(_px + C *  8));
   addpool(C, _px + C *  9, _px + C *  15, (cnn_type_t *)(_px + C *  9));
   addpool(C, _px + C * 10, _px + C *  14, (cnn_type_t *)(_px + C * 10));
   addpool(C, _px + C * 11, _px + C *  13, (cnn_type_t *)(_px + C * 11));

   /* S - 3 */   
   addpool(C, _px + C *  0, _px + C *  12, (cnn_type_t *)(_px + C *  0));
   addpool(C, _px + C *  1, _px + C *  11, (cnn_type_t *)(_px + C *  1));
   addpool(C, _px + C *  2, _px + C *  10, (cnn_type_t *)(_px + C *  2));
   addpool(C, _px + C *  3, _px + C *   9, (cnn_type_t *)(_px + C *  3));
   addpool(C, _px + C *  4, _px + C *   8, (cnn_type_t *)(_px + C *  4));
   addpool(C, _px + C *  5, _px + C *   7, (cnn_type_t *)(_px + C *  5));

   /* S - 4 */   
   addpool(C, _px + C *  0, _px + C *   6, (cnn_type_t *)(_px + C *  0));
   addpool(C, _px + C *  1, _px + C *   5, (cnn_type_t *)(_px + C *  1));
   addpool(C, _px + C *  2, _px + C *   4, (cnn_type_t *)(_px + C *  2));

   /* S - 5 */   
   addpool(C, _px + C *  0, _px + C *   3, (cnn_type_t *)(_px + C *  0));
   addpool(C, _px + C *  1, _px + C *   2, (cnn_type_t *)(_px + C *  1));

   /* S - 6 */   
   addpool(C, _px + C *  0, _px + C *   1, (cnn_type_t *)(_py + C *  0));
}

void MobileNetV2::range(
   cnn_type_t * prX,
   int M, int N, int C
   )
{
   cnn_type_t fmin = +999.0;
   cnn_type_t fmax = -999.0;

   for (int i = 0; i < M * N * C; i++)
   {
      cnn_type_t f = *prX++;
      fmin = (f < fmin) ? (f) : (fmin);
      fmax = (f > fmax) ? (f) : (fmax);
   }

   // printf("(fmin, fmax) = (%9.2f, %9.2f).\n", fmin, fmax);
}

cnn_type_t * MobileNetV2::Run(
   const cnn_type_t *  prX
   )
{
   /* ********************************************************************** *
   *  conv2d_3x3, stride = 2, same                                           *
   * ********************************************************************** */
   conv3x3_p1s2_bnrelu_inp(
      M, N, 
      4, C1a,

          prX, 
      
          w1a,
        mu_1a,
       var_1a,
      beta_1a,
          6.0,

          Y1a
      );


   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M1a, N1a, 
      C1a, C1e,

          Y1a, 
      
          w1e, 
        mu_1e,
       var_1e,
      beta_1e,
          6.0,

          Y1e
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M1e, N1e, 
      C1e, C1p,

          Y1e, 
      
          w1p, 
        mu_1p,
       var_1p,
      beta_1p,

          Y1p
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-2                                                                                                         //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M1p, N1p, 
      C1p, C2e,

          Y1p, 
      
          w2e, 
        mu_2e,
       var_2e,
      beta_2e,
          6.0,

          Y2e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 2, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s2_bnrelu(
      M2e, N2e, 
      C2e, C2d,

          Y2e, 
      
          w2d, 
        mu_2d,
       var_2d,
      beta_2d,
          6.0,

          Y2d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M2d, N2d, 
      C2d, C2p,

          Y2d, 
      
          w2p, 
        mu_2p,
       var_2p,
      beta_2p,

          Y2p
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-3                                                                                                         //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M2p, N2p, 
      C2p, C3e,

          Y2p, 
      
          w3e, 
        mu_3e,
       var_3e,
      beta_3e,
          6.0,

          Y3e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 2, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M3e, N3e, 
      C3e, C3d,

          Y3e, 
      
          w3d, 
        mu_3d,
       var_3d,
      beta_3d,
          6.0,

          Y3d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M3d, N3d, 
      C3d, C3p,

          Y3d, 
      
          w3p, 
        mu_3p,
       var_3p,
      beta_3p,

          Y3p
      );

   /* ********************************************************************** *
   *  Y2p + Y3p                                                              *
   * ********************************************************************** */
   addpool(
      M3s * N3s * C3s,
      Y2p , Y3p , Y3s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-4                                                                                                         //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M3p, N3p, 
      C3p, C4e,

          Y3s, 
      
          w4e, 
        mu_4e,
       var_4e,
      beta_4e,
          6.0,

          Y4e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 2, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s2_bnrelu(
      M4e, N4e, 
      C4e, C4d,

          Y4e, 
      
          w4d, 
        mu_4d,
       var_4d,
      beta_4d,
          6.0,

          Y4d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M4d, N4d, 
      C4d, C4p,

          Y4d, 
      
          w4p, 
        mu_4p,
       var_4p,
      beta_4p,

          Y4p
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-5                                                                                                         //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M4p, N4p, 
      C4p, C5e,

          Y4p, 
      
          w5e, 
        mu_5e,
       var_5e,
      beta_5e,
          6.0,

          Y5e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 2, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M5e, N5e, 
      C5e, C5d,

          Y5e, 
      
          w5d, 
        mu_5d,
       var_5d,
      beta_5d,
          6.0,

          Y5d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M5d, N5d, 
      C5d, C5p,

          Y5d, 
      
          w5p, 
        mu_5p,
       var_5p,
      beta_5p,

          Y5p
      );

   /* ********************************************************************** *
   *  Y4p + Y5p                                                              *
   * ********************************************************************** */
   addpool(
      M5s * N5s * C5s,
      Y4p , Y5p , Y5s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-6                                                                                                         //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M5p, N5p, 
      C5p, C6e,

          Y5s, 
      
          w6e, 
        mu_6e,
       var_6e,
      beta_6e,
          6.0,

          Y6e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 2, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M6e, N6e, 
      C6e, C6d,

          Y6e, 
      
          w6d, 
        mu_6d,
       var_6d,
      beta_6d,
          6.0,

          Y6d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M6d, N6d, 
      C6d, C6p,

          Y6d, 
      
          w6p, 
        mu_6p,
       var_6p,
      beta_6p,

          Y6p
      );

   /* ********************************************************************** *
   *  Y5s + Y6p                                                              *
   * ********************************************************************** */
   addpool(
      M6s * N6s * C6s,
      Y5s , Y6p , Y6s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-7                                                                                                         //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M6p, N6p, 
      C6p, C7e,

          Y6s, 
      
          w7e, 
        mu_7e,
       var_7e,
      beta_7e,
          6.0,

          Y7e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 2, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s2_bnrelu(
      M7e, N7e, 
      C7e, C7d,

          Y7e, 
      
          w7d, 
        mu_7d,
       var_7d,
      beta_7d,
          6.0,

          Y7d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M7d, N7d, 
      C7d, C7p,

          Y7d, 
      
          w7p, 
        mu_7p,
       var_7p,
      beta_7p,

          Y7p
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-8                                                                                                         //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M7p, N7p, 
      C7p, C8e,

          Y7p, 
      
          w8e, 
        mu_8e,
       var_8e,
      beta_8e,
          6.0,

          Y8e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 2, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M8e, N8e, 
      C8e, C8d,

          Y8e, 
      
          w8d, 
        mu_8d,
       var_8d,
      beta_8d,
          6.0,

          Y8d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M8d, N8d, 
      C8d, C8p,

          Y8d, 
      
          w8p, 
        mu_8p,
       var_8p,
      beta_8p,

          Y8p
      );

   /* ********************************************************************** *
   *  Y7p + Y8p                                                              *
   * ********************************************************************** */
   addpool(
      M8s * N8s * C8s,
      Y7p , Y8p , Y8s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-9                                                                                                         //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M8p, N8p, 
      C8p, C9e,

          Y8s, 
      
          w9e, 
        mu_9e,
       var_9e,
      beta_9e,
          6.0,

          Y9e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M9e, N9e, 
      C9e, C9d,

          Y9e, 
      
          w9d, 
        mu_9d,
       var_9d,
      beta_9d,
          6.0,

          Y9d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M9d, N9d, 
      C9d, C9p,

          Y9d, 
      
          w9p, 
        mu_9p,
       var_9p,
      beta_9p,

          Y9p
      );

   /* ********************************************************************** *
   *  Y8s + Y9p                                                              *
   * ********************************************************************** */
   addpool(
      M9s * N9s * C9s,
      Y8s , Y9p , Y9s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-10                                                                                                        //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M9p, N9p, 
      C9p, C10e,

          Y9s, 
      
          w10e, 
        mu_10e,
       var_10e,
      beta_10e,
           6.0,

          Y10e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M10e, N10e, 
      C10e, C10d,

          Y10e, 
      
          w10d, 
        mu_10d,
       var_10d,
      beta_10d,
           6.0,

          Y10d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M10d, N10d, 
      C10d, C10p,

          Y10d, 
      
          w10p, 
        mu_10p,
       var_10p,
      beta_10p,

          Y10p
      );

   /* ********************************************************************** *
   *  Y8s + Y9p                                                              *
   * ********************************************************************** */
   addpool(
      M10s * N10s * C10s,
      Y9s  , Y10p , Y10s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-11                                                                                                        //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M10p, N10p, 
      C10p, C11e,

          Y10s, 
      
          w11e, 
        mu_11e,
       var_11e,
      beta_11e,
           6.0,

          Y11e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M11e, N11e, 
      C11e, C11d,

          Y11e, 
      
          w11d, 
        mu_11d,
       var_11d,
      beta_11d,
           6.0,

          Y11d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M11d, N11d, 
      C11d, C11p,

          Y11d, 
      
          w11p, 
        mu_11p,
       var_11p,
      beta_11p,

          Y11p
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-12                                                                                                        //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M11p, N11p, 
      C11p, C12e,

          Y11p, 
      
          w12e, 
        mu_12e,
       var_12e,
      beta_12e,
          6.0,

          Y12e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M12e, N12e, 
      C12e, C12d,

          Y12e, 
      
          w12d, 
        mu_12d,
       var_12d,
      beta_12d,
           6.0,

          Y12d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M12d, N12d, 
      C12d, C12p,

          Y12d, 
      
          w12p, 
        mu_12p,
       var_12p,
      beta_12p,

          Y12p
      );

   /* ********************************************************************** *
   *  Y11p + Y12p                                                            *
   * ********************************************************************** */
   addpool(
      M12s * N12s * C12s,
      Y11p , Y12p , Y12s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-13                                                                                                        //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M12p, N12p, 
      C12p, C13e,

          Y12s, 
      
          w13e, 
        mu_13e,
       var_13e,
      beta_13e,
          6.0,

          Y13e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M13e, N13e, 
      C13e, C13d,

          Y13e, 
      
          w13d, 
        mu_13d,
       var_13d,
      beta_13d,
           6.0,

          Y13d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M13d, N13d, 
      C13d, C13p,

          Y13d, 
      
          w13p, 
        mu_13p,
       var_13p,
      beta_13p,

          Y13p
      );

   /* ********************************************************************** *
   *  Y12s + Y13p                                                            *
   * ********************************************************************** */
   addpool(
      M13s * N13s * C13s,
      Y12s , Y13p , Y13s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-14                                                                                                        //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M13s, N13s, 
      C13s, C14e,

          Y13s, 
      
          w14e, 
        mu_14e,
       var_14e,
      beta_14e,
           6.0,

          Y14e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s2_bnrelu(
      M14e, N14e, 
      C14e, C14d,

          Y14e, 
      
          w14d, 
        mu_14d,
       var_14d,
      beta_14d,
           6.0,

          Y14d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M14d, N14d, 
      C14d, C14p,

          Y14d, 
      
          w14p, 
        mu_14p,
       var_14p,
      beta_14p,

          Y14p
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-15                                                                                                        //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M14p, N14p, 
      C14p, C15e,

          Y14p, 
      
          w15e, 
        mu_15e,
       var_15e,
      beta_15e,
          6.0,

          Y15e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M15e, N15e, 
      C15e, C15d,

          Y15e, 
      
          w15d, 
        mu_15d,
       var_15d,
      beta_15d,
           6.0,

          Y15d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M15d, N15d, 
      C15d, C15p,

          Y15d, 
      
          w15p, 
        mu_15p,
       var_15p,
      beta_15p,

          Y15p
      );

   /* ********************************************************************** *
   *  Y14p + Y15p                                                            *
   * ********************************************************************** */
   addpool(
      M15s * N15s * C15s,
      Y14p , Y15p , Y15s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-16                                                                                                        //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M15p, N15p, 
      C15p, C16e,

          Y15s, 
      
          w16e, 
        mu_16e,
       var_16e,
      beta_16e,
          6.0,

          Y16e
      );

   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M16e, N16e, 
      C16e, C16d,

          Y16e, 
      
          w16d, 
        mu_16d,
       var_16d,
      beta_16d,
           6.0,

          Y16d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M16d, N16d, 
      C16d, C16p,

          Y16d, 
      
          w16p, 
        mu_16p,
       var_16p,
      beta_16p,

          Y16p
      );

   /* ********************************************************************** *
   *  Y15s + Y16p                                                            *
   * ********************************************************************** */
   addpool(
      M16s * N16s * C16s,
      Y15s , Y16p , Y16s
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // BOTTLENECK-17                                                                                                        //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M16p, N16p, 
      C16p, C17e,

          Y16s, 
      
          w17e, 
        mu_17e,
       var_17e,
      beta_17e,
          6.0,

          Y17e
      );

   // range(
   //    Y16s,
   //    M16s,
   //    N16s, 
   //    C16s
   //    );

   // printf(
   //    "(M1, N1, C1, M2, N2, C2) = (%3d, %3d, %3d, %3d, %3d, %3d)\n", 
   //    M16s, 
   //    N16s, 
   //    C16s, 
   //    M17e, 
   //    N17e, 
   //    C17e
   //    );
   /* ********************************************************************** *
   *  depthwise, conv2d_3x3, stride = 1, same                                *
   * ********************************************************************** */
   dw_conv3x3_p1s1_bnrelu(
      M17e, N17e, 
      C17e, C17d,

          Y17e, 
      
          w17d, 
        mu_17d,
       var_17d,
      beta_17d,
           6.0,

          Y17d
      );

   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bn(
      M17d, N17d, 
      C17d, C17p,

          Y17d, 
      
          w17p, 
        mu_17p,
       var_17p,
      beta_17p,

          Y17p
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // CONV_1                                                                                                               //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   /* ********************************************************************** *
   *  conv2d_1x1, stride = 1, same                                           *
   * ********************************************************************** */
   conv1x1_p0s1_bnrelu(
      M17p, N17p, 
      C17p, C18a,

          Y17p, 
      
          w18a, 
        mu_18a,
       var_18a,
      beta_18a,
          6.0,

          Y18a
      );

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Final-Layer                                                                                                          //
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   mobilenet_avg7x7(
      C18a,
      Y18a, Y19a
      );

   cnn_type_t md = macpool(C18a, Y19a, AxB_dense_1);
   md = md / 49.0 + dense_1_bias;

   return (Y19a);
};

////////////////////////////////////////////////////////////////////////////////////////
// Local tests                                                                        //
////////////////////////////////////////////////////////////////////////////////////////
#define M0 (224)
#define N0 (224)

static cnn_type_t XI[M0 * N0 * 8];

int main()
{  
   srand(4);

   for (int i=0; i < 1024*1024; i++)
   {
      zbuf[i] = 0;
      obuf[i] = 1.0;
   }

   for(int i=0; i < M0 * N0 * 4; i++)
      XI[i] = rand() / 32768.0;

   MobileNetV2 mobilenetv2;
   mobilenetv2.Initialize(M0, N0, 4, MOBILENET_1_4);

   printf("================ SIMD ================\n");
   tic_t start = tic();
   for (int loop = 0; loop < 100; loop++)
   {
      mobilenetv2.Run(XI);
   }
   double ms_per_pic = (toc(&start)) / 100.0;;
   printf("SIMD: MobileNetV2 %9.3f ms per pic.\n", ms_per_pic);

   mobilenetv2.Deinitialize();
   return (0);
}

#ifdef __MATLAB__
/*---------------------------------------------------------------------------*/
/* Matlab "gateway" interface function */
/*---------------------------------------------------------------------------*/
#include <stdio.h>

#include "mex.h"
#include "mexparams.h"

/* input, s2.7, output, s2.6 */
void mexFunction(
   int nlhs,       mxArray *plhs[],
   int nrhs, const mxArray *prhs[]
)
{
   mxload(X, 0);
   mxnew (Y, 1, 1 * 1 * 1792, mxREAL);   /*conv-output:w-h-c*/

   /////////////////////////////////////////////////////////////////////////////
   cnn_type_t * X2 = (cnn_type_t *)_aligned_malloc(X_size * sizeof(cnn_type_t), 16);
   cnn_type_t * Y2;
   for (int i = 0; i < X_size; i++)
   {
      X2[i] = (cnn_type_t)(prX[i]);
   }
   /////////////////////////////////////////////////////////////////////////////

   MobileNetV2 mobilenetv2;
   mobilenetv2.Initialize(M0, N0, 4, MOBILENET_1_4);
   Y2 = mobilenetv2.Run(X2);
   /////////////////////////////////////////////////////////////////////////////
   for (int i = 0; i < Y_size; i++)
   {
      prY[i] = Y2[i];
   }
   /////////////////////////////////////////////////////////////////////////////

   mobilenetv2.Deinitialize();
   mxsave(Y, 0);
}
#endif