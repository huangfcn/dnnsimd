#ifndef __UTILS_TIMER_TICTOC_H__
#define __UTILS_TIMER_TICTOC_H__

#ifdef __cplusplus
extern "C"{
#endif

#ifdef _WIN32

#include <windows.h>

typedef LARGE_INTEGER tic_t;

static tic_t tic()
{
   LARGE_INTEGER curtic;
   QueryPerformanceCounter(&curtic);

   return  curtic;
}

static double toc(tic_t * oldCount)
{
    LARGE_INTEGER frequency;           // ticks per second
    LARGE_INTEGER t1 = *oldCount, t2;  // ticks
    double elapsedTime;

    // get ticks per second
    QueryPerformanceFrequency(&frequency);

    // stop timer
    QueryPerformanceCounter(&t2);

    // compute and print the elapsed time in millisec
    elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;

    return elapsedTime;
}

#else

#include <sys/time.h>

typedef struct timeval tic_t;

static tic_t tic()
{
   timeval curtic;
   gettimeofday(&curtic, NULL);
    return  curtic;
}

static double toc(tic_t * oldCount)
{
    timeval newCount;

   gettimeofday(&newCount, NULL);
   double t = double(newCount.tv_sec  - oldCount->tv_sec ) + 
              double(newCount.tv_usec - oldCount->tv_usec) * 1.e-9;
   return (t * 1000.0);
}
#endif

typedef tic_t tictoc_t;

#ifdef __cplusplus
}
#endif

#endif