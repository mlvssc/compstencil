#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>

#define STENCILBENCH

#define SB_START_INSTRUMENTS do { start_time = sb_time(); } while (0)
#define SB_STOP_INSTRUMENTS  do { end_time   = sb_time(); } while (0)

#ifndef SB_TYPE
#define SB_TYPE double
#endif

#define __HALO 1

void printPnts(SB_TYPE *A, int dimsize)
{
	   int seed[] = {512,26,8964,114,514,47,564,100,89,64};
	   for (int i =0;i<10;i++)
	   {
   	          srand(seed[i]);
		  double rdn = (double)rand() / (double)RAND_MAX;
		  int   randn = ceil(rdn*(dimsize));
		  if (randn<__HALO) randn *= 2;
														                if (randn>=dimsize-__HALO) randn /= 2;			

																size_t vol = 1ull*dimsize*dimsize*dimsize;
	
																size_t addr1 = 1ull*randn*dimsize*dimsize+randn*dimsize+randn;

																printf("A[0][%d^3]=%f A[1][%d^3]=%f\n",randn, A[addr1],randn,A[vol+addr1]);

	   }


	        printf("--------------------------------------------------------\n");

       }





void usage(char *cmd)
{
  printf("%s [OPTION] ...\n", cmd);
  printf("\n");
  printf("-s" "\t" "Specify compute size" "\n");
  printf("-t" "\t" "Specify the number of timesteps" "\n");
  printf("-n" "\t" "Specify the number of execution" "\n");
  printf("-c" "\t" "Enable comparison to CPU execution" "\n");
  printf("-h" "\t" "Show this usage" "\n");
}


double sb_time()
{
  struct timespec tp;

  if (clock_gettime(CLOCK_MONOTONIC_RAW, &tp) != 0)
    perror("clock_gettime failed");

  return (tp.tv_sec + tp.tv_nsec * 1.0e-9) * 1.0e3;
}

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop);
double kernel_stencil_outcore(SB_TYPE *A1, int compsize, int timestep, bool scop);

void init_grid(SB_TYPE *A, int compsize)
{
  size_t gridarea = 1, dimsize = compsize + BENCH_RAD * 2;
  for (int i = 0; i < BENCH_DIM; i++)
    gridarea *= dimsize;

#pragma omp parallel for
  for (size_t i = 0; i < gridarea; i++) {
    unsigned int seed = i + compsize;
    A[0 * gridarea + i] = 1000 * (SB_TYPE)(rand_r(&seed)) / (SB_TYPE)(RAND_MAX);
    A[1 * gridarea + i] = A[0 * gridarea + i];
  }
}

void check_result(SB_TYPE *A, SB_TYPE *B, int compsize, int timestep)
{
  int t = timestep % 2;

  size_t gridarea = 1, dimsize = compsize + BENCH_RAD * 2;
  for (int i = 0; i < BENCH_DIM; i++)
    gridarea *= dimsize;

  double error_max = 0.0;
  long double error_sum = 0.0;

#pragma omp parallel for reduction(max:error_max) reduction(+:error_sum)
  for (size_t i = 0; i < gridarea; i++) {
    double error = fabs((double)(A[t * gridarea + i] - B[t * gridarea + i]));
    error_max = (error > error_max) ? error : error_max;
    error_sum += error * error;
  }

  printf("Max Error: %e\n",  error_max);
  printf("RMS Error: %Le\n", sqrtl(error_sum / gridarea));
}

double calc_gflops(double total_flop, double ms)
{
  return (total_flop / ms) / (1000 * 1000);
}

int benchmark(int compsize, int timestep, int reptnum, bool cpucomp)
{
  size_t gridarea = 1, comparea = 1;

  for (int i = 0; i < BENCH_DIM; i++) {
    gridarea *= (compsize + BENCH_RAD * 2);
    comparea *= compsize;
  }

  double total_flop = (double)timestep * comparea * BENCH_FPP;

  SB_TYPE *A = (SB_TYPE *)malloc(2 * gridarea * sizeof(SB_TYPE));
  SB_TYPE *B = (SB_TYPE *)malloc(2 * gridarea * sizeof(SB_TYPE));

  if (A == NULL || B == NULL) {
    fprintf(stderr, "Host memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  //puts("[Device warm-up]");
  //printf("%.4f ms\n", kernel_stencil(A, compsize, timestep, true));

  double elapsed = 0.0f;
  puts("\n" "[Device execution]");
  for (int n = 0; n < reptnum; n++) {
    init_grid(A, compsize);
    double ms = kernel_stencil(A, compsize, timestep, true);
    printf("%7d: %10.4lf GFLOPS, %10.4f ms\n",
           n, calc_gflops(total_flop, ms), ms);
    elapsed += ms;
  }

  printf("Average: %10.4lf GFLOPS, %10.4f ms\n",
         calc_gflops(total_flop, elapsed / reptnum), elapsed / reptnum);

  if (cpucomp) {
    puts("\n" "[CPU execution]");
    init_grid(B, compsize);
    printf("%.4f ms\n", kernel_stencil(B, compsize, timestep, false));
    check_result(A, B, compsize, timestep);
  }

  free(A);
  free(B);

  return EXIT_SUCCESS;
}

int benchmark_outcore(int compsize, int timestep, int reptnum, bool cpucomp)
{
  size_t gridarea = 1, comparea = 1;

  for (int i = 0; i < BENCH_DIM; i++) {
    gridarea *= (compsize + BENCH_RAD * 2);
    comparea *= compsize;
  }

  double total_flop = (double)timestep * comparea * BENCH_FPP;

  SB_TYPE *A; // use pinned memory 
  size_t byte_len = 2ull * gridarea * sizeof(SB_TYPE); 
  cudaError_t status = cudaMallocHost((void**)&A, byte_len);
  if (status != cudaSuccess)
    printf("Error allocating pinned host memory\n");

  //puts("[Device warm-up]");
  //printf("%.4f ms\n", kernel_stencil(A, compsize, timestep, true));

  double elapsed = 0.0f;
  puts("\n" "[Device execution]");
  for (int n = 0; n < reptnum; n++) {
    init_grid(A, compsize);
    double ms = kernel_stencil_outcore(A, compsize, timestep, true);
    printf("%7d: %10.4lf GFLOPS, %10.4f ms\n",
           n, calc_gflops(total_flop, ms), ms);
    elapsed += ms;
  }

  printf("Average: %10.4lf GFLOPS, %10.4f ms\n",
         calc_gflops(total_flop, elapsed / reptnum), elapsed / reptnum);


  cudaFreeHost(A);

  return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
  int compsize = -1;
  int timestep = -1;
  int reptnum = 1;
  bool cpucomp = false;
  
  if(argc<3) {printf("enter size_z, size_x/y and total steps.\n"); return -1;}
  
  compsize = atoi(argv[1]);
  timestep = atoi(argv[2]);

  return benchmark_outcore(compsize, timestep, reptnum, cpucomp);
}
