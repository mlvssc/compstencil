#include <assert.h>
#include <stdio.h>
#include <sys/time.h>

/* using cuZFP */
#include <math.h>
#include <string.h>
#include "zfp.h"
#include "../src/cuda_zfp/cuWrapper.h"

#include "kernel.hu"
#define BENCH_DIM 3
#define BENCH_FPP 53
#define BENCH_RAD 1

#include "../common_oocd.h"


#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)


#ifndef AN5D_TYPE
#define AN5D_TYPE unsigned
#endif

#define ull size_t
double second()
{
  struct timeval tm;
  double t ;

  static int base_sec = 0,base_usec = 0;

  gettimeofday(&tm, NULL);

  if(base_sec == 0 && base_usec == 0)
  {
      base_sec = tm.tv_sec;
      base_usec = tm.tv_usec;
      t = 0.0;
  } else {
    t = (double) (tm.tv_sec-base_sec) +
      ((double) (tm.tv_usec-base_usec))/1.0e6 ;
  }

  return t ;
}

double kernel_stencil_outcore(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  double start_time = sb_time(), end_time = 0.0;
  int dimsize = compsize + BENCH_RAD * 2;
  SB_TYPE (*A)[dimsize][dimsize][dimsize]
    = (SB_TYPE (*)[dimsize][dimsize][dimsize])A1;
 if (getenv("DIV_N") == NULL || getenv("TB_TPS") == NULL)
	                fprintf(stderr,"ERROR: environment variables (DIV_N, TB_TPS) are not defined.");

			  const int n_div = atoi((getenv("DIV_N")));
			    const int tb_tps = atoi((getenv("TB_TPS")));  


  cudaSetDevice(1);

  const int n_strm = 3;
  size_t plane_xy = dimsize*dimsize; 
  int comp_zdiv = compsize/n_div;
  int si;
  int buf_zlen = comp_zdiv + (tb_tps+1)*BENCH_RAD;
  size_t vol_A = dimsize*plane_xy;
  size_t vol_b = buf_zlen*plane_xy; 
 
  cudaStream_t *pipes;
  pipes = createCuStrms(); // use cuda streams

  SB_TYPE *buf_A;  // only one compute buffer
  cudaCheckReturn(cudaMalloc((void **) &buf_A, (size_t)(2) * (size_t)(vol_b) * sizeof(SB_TYPE)));
  

  SB_TYPE *buf_endo;
  size_t endo_z = 2 * tb_tps * 2 * BENCH_RAD;
  cudaCheckReturn(cudaMalloc((void **) &buf_endo, (size_t)(endo_z*plane_xy) * sizeof(SB_TYPE)));
  size_t vol_e = tb_tps * 2 * BENCH_RAD * plane_xy;
  #define endo_buf buf_endo
  printPnts(A1,dimsize);
  

  // using cuZfp
  #define NUM_STREAM n_strm
  zfp_field *fields_sreg_p1[NUM_STREAM];  /* meta data for shared region in a block */
  zfp_field *fields_remn_p1[NUM_STREAM];  /* meta data for the remnant of the block */  
  zfp_field *fields_sreg_p1_1b;
  zfp_field *fields_sreg_p2[NUM_STREAM];  /* meta data for shared region in a block */
  zfp_field *fields_remn_p2[NUM_STREAM];  /* meta data for the remnant of the block */  
  zfp_field *fields_sreg_p2_1b;

  zfp_stream *zfps_sreg_p1[NUM_STREAM];
  zfp_stream *zfps_remn_p1[NUM_STREAM];  
  zfp_stream *zfps_sreg_p1_1b; 
  zfp_stream *zfps_sreg_p2[NUM_STREAM];
  zfp_stream *zfps_remn_p2[NUM_STREAM];  
  zfp_stream *zfps_sreg_p2_1b;

  bitstream *streams_sreg_p1[NUM_STREAM]; /* bit stream handler to write to or read from */
  bitstream *streams_remn_p1[NUM_STREAM]; /* bit stream handler to write to or read from */
  bitstream *streams_sreg_p1_1b;
  bitstream *streams_sreg_p2[NUM_STREAM]; /* bit stream handler to write to or read from */
  bitstream *streams_remn_p2[NUM_STREAM]; /* bit stream handler to write to or read from */
  bitstream *streams_sreg_p2_1b;
  
  zfp_type dttype = zfp_type_double;
  int rate = 32;

  size_t nx_sreg = tb_tps* BENCH_RAD*plane_xy;
  size_t nx_remn = (comp_zdiv-tb_tps* BENCH_RAD)*plane_xy;
  
  for(int i=0;i<NUM_STREAM;i++)
  {
      fields_sreg_p1[i] = zfp_field_1d(NULL, dttype, nx_sreg);
      fields_remn_p1[i] = zfp_field_1d(NULL, dttype, nx_remn);
      fields_sreg_p2[i] = zfp_field_1d(NULL, dttype, nx_sreg);
      fields_remn_p2[i] = zfp_field_1d(NULL, dttype, nx_remn);

      zfps_sreg_p1[i] = zfp_stream_open(NULL);
      zfps_remn_p1[i] = zfp_stream_open(NULL);
      zfps_sreg_p2[i] = zfp_stream_open(NULL);
      zfps_remn_p2[i] = zfp_stream_open(NULL);

      zfp_stream_set_execution(zfps_sreg_p1[i], zfp_exec_cuda);
      zfp_stream_set_execution(zfps_remn_p1[i], zfp_exec_cuda);
      zfp_stream_set_execution(zfps_sreg_p2[i], zfp_exec_cuda);
      zfp_stream_set_execution(zfps_remn_p2[i], zfp_exec_cuda);

      zfp_stream_set_rate(zfps_sreg_p1[i], rate, dttype, zfp_field_dimensionality(fields_sreg_p1[i]), zfp_false);
      zfp_stream_set_rate(zfps_remn_p1[i], rate, dttype, zfp_field_dimensionality(fields_remn_p1[i]), zfp_false);
      zfp_stream_set_rate(zfps_sreg_p2[i], rate, dttype, zfp_field_dimensionality(fields_sreg_p1[i]), zfp_false);
      zfp_stream_set_rate(zfps_remn_p2[i], rate, dttype, zfp_field_dimensionality(fields_remn_p1[i]), zfp_false);
  }
  
  fields_sreg_p1_1b = zfp_field_1d(NULL, dttype, nx_sreg);
  fields_sreg_p2_1b = zfp_field_1d(NULL, dttype, nx_sreg);
  zfps_sreg_p1_1b = zfp_stream_open(NULL);
  zfps_sreg_p2_1b = zfp_stream_open(NULL);
  zfp_stream_set_execution(zfps_sreg_p1_1b, zfp_exec_cuda);
  zfp_stream_set_rate(zfps_sreg_p1_1b, rate, dttype, zfp_field_dimensionality(fields_sreg_p1[0]), zfp_false);
  zfp_stream_set_execution(zfps_sreg_p2_1b, zfp_exec_cuda);
  zfp_stream_set_rate(zfps_sreg_p2_1b, rate, dttype, zfp_field_dimensionality(fields_sreg_p2[0]), zfp_false);

  size_t size_bs_sreg = zfp_stream_maximum_size(zfps_sreg_p1[0], fields_sreg_p1[0]); // must be fixed
  size_t size_bs_remn = zfp_stream_maximum_size(zfps_remn_p1[0], fields_remn_p1[0]); // must be fixed
  
  void *bss_sreg_p1;     // stream->stream->begin (host)
  void *bss_remn_p1;     // stream->stream->begin (host)
  void *bss_sreg_p2;     // stream->stream->begin (host)
  void *bss_remn_p2;     // stream->stream->begin (host)

  void *d_bss_sreg_p1[NUM_STREAM];   // stream->stream->begin (device)
  void *d_bss_remn_p1[NUM_STREAM];   // stream->stream->begin (device)
  void *d_bss_sreg_p2[NUM_STREAM];   // stream->stream->begin (device)
  void *d_bss_remn_p2[NUM_STREAM];   // stream->stream->begin (device)
  
  void *d_bss_sreg_p1_1b; // for the 1st block
  void *d_bss_sreg_p2_1b; // for the 1st block
  
  cudaMallocHost((void**)&bss_sreg_p1, size_bs_sreg*n_div);
  cudaMallocHost((void**)&bss_remn_p1, size_bs_remn*n_div);
  cudaMallocHost((void**)&bss_sreg_p2, size_bs_sreg*n_div);
  cudaMallocHost((void**)&bss_remn_p2, size_bs_remn*n_div);
  
  for(int i=0;i<NUM_STREAM;i++)
  {
     cudaMalloc((void**)&d_bss_sreg_p1[i], size_bs_sreg);
     cudaMalloc((void**)&d_bss_remn_p1[i], size_bs_remn);
     cudaMalloc((void**)&d_bss_sreg_p2[i], size_bs_sreg);
     cudaMalloc((void**)&d_bss_remn_p2[i], size_bs_remn);
  }
  cudaMalloc((void**)&d_bss_sreg_p1_1b, size_bs_sreg);
  cudaMalloc((void**)&d_bss_sreg_p2_1b, size_bs_sreg);


  /* init zfp */
  {
      int si = 0;
      ull size_jk = plane_xy;
      for (int nb=0;nb<n_div;nb++)
      {
             /* read block to gpu*/
             ull d_start = BENCH_RAD*plane_xy;
             ull h_start = (BENCH_RAD + nb * comp_zdiv) * plane_xy;
             ull size_ijk = comp_zdiv*plane_xy;
             cudaMemcpyAsync(buf_A+d_start,       A1+h_start,       size_ijk*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);
             cudaMemcpyAsync(buf_A+vol_b+d_start, A1+vol_A+h_start, size_ijk*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);            
	         
			 /* bind array to zfp */
             ull d_start_remn = (1+tb_tps)*BENCH_RAD*plane_xy;            
             fields_sreg_p1[si]->data = buf_A + d_start;
             fields_remn_p1[si]->data = buf_A + d_start_remn;
             fields_sreg_p2[si]->data = buf_A + vol_b + d_start;
             fields_remn_p2[si]->data = buf_A + vol_b + d_start_remn;

             streams_sreg_p1[si] = stream_open(d_bss_sreg_p1[si], size_bs_sreg);
             streams_remn_p1[si] = stream_open(d_bss_remn_p1[si], size_bs_remn);
             streams_sreg_p2[si] = stream_open(d_bss_sreg_p2[si], size_bs_sreg);
             streams_remn_p2[si] = stream_open(d_bss_remn_p2[si], size_bs_remn);

             zfp_stream_set_bit_stream(zfps_sreg_p1[si], streams_sreg_p1[si]);
             zfp_stream_set_bit_stream(zfps_remn_p1[si], streams_remn_p1[si]);
             zfp_stream_set_bit_stream(zfps_sreg_p2[si], streams_sreg_p2[si]);
             zfp_stream_set_bit_stream(zfps_remn_p2[si], streams_remn_p2[si]);
			 
             /* compression */
             zfp_compress_async(zfps_sreg_p1[si], fields_sreg_p1[si], si);
             zfp_compress_async(zfps_remn_p1[si], fields_remn_p1[si], si);
             zfp_compress_async(zfps_sreg_p2[si], fields_sreg_p2[si], si);
             zfp_compress_async(zfps_remn_p2[si], fields_remn_p2[si], si);
			 
             /* d2h transfer */
             cudaMemcpyAsync(bss_sreg_p1+nb*size_bs_sreg, d_bss_sreg_p1[si],  size_bs_sreg, cudaMemcpyDeviceToHost, pipes[si]);
             cudaMemcpyAsync(bss_remn_p1+nb*size_bs_remn, d_bss_remn_p1[si],  size_bs_remn, cudaMemcpyDeviceToHost, pipes[si]);
             cudaMemcpyAsync(bss_sreg_p2+nb*size_bs_sreg, d_bss_sreg_p2[si],  size_bs_sreg, cudaMemcpyDeviceToHost, pipes[si]);
             cudaMemcpyAsync(bss_remn_p2+nb*size_bs_remn, d_bss_remn_p2[si],  size_bs_remn, cudaMemcpyDeviceToHost, pipes[si]);
			 
	         cudaStreamSynchronize(pipes[si]);	     
             si=(si+1)%NUM_STREAM;
      }
      cudaDeviceSynchronize();
  }
  
  streams_sreg_p1_1b = stream_open(d_bss_sreg_p1_1b, size_bs_sreg);
  zfp_stream_set_bit_stream(zfps_sreg_p1_1b, streams_sreg_p1_1b);
  streams_sreg_p2_1b = stream_open(d_bss_sreg_p2_1b, size_bs_sreg);
  zfp_stream_set_bit_stream(zfps_sreg_p2_1b, streams_sreg_p2_1b);
  
  /* finish init zfp */
  
  
  cudaEvent_t kernel_evts[NUM_STREAM]; // use cuda events to poll streams

  for(int ie=0;ie<NUM_STREAM;ie++)
  {
	cudaEventCreate(&kernel_evts[ie]);
  }

  bool isfirst = true;
  double cpu0 = second();
  int last_blk=-1;
   int last_strm=-1;
   int blk_i = 0;
   int minus;
   si = 0;
   int nn = timestep;
   while(nn>0)
   {
		minus = nn < tb_tps? nn:tb_tps;
        nn -= minus;
        for (blk_i=0;blk_i<n_div;blk_i++)
		{

            if(blk_i==0)
			{						
				/* transfer compressed blocks and decode them */
                cudaMemcpyAsync(d_bss_sreg_p1[si], bss_sreg_p1, size_bs_sreg, cudaMemcpyHostToDevice, pipes[si]);
                cudaMemcpyAsync(d_bss_remn_p1[si], bss_remn_p1, size_bs_remn, cudaMemcpyHostToDevice, pipes[si]);
                cudaMemcpyAsync(d_bss_sreg_p1_1b,  bss_sreg_p1+size_bs_sreg, size_bs_sreg, cudaMemcpyHostToDevice, pipes[si]);
                cudaMemcpyAsync(d_bss_sreg_p2[si], bss_sreg_p2, size_bs_sreg, cudaMemcpyHostToDevice, pipes[si]);
                cudaMemcpyAsync(d_bss_remn_p2[si], bss_remn_p2, size_bs_remn, cudaMemcpyHostToDevice, pipes[si]);
                cudaMemcpyAsync(d_bss_sreg_p2_1b,  bss_sreg_p2+size_bs_sreg, size_bs_sreg, cudaMemcpyHostToDevice, pipes[si]);
				
				/* wait for previous chunk to finish kernels */
				if(!isfirst){
					int pri_e = (si-1)>=0?(si-1):2;
					cudaStreamWaitEvent(pipes[si],kernel_evts[pri_e],0);
				}
                else {isfirst = false;}
				
				/* set top halo */
				cudaMemcpyAsync(buf_A,       A1,       BENCH_RAD*plane_xy*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);
				cudaMemcpyAsync(buf_A+vol_b, A1+vol_A, BENCH_RAD*plane_xy*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);
				
				/* bind with working space */
				fields_sreg_p1[si]->data = buf_A + BENCH_RAD*plane_xy;
                fields_remn_p1[si]->data = buf_A + (1+tb_tps)*BENCH_RAD*plane_xy; 
                fields_sreg_p1_1b->data =  buf_A + (BENCH_RAD+comp_zdiv)*plane_xy;
                fields_sreg_p2[si]->data = buf_A+vol_b + BENCH_RAD*plane_xy;
                fields_remn_p2[si]->data = buf_A+vol_b + (1+tb_tps)*BENCH_RAD*plane_xy; 
                fields_sreg_p2_1b->data =  buf_A+vol_b + (BENCH_RAD+comp_zdiv)*plane_xy;

                streams_sreg_p1[si] = stream_open(d_bss_sreg_p1[si], size_bs_sreg);
                streams_remn_p1[si] = stream_open(d_bss_remn_p1[si], size_bs_remn);
                streams_sreg_p1_1b = stream_open(d_bss_sreg_p1_1b, size_bs_sreg);
                streams_sreg_p2[si] = stream_open(d_bss_sreg_p2[si], size_bs_sreg);
                streams_remn_p2[si] = stream_open(d_bss_remn_p2[si], size_bs_remn);
                streams_sreg_p2_1b = stream_open(d_bss_sreg_p2_1b, size_bs_sreg);
  
                zfp_stream_set_bit_stream(zfps_sreg_p1[si], streams_sreg_p1[si]);
                zfp_stream_set_bit_stream(zfps_remn_p1[si], streams_remn_p1[si]);
                zfp_stream_set_bit_stream(zfps_sreg_p1_1b, streams_sreg_p1_1b);
                zfp_stream_set_bit_stream(zfps_sreg_p2[si], streams_sreg_p2[si]);
                zfp_stream_set_bit_stream(zfps_remn_p2[si], streams_remn_p2[si]);
                zfp_stream_set_bit_stream(zfps_sreg_p2_1b, streams_sreg_p2_1b);

		        /* decompresee */
                zfp_decompress_async(zfps_sreg_p1[si], fields_sreg_p1[si], si);
                zfp_decompress_async(zfps_remn_p1[si], fields_remn_p1[si], si); 
                zfp_decompress_async(zfps_sreg_p1_1b, fields_sreg_p1_1b, si); 
                zfp_decompress_async(zfps_sreg_p2[si], fields_sreg_p2[si], si);
                zfp_decompress_async(zfps_remn_p2[si], fields_remn_p2[si], si); 
                zfp_decompress_async(zfps_sreg_p2_1b, fields_sreg_p2_1b, si);
			}
			else if(blk_i==n_div-1)
			{
			    /* transfer compressed blocks and decode them */
                cudaMemcpyAsync(d_bss_remn_p1[si], bss_remn_p1+blk_i*size_bs_remn, size_bs_remn, cudaMemcpyHostToDevice, pipes[si]);
                cudaMemcpyAsync(d_bss_remn_p2[si], bss_remn_p2+blk_i*size_bs_remn, size_bs_remn, cudaMemcpyHostToDevice, pipes[si]);
				
				int pri_e = (si-1)>=0?(si-1):2;
				cudaStreamWaitEvent(pipes[si],kernel_evts[pri_e],0);

				/* set bottom halo */
				ull lh_strt = (dimsize-BENCH_RAD) * plane_xy;
				cudaMemcpyAsync(buf_A+(BENCH_RAD+comp_zdiv)*plane_xy,       A1+lh_strt,       BENCH_RAD*plane_xy*sizeof(SB_TYPE),cudaMemcpyHostToDevice,pipes[si]);
				cudaMemcpyAsync(buf_A+vol_b+(BENCH_RAD+comp_zdiv)*plane_xy, A1+vol_A+lh_strt, BENCH_RAD*plane_xy*sizeof(SB_TYPE),cudaMemcpyHostToDevice,pipes[si]);		

                /* bind with working space and decompress */
				fields_remn_p1[si]->data = buf_A+(1+tb_tps)*BENCH_RAD*plane_xy; 
                fields_remn_p2[si]->data = buf_A+vol_b+(1+tb_tps)*BENCH_RAD*plane_xy; 

                streams_remn_p1[si] = stream_open(d_bss_remn_p1[si], size_bs_remn);
                streams_remn_p2[si] = stream_open(d_bss_remn_p2[si], size_bs_remn);

                zfp_stream_set_bit_stream(zfps_remn_p1[si], streams_remn_p1[si]);
                zfp_stream_set_bit_stream(zfps_remn_p2[si], streams_remn_p2[si]);

                zfp_decompress_async(zfps_remn_p1[si], fields_remn_p1[si], si); 
                zfp_decompress_async(zfps_remn_p2[si], fields_remn_p2[si], si); 
			}
			else
			{
				 /* transfer compressed blocks and decode them */
				cudaMemcpyAsync(d_bss_remn_p1[si], bss_remn_p1+blk_i*size_bs_remn, size_bs_remn, cudaMemcpyHostToDevice, pipes[si]);
				cudaMemcpyAsync(d_bss_sreg_p1[si], bss_sreg_p1+(blk_i+1)*size_bs_sreg, size_bs_sreg, cudaMemcpyHostToDevice, pipes[si]);
				cudaMemcpyAsync(d_bss_remn_p2[si], bss_remn_p2+blk_i*size_bs_remn, size_bs_remn, cudaMemcpyHostToDevice, pipes[si]);
				cudaMemcpyAsync(d_bss_sreg_p2[si], bss_sreg_p2+(blk_i+1)*size_bs_sreg, size_bs_sreg, cudaMemcpyHostToDevice, pipes[si]);
				int pri_e = (si-1)>=0?(si-1):2;
				cudaStreamWaitEvent(pipes[si],kernel_evts[pri_e],0); 

				fields_remn_p1[si]->data = buf_A+(1+tb_tps)*BENCH_RAD*plane_xy;  
				fields_sreg_p1[si]->data = buf_A+(BENCH_RAD+comp_zdiv)*plane_xy;  
				fields_remn_p2[si]->data = buf_A+vol_b+(1+tb_tps)*BENCH_RAD*plane_xy; 
				fields_sreg_p2[si]->data = buf_A+vol_b+(BENCH_RAD+comp_zdiv)*plane_xy;  

				zfp_decompress_async(zfps_remn_p1[si], fields_remn_p1[si], si);
				zfp_decompress_async(zfps_sreg_p1[si], fields_sreg_p1[si], si); 
				zfp_decompress_async(zfps_remn_p2[si], fields_remn_p2[si], si);
				zfp_decompress_async(zfps_sreg_p2[si], fields_sreg_p2[si], si); 
			}
	         
if(last_blk!=-1)
{
	 //printf("d2h....\n",last_strm);
	 size_t h_start = (BENCH_RAD+comp_zdiv*last_blk)*plane_xy;
	  size_t d_start = BENCH_RAD*plane_xy;
	   size_t trans_len = comp_zdiv*plane_xy;

	   /* d2h transfer */
	   cudaMemcpyAsync(bss_sreg_p1+last_blk*size_bs_sreg, d_bss_sreg_p1[last_strm],  size_bs_sreg, cudaMemcpyDeviceToHost, pipes[last_strm]);
	   cudaMemcpyAsync(bss_remn_p1+last_blk*size_bs_remn, d_bss_remn_p1[last_strm],  size_bs_remn, cudaMemcpyDeviceToHost, pipes[last_strm]);
	   cudaMemcpyAsync(bss_sreg_p2+last_blk*size_bs_sreg, d_bss_sreg_p2[last_strm],  size_bs_sreg, cudaMemcpyDeviceToHost, pipes[last_strm]);
	   cudaMemcpyAsync(bss_remn_p2+last_blk*size_bs_remn, d_bss_remn_p2[last_strm],  size_bs_remn, cudaMemcpyDeviceToHost, pipes[last_strm]);

	   last_blk = -1;
	   last_strm = -1;
}
			
			
			//printf("computing....\n");

             //#if 0	
	     // computing for tb_tps times
	     {
        	#define __c0 c0
        	AN5D_TYPE __c1Len = buf_zlen - 1 - 1;
        	AN5D_TYPE __c1Pad = (1);

        	#define __c1 c1
                AN5D_TYPE __c2Len = (dimsize - 1 - 1);
        	AN5D_TYPE __c2Pad = (1);
        	#define __c2 c2
        	AN5D_TYPE __c3Len = (dimsize - 1 - 1);
        	AN5D_TYPE __c3Pad = (1);
        	#define __c3 c3
        	AN5D_TYPE __halo1 = 1;
        	AN5D_TYPE __halo2 = 1;
        	AN5D_TYPE __halo3 = 1;
        	AN5D_TYPE c0;
        	AN5D_TYPE __side0LenMax;

        	AN5D_TYPE __side0Len = 1;
        	AN5D_TYPE __side1Len = 128;
        	AN5D_TYPE __side2Len = 30;
        	AN5D_TYPE __side3Len = 30;
        	AN5D_TYPE __OlLen1 = (__halo1 * __side0Len);
        	AN5D_TYPE __OlLen2 = (__halo2 * __side0Len);
        	AN5D_TYPE __OlLen3 = (__halo3 * __side0Len);
        	AN5D_TYPE __side1LenOl = (__side1Len + 2 * __OlLen1);
        	AN5D_TYPE __side2LenOl = (__side2Len + 2 * __OlLen2);
        	AN5D_TYPE __side3LenOl = (__side3Len + 2 * __OlLen3);
        	AN5D_TYPE __blockSize = 1 * __side2LenOl * __side3LenOl;
        	dim3 k0_dimBlock(__blockSize, 1, 1);
        	
			for (c0 = 0; c0 < tb_tps; c0 += 1)
        	{

			if(blk_i!=0) // read two planes
             		{	
                  	   size_t buf_start  = (tb_tps-1-c0)*BENCH_RAD*plane_xy;
                           size_t endo_start = 2*BENCH_RAD*c0*plane_xy;
			   size_t biplane = 2ull*BENCH_RAD*plane_xy;
                          #ifdef DDEBUG_ON
			   printf("s%d blk_%d read: endo_strt=%d -> buf_strt=%d \n", si, blk_i, endo_start/plane_xy, buf_start/plane_xy);
			  #endif
			   cudaMemcpyAsync(buf_A+buf_start, endo_buf+endo_start, biplane*sizeof(SB_TYPE), cudaMemcpyDeviceToDevice, pipes[si]);
			   cudaMemcpyAsync(buf_A+vol_b+buf_start, endo_buf+vol_e+endo_start, biplane*sizeof(SB_TYPE), cudaMemcpyDeviceToDevice, pipes[si]);
             		}
                	if(blk_i!=n_div-1) // write two planes
             		{
			   size_t buf_start  = ((tb_tps-1-c0)*BENCH_RAD+comp_zdiv)*plane_xy;
			   size_t endo_start = 2*BENCH_RAD*c0*plane_xy;
			   size_t biplane = 2ull*BENCH_RAD*plane_xy;
#ifdef DDEBUG_ON 
			   printf("s%d blk_%d write: endo_strt=%d <- buf_strt=%d \n",si, blk_i, endo_start/plane_xy, buf_start/plane_xy);
#endif 
			   cudaMemcpyAsync(endo_buf+endo_start, buf_A+buf_start, biplane*sizeof(SB_TYPE), cudaMemcpyDeviceToDevice, pipes[si]);
													                                           cudaMemcpyAsync(endo_buf+vol_e+endo_start ,buf_A+vol_b+buf_start, biplane*sizeof(SB_TYPE), cudaMemcpyDeviceToDevice, pipes[si]);
             		}	     
             

			__c1Len = comp_zdiv;
			__c1Pad = (tb_tps-c0)*BENCH_RAD;
			int blktype = -1;
			if(blk_i==0)
			{
				__c1Len = (tb_tps-1-c0)*BENCH_RAD+comp_zdiv;
				__c1Pad = BENCH_RAD;
				blktype = 0;
			}
			if(blk_i==n_div-1)
			{
				__c1Len = comp_zdiv+(c0+1-tb_tps)*BENCH_RAD;
				blktype = 1;
			}
#ifdef DDEBUG_ON
	                 printf("s%d blk_%d comp.: buf_strt=%d, comp_len=%d btype=%d\n",si, blk_i,  __c1Pad, __c1Len,blktype);		
		        #endif
			 unsigned int n_zblk = (__c1Len + __side1Len - 1) / __side1Len;
			 unsigned int l_blk = __c1Len%__side1Len;
			 if(l_blk!=0&&n_zblk!=1) n_zblk--;
			dim3 k0_dimGrid(1 * (n_zblk) * ((__c2Len + __side2Len - 1) / __side2Len) * ((__c3Len + __side3Len - 1) / __side3Len), 1, 1);

		        kernel0_1_pipe<<<k0_dimGrid, k0_dimBlock, 0, pipes[si]>>> (buf_A, dimsize, buf_zlen, tb_tps, c0, blktype);
        	}
	     }
fields_sreg_p1[si]->data = buf_A+BENCH_RAD*plane_xy;
fields_remn_p1[si]->data = buf_A+(1+tb_tps)*BENCH_RAD*plane_xy;
fields_sreg_p2[si]->data = buf_A+vol_b+BENCH_RAD*plane_xy;
fields_remn_p2[si]->data = buf_A+vol_b+(1+tb_tps)*BENCH_RAD*plane_xy;

streams_sreg_p1[si] = stream_open(d_bss_sreg_p1[si], size_bs_sreg);
streams_remn_p1[si] = stream_open(d_bss_remn_p1[si], size_bs_remn);
streams_sreg_p2[si] = stream_open(d_bss_sreg_p2[si], size_bs_sreg);
streams_remn_p2[si] = stream_open(d_bss_remn_p2[si], size_bs_remn);

zfp_stream_set_bit_stream(zfps_sreg_p1[si], streams_sreg_p1[si]);
zfp_stream_set_bit_stream(zfps_remn_p1[si], streams_remn_p1[si]);
zfp_stream_set_bit_stream(zfps_sreg_p2[si], streams_sreg_p2[si]);
zfp_stream_set_bit_stream(zfps_remn_p2[si], streams_remn_p2[si]);

/* compression */
zfp_compress_async(zfps_sreg_p1[si], fields_sreg_p1[si], si);
zfp_compress_async(zfps_remn_p1[si], fields_remn_p1[si], si);
zfp_compress_async(zfps_sreg_p2[si], fields_sreg_p2[si], si);
zfp_compress_async(zfps_remn_p2[si], fields_remn_p2[si], si);
cudaEventRecord(kernel_evts[si],pipes[si]);

	     last_blk = blk_i;
             last_strm = si;
             si = (si+1)%n_strm;
	} // end task loop
	} // end nn loop 
    #ifdef RUN_SEQ
                cudaDeviceSynchronize();
		             #endif

	if(last_blk!=-1)
			{
				/* d2h transfer */
                cudaMemcpyAsync(bss_sreg_p1+last_blk*size_bs_sreg, d_bss_sreg_p1[last_strm],  size_bs_sreg, cudaMemcpyDeviceToHost, pipes[last_strm]);
                cudaMemcpyAsync(bss_remn_p1+last_blk*size_bs_remn, d_bss_remn_p1[last_strm],  size_bs_remn, cudaMemcpyDeviceToHost, pipes[last_strm]);
                cudaMemcpyAsync(bss_sreg_p2+last_blk*size_bs_sreg, d_bss_sreg_p2[last_strm],  size_bs_sreg, cudaMemcpyDeviceToHost, pipes[last_strm]);
                cudaMemcpyAsync(bss_remn_p2+last_blk*size_bs_remn, d_bss_remn_p2[last_strm],  size_bs_remn, cudaMemcpyDeviceToHost, pipes[last_strm]);
				 
				last_blk = -1;
				last_strm = -1;
			}
	
	
	cudaDeviceSynchronize();
     double cpu1 = second();
  int need_decode = 1;
  if(need_decode)
  {
      int si = 0;
      for (int nb=0;nb<n_div;nb++)
      {
             ull d_start = BENCH_RAD*plane_xy;
             ull h_start = (BENCH_RAD + nb * comp_zdiv) * plane_xy;
             ull size_ijk = comp_zdiv*plane_xy;
             /* h2d transfer */
			 cudaMemcpyAsync(d_bss_sreg_p1[si], bss_sreg_p1+nb*size_bs_sreg,   size_bs_sreg, cudaMemcpyHostToDevice, pipes[si]);
             cudaMemcpyAsync(d_bss_remn_p1[si], bss_remn_p1+nb*size_bs_remn,   size_bs_remn, cudaMemcpyHostToDevice, pipes[si]);
             cudaMemcpyAsync(d_bss_sreg_p2[si], bss_sreg_p2+nb*size_bs_sreg,   size_bs_sreg, cudaMemcpyHostToDevice, pipes[si]);
             cudaMemcpyAsync(d_bss_remn_p2[si], bss_remn_p2+nb*size_bs_remn,   size_bs_remn, cudaMemcpyHostToDevice, pipes[si]);
             
             /* bind array to zfp */
             ull d_start_remn = (1+tb_tps)*BENCH_RAD*plane_xy;            
             fields_sreg_p1[si]->data = buf_A+d_start;
             fields_remn_p1[si]->data = buf_A+d_start_remn;
             fields_sreg_p2[si]->data = buf_A+vol_b+d_start;
             fields_remn_p2[si]->data = buf_A+vol_b+d_start_remn;
			 
             streams_sreg_p1[si] = stream_open(d_bss_sreg_p1[si], size_bs_sreg);
             streams_remn_p1[si] = stream_open(d_bss_remn_p1[si], size_bs_remn);
             streams_sreg_p2[si] = stream_open(d_bss_sreg_p2[si], size_bs_sreg);
             streams_remn_p2[si] = stream_open(d_bss_remn_p2[si], size_bs_remn);
			 
             zfp_stream_set_bit_stream(zfps_sreg_p1[si], streams_sreg_p1[si]);
             zfp_stream_set_bit_stream(zfps_remn_p1[si], streams_remn_p1[si]);            
			 zfp_stream_set_bit_stream(zfps_sreg_p2[si], streams_sreg_p2[si]);
             zfp_stream_set_bit_stream(zfps_remn_p2[si], streams_remn_p2[si]);
             
             /* decompress */
             zfp_decompress_async(zfps_sreg_p1[si], fields_sreg_p1[si], si);
             zfp_decompress_async(zfps_remn_p1[si], fields_remn_p1[si], si);			 
             zfp_decompress_async(zfps_sreg_p2[si], fields_sreg_p2[si], si);
             zfp_decompress_async(zfps_remn_p2[si], fields_remn_p2[si], si);

             /* read block to gpu*/

             cudaMemcpyAsync(A1+h_start, 	   buf_A+d_start,  		 size_ijk*sizeof(SB_TYPE),  cudaMemcpyDeviceToHost, pipes[si]);
             cudaMemcpyAsync(A1+vol_A+h_start, buf_A+vol_b+d_start,  size_ijk*sizeof(double), cudaMemcpyDeviceToHost, pipes[si]);

			 cudaStreamSynchronize(pipes[si]);
			 si=(si+1)%NUM_STREAM;
      }
      cudaDeviceSynchronize();
  }	
	
	printPnts(A1,dimsize);

	cudaCheckReturn(cudaFree(buf_A));
	cudaCheckReturn(cudaFree(buf_endo));
  
  cudaFreeHost(bss_sreg_p1);
  cudaFreeHost(bss_remn_p1);
  cudaFreeHost(bss_sreg_p2);
  cudaFreeHost(bss_remn_p2);
  
  for(int i=0;i<NUM_STREAM;i++)
  {
     cudaFree(d_bss_sreg_p1[i]);
     cudaFree(d_bss_remn_p1[i]);
     cudaFree(d_bss_sreg_p2[i]);
     cudaFree(d_bss_remn_p2[i]);
  }
  cudaFree(d_bss_sreg_p1_1b);
  cudaFree(d_bss_sreg_p2_1b);
   
 printf("elapsed time =%.4f\n", cpu1-cpu0);
  return (cpu1-cpu0)*1000.0;
 }

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  
  return 0.0;
  }
