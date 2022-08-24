#include <assert.h>
#include <stdio.h>
#include "kernel.hu"
#define BENCH_DIM 3
#define BENCH_FPP 25
#define BENCH_RAD 2

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

#include <sys/time.h>


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
  
  cudaSetDevice(1);
  const int n_div = atoi((getenv("DIV_N")));
  const int tb_tps = atoi((getenv("TB_TPS")));
  const int n_strm = 3;
  size_t plane_xy = dimsize*dimsize; 
  int comp_zdiv = compsize/n_div;
  int si;
  cudaStream_t pipes[n_strm];
  int buf_zlen = comp_zdiv + (tb_tps+1)*BENCH_RAD;
  size_t vol_A = dimsize*plane_xy;
  size_t vol_b = buf_zlen*plane_xy; 
 
  SB_TYPE *buf_A[n_strm]; 
  for (si=0;si<n_strm;si++){
    cudaCheckReturn(cudaMalloc((void **) &buf_A[si], (size_t)(2) * (size_t)(vol_b) * sizeof(SB_TYPE)));
    cudaCheckReturn(cudaStreamCreate(&pipes[si]));
  }


  SB_TYPE *buf_endo;
  size_t endo_z = 2 * tb_tps * 2 * BENCH_RAD;
  cudaCheckReturn(cudaMalloc((void **) &buf_endo, (size_t)(endo_z*plane_xy) * sizeof(SB_TYPE)));
  size_t vol_e = tb_tps * 2 * BENCH_RAD * plane_xy;
  #define endo_buf buf_endo
  printPnts(A1,dimsize);
   int last_blk=-1;
   int last_strm=-1;
   int blk_i = 0;
   int minus;
   si = 0;
   int nn = timestep;
   double cpu0=second();
   while(nn>0)
   {
	minus = nn < tb_tps? nn:tb_tps;
        nn -= minus;
        for (blk_i=0;blk_i<n_div;blk_i++)
	{
	     /* HtoD transfer */
	     //printf("h2d....\n");
             if(blk_i==0)
	     {
	        // size_t d_start = (tb_tps-1)*BENCH_RAD*plane_xy;
   		size_t trans_len = (BENCH_RAD*(1+tb_tps)+comp_zdiv)*plane_xy;
		#ifdef DDEBUG_ON
		printf("s%d blk%d h2d A[0]->dbuf[%d] len=%d\n",si, blk_i, 0, trans_len/plane_xy);
		#endif
		cudaMemcpyAsync(buf_A[si],        A1,       trans_len*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);
		cudaMemcpyAsync(buf_A[si]+vol_b,  A1+vol_A, trans_len*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);
                //printf("1st block h2d completed....\n");
	     }
	     else if(blk_i==n_div-1)
	     {
                size_t d_start = BENCH_RAD*(1+tb_tps)*plane_xy;
		size_t h_start = (BENCH_RAD*(1+tb_tps)+comp_zdiv*blk_i)*plane_xy;
		size_t trans_len = (BENCH_RAD*(1-tb_tps)+comp_zdiv)*plane_xy;
		#ifdef DDEBUG_ON
		printf("s%d blk%d h2d A[%d]->dbuf[0] len=%d\n",si, blk_i, h_start/plane_xy, trans_len/plane_xy);
		#endif
		cudaMemcpyAsync(buf_A[si]+d_start, A1+h_start, trans_len*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);
                cudaMemcpyAsync(buf_A[si]+vol_b+d_start, A1+vol_A+h_start, trans_len*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);
		}
	     else
	     {
		size_t h_start = (BENCH_RAD*(1+tb_tps)+comp_zdiv*blk_i)*plane_xy;
                size_t d_start = BENCH_RAD*(1+tb_tps)*plane_xy;
		size_t trans_len = comp_zdiv*plane_xy;
		#ifdef DDEBUG_ON
		printf("s%d blk%d h2d A[%ld]->dbuf[0] len=%ld\n",si, blk_i, h_start/plane_xy, trans_len/plane_xy);
		#endif
		cudaMemcpyAsync(buf_A[si]+d_start, A1+h_start, trans_len*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);
                cudaMemcpyAsync(buf_A[si]+vol_b+d_start, A1+vol_A+h_start, trans_len*sizeof(SB_TYPE), cudaMemcpyHostToDevice, pipes[si]);
	       //printf("middle block h2d complete....\n");
	     }
	      #ifdef RUN_SEQ
	                  cudaDeviceSynchronize();
			               #endif
	     if(last_blk!=-1)
	     {
		 //printf("d2h....\n",last_strm);
		 size_t h_start = (BENCH_RAD+comp_zdiv*last_blk)*plane_xy;
		 size_t d_start = BENCH_RAD*plane_xy;
		 size_t trans_len = comp_zdiv*plane_xy;
#ifdef DDEBUG_ON
		 printf("s%d blk%d d2h A[%d]<-dbuf[%d] len=%d\n",last_strm, last_blk, h_start/plane_xy, d_start/plane_xy, trans_len/plane_xy);
#endif 
		 cudaMemcpyAsync(A1+h_start, buf_A[last_strm]+d_start,  trans_len*sizeof(SB_TYPE), cudaMemcpyDeviceToHost, pipes[last_strm]);
                 cudaMemcpyAsync(A1+vol_A+h_start, buf_A[last_strm]+vol_b+d_start, trans_len*sizeof(SB_TYPE), cudaMemcpyDeviceToHost, pipes[last_strm]);
		 //cudaCheckReturn(cudaMemcpy(A1+vol_A+h_start, buf_A[last_strm]+vol_b+d_start, trans_len*sizeof(SB_TYPE), cudaMemcpyDeviceToHost));
		 last_blk = -1;
		 last_strm = -1;
	     }
             #ifdef RUN_SEQ
	     cudaDeviceSynchronize();
	     #endif
	         //printf("computing....\n");

             //#if 0	
	     // computing for tb_tps times
	     {
        	#define __c0 c0
        	AN5D_TYPE __c1Len = buf_zlen - 2 - 2;
        	AN5D_TYPE __c1Pad = (2);

        	#define __c1 c1
                AN5D_TYPE __c2Len = (dimsize - 2 - 2);
        	AN5D_TYPE __c2Pad = (2);
        	#define __c2 c2
        	AN5D_TYPE __c3Len = (dimsize - 2 - 2);
        	AN5D_TYPE __c3Pad = (2);
        	#define __c3 c3
        	AN5D_TYPE __halo1 = 2;
        	AN5D_TYPE __halo2 = 2;
        	AN5D_TYPE __halo3 = 2;
        	AN5D_TYPE c0;
        	AN5D_TYPE __side0LenMax;

        	AN5D_TYPE __side0Len = 1;
        	AN5D_TYPE __side1Len = 64;;
        	AN5D_TYPE __side2Len = 28;
        	AN5D_TYPE __side3Len = 28;
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
			   cudaMemcpyAsync(buf_A[si]+buf_start, endo_buf+endo_start, biplane*sizeof(SB_TYPE), cudaMemcpyDeviceToDevice, pipes[si]);
			   cudaMemcpyAsync(buf_A[si]+vol_b+buf_start, endo_buf+vol_e+endo_start, biplane*sizeof(SB_TYPE), cudaMemcpyDeviceToDevice, pipes[si]);
             		}
                	if(blk_i!=n_div-1) // write two planes
             		{
			   size_t buf_start  = ((tb_tps-1-c0)*BENCH_RAD+comp_zdiv)*plane_xy;
			   size_t endo_start = 2*BENCH_RAD*c0*plane_xy;
			   size_t biplane = 2ull*BENCH_RAD*plane_xy;
#ifdef DDEBUG_ON 
			   printf("s%d blk_%d write: endo_strt=%d <- buf_strt=%d \n",si, blk_i, endo_start/plane_xy, buf_start/plane_xy);
#endif 
			   cudaMemcpyAsync(endo_buf+endo_start, buf_A[si]+buf_start, biplane*sizeof(SB_TYPE), cudaMemcpyDeviceToDevice, pipes[si]);
													                                           cudaMemcpyAsync(endo_buf+vol_e+endo_start ,buf_A[si]+vol_b+buf_start, biplane*sizeof(SB_TYPE), cudaMemcpyDeviceToDevice, pipes[si]);
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
			dim3 k0_dimGrid(1 * ((__c1Len + __side1Len - 1) / __side1Len) * ((__c2Len + __side2Len - 1) / __side2Len) * ((__c3Len + __side3Len - 1) / __side3Len), 1, 1);
                        //printf("c0=%d __c1Len=%d __c1Pad=%d buf_zlen=%d\n",c0, __c1Len,__c1Pad,buf_zlen);

		        kernel0_1_pipe<<<k0_dimGrid, k0_dimBlock, 0, pipes[si]>>> (buf_A[si], dimsize, buf_zlen, tb_tps, c0, blktype);
        	}
	     }
	     //printf("computing completed....\n");
             //#endif
	      #ifdef RUN_SEQ
	                  cudaDeviceSynchronize();
			               #endif
	     last_blk = blk_i;
             last_strm = si;
             si = (si+1)%n_strm;
	} // end task loop
	} // end nn loop 
//#if 0  
    #ifdef RUN_SEQ
                cudaDeviceSynchronize();
		             #endif
        if(last_blk!=-1)
        {
                 size_t h_start = (BENCH_RAD+comp_zdiv*last_blk)*plane_xy;
                 size_t d_start = BENCH_RAD*plane_xy;
                 size_t trans_len = comp_zdiv*plane_xy;
#ifdef DDEBUG_ON
		 printf("s%d blk_%d d2h buf[%d]->A[%d] len=%d\n",last_strm, last_blk, d_start/plane_xy,h_start/plane_xy, trans_len/plane_xy);
#endif
		 cudaMemcpyAsync(A1+h_start, buf_A[last_strm]+d_start,  trans_len*sizeof(SB_TYPE), cudaMemcpyDeviceToHost, pipes[last_strm]);
                 cudaMemcpyAsync(A1+vol_A+h_start, buf_A[last_strm]+vol_b+d_start, trans_len*sizeof(SB_TYPE), cudaMemcpyDeviceToHost, pipes[last_strm]);
                 
		 //cudaCheckReturn(cudaMemcpy(A1+h_start, buf_A[last_strm]+d_start,  trans_len*sizeof(SB_TYPE), cudaMemcpyDeviceToHost));
		 //cudaCheckReturn(cudaMemcpy(A1+vol_A+h_start, buf_A[last_strm]+vol_b+d_start, trans_len*sizeof(SB_TYPE), cudaMemcpyDeviceToHost));
       }
//#endif
	cudaDeviceSynchronize();
	double cpu1 = second();
        printPnts(A1,dimsize);

	cudaCheckReturn(cudaFree(buf_A[0]));
        cudaCheckReturn(cudaFree(buf_A[1]));
	cudaCheckReturn(cudaFree(buf_endo));

	//cudaCheckReturn(cudaFreeHost(B));
  //return (((end_time != 0.0) ? end_time : sb_time()) - start_time);
	printf("elapsed time=%.5fs\n", cpu1-cpu0);
	return (cpu1-cpu0)*1000.0;
 }

double kernel_stencil(SB_TYPE *A1, int compsize, int timestep, bool scop)
{
  
  return 0.0;
  }
