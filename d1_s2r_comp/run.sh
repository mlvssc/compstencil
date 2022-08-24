rm tst
nvcc -O3 -Xcompiler -fopenmp -lgomp -I../../include -L../../lib -lzfp  -lm -o tst host.cu kernel.cu -maxrregcount 32

