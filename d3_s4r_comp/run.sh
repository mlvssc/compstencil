rm tst
#nvcc -o tst host.cu kernel.cu  -DDDEBUG_ON -maxrregcount 32 -Xcompiler -O3
#nvcc -o tst host.cu kernel.cu -maxrregcount 32 -Xcompiler -O3
nvcc -O3 -Xcompiler -fopenmp -lgomp -I../../include -L../../lib -lzfp  -lm -o tst  host.cu kernel.cu -maxrregcount 32
#./tst
