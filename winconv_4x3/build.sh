mpiicc -O3 -qopenmp -xHost -restrict -I ./  -lmkl_rt  -lmkl_blacs_intelmpi_ilp64 -liomp5 -lpthread -ldl  winconv.cpp winconv_4x3.cpp test.cpp -o test -lm
