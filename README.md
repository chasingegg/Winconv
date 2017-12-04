# Winograd on CNN 

## Background

The success of convolutional neural networks is limited by how fast we can compute them. We use the winograd's minimal filtering algorithms for small, 3*3 filters on CNN for intel architecture, specifically the Knights Landing Architecture.  However, the reduction of arithmetic operations in Winograd algorithm comes at the cost of complicating the memory accesses. 

---

Direct convolution of a batch of N input images of dimension HxWxC with F filters of dimension RxSxC requires O(NFHWCRS) floating point operations. One useful method is the Fast Fourier Transform method(FFT). It is required that the size of th filter should be large enough to take advantage of it. 

---

## Winograd

In simple words, Winograd works on one image tile at a time. Let the tile size be 4x4, and a 3x3 filter, we can get 2x2 output. We can do the transformation that 4x4 for input data, 4x3 for filter data and 2x4 for the inverse transform that produces a 2x2 filtered output on one tile. Ignoring the transformation, we ohly need 16 element wise product, this results in a speedup of 36/16=2.25. To ignore the transformation cost, we need to transform element wise product to a GEMM operation. 

---

Each image frame has T = (H-2)x(W-2)/4 tiles and there are NxC such frames. We hope that a transformed input tile can be reused to multiply with corresponding F filter tiles. Similarly, a transformed filter tile should be reused to multiply with corresponding input tiles across all the batches. This can be turned into GEMM form by scattering the 16 elements of every tile to 16 different matrices to form the inputs for the GEMM. Therefore, input is converted 16 matrices each of T = (H-2)x(W-2)/4 rows and C columns. Filter is converted to 16 matrices each of C rows and F columns. 

---

So there are 16 matrices of dimension TxC for an image input and 16 matrices of dimension CxF that represent filters, resulting in 16 one to one matrix multiplications. Turning to GEMM results in the maximum reuse of every transformed tile and increases the code-vectorization capabilities that will lead to better performance. OpenMP is used for multithreading since it is more scalable and portable than pthreads. 

## Reference 

- [Fast Algorithms for Convolutional Neural Networks](reference/1509.09308.pdf)
- [Optimization of Spatial Convolution in ConvNets on Intel KNL](reference/Optimization.pdf)
- [FLACON Library](https://github.com/ColfaxResearch/FALCON)