# dnnsimd
deep learning convolution neural network implemented with SIMD acceleration (auto-vectorization)

# method
	1, align the tensor data address
	2, unroll add/multiplications
	3, loop as simple as possible
	4, processing block as big as possible

# Sample assembly language generated
Here we dump typical assembly code generated by gcc. We can see that most code are vectorized.

	0000000000000a00 <_b35respool_03>:
	a00:	c5 fa 10 4c 24 28    	vmovss 0x28(%rsp),%xmm1
	a06:	4c 8b 54 24 30       	mov    0x30(%rsp),%r10
	a0b:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
	a10:	c5 fa 10 01          	vmovss (%rcx),%xmm0
	a14:	c5 fa 58 02          	vaddss (%rdx),%xmm0,%xmm0
	a18:	c4 c1 7a 58 00       	vaddss (%r8),%xmm0,%xmm0
	a1d:	c4 c1 7a 58 01       	vaddss (%r9),%xmm0,%xmm0
	a22:	c4 c2 71 a9 02       	vfmadd213ss (%r10),%xmm1,%xmm0
	a27:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
	a2b:	c5 fa 5f c2          	vmaxss %xmm2,%xmm0,%xmm0
	a2f:	c5 fa 11 00          	vmovss %xmm0,(%rax)
	a33:	c5 fa 10 41 04       	vmovss 0x4(%rcx),%xmm0
	a38:	c5 fa 58 42 04       	vaddss 0x4(%rdx),%xmm0,%xmm0
	a3d:	c4 c1 7a 58 40 04    	vaddss 0x4(%r8),%xmm0,%xmm0
	a43:	c4 c1 7a 58 41 04    	vaddss 0x4(%r9),%xmm0,%xmm0
	a49:	c4 c2 71 a9 42 04    	vfmadd213ss 0x4(%r10),%xmm1,%xmm0
	a4f:	c5 fa 5f c2          	vmaxss %xmm2,%xmm0,%xmm0
	a53:	c5 fa 11 40 04       	vmovss %xmm0,0x4(%rax)
	a58:	c5 fa 10 42 08       	vmovss 0x8(%rdx),%xmm0
	a5d:	c5 fa 58 41 08       	vaddss 0x8(%rcx),%xmm0,%xmm0
	a62:	c4 c1 7a 58 40 08    	vaddss 0x8(%r8),%xmm0,%xmm0
	a68:	c4 c1 7a 58 41 08    	vaddss 0x8(%r9),%xmm0,%xmm0
	a6e:	c4 c2 71 a9 42 08    	vfmadd213ss 0x8(%r10),%xmm1,%xmm0
	a74:	c5 fa 5f c2          	vmaxss %xmm2,%xmm0,%xmm0
	a78:	c5 fa 11 40 08       	vmovss %xmm0,0x8(%rax)
	a7d:	c3                   	retq   
	a7e:	66 90                	xchg   %ax,%ax

# performance

running on i-7 8750H 2.2G with one thread, compiler gcc (MinGW-W64) 8.1.0. mobilenet-v2 input "224 x 224 x 3".
	
	$ ./mobilenet_v2.exe
	================ SIMD ================
	SIMD: MobileNetV2    80.919 ms per pic.



