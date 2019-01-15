from __future__ import print_function
import torch
import time

def benchmark_gpu(maxIter):
	print ("CUDA available", torch.cuda.is_available())
	device = torch.device("cuda")
	for i in range(1, maxIter):
		x = torch.rand(100, 100, device=device)
		y = torch.rand(100, 100, device=device)
		z = torch.mul(x, y)

def benchmark_cpu(maxIter):
	for i in range(1, maxIter):
		x = torch.rand(100, 100).to("cpu")
		y = torch.rand(100, 100).to("cpu")
		z = torch.mul(x, y)

def benchmark():
	iterations = [1, 10, 100, 1000, 10000, 100000, 1000000]
	for i in iterations:
		t0 = time.time()
		benchmark_cpu(i)
		t1 = time.time()
		benchmark_gpu(i)
		t2 = time.time()
		print('IterCount: {}'.format(i))
		print ('CPU: {}'.format(t1 - t0))
		print ('GPU: {}'.format(t2 - t1))

benchmark()
