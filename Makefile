
all: gpu

gpu: main.cu imagem.cpp imagem.h
	nvcc -std=c++11 -lnvgraph main.cu imagem.cpp -o gpu 
