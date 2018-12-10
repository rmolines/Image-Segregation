all: gpu seq

gpu: main.cu imagem.cpp imagem.h
	nvcc -std=c++11 -lnvgraph main.cu imagem.cpp -o gpu 


seq: main.cpp imagem.cpp imagem.h
	nvcc -std=c++11 main.cpp imagem.cpp -o seq 