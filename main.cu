#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>
#include "nvgraph.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "imagem.h"

using namespace std;

typedef std::pair<double, int> custo_caminho;

typedef std::pair<double *, int *> result_sssp;

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

using namespace std;


 __global__ void edge_filter(unsigned char *img, unsigned char *out, int rows, int cols)
 {
    int di,dj;
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i< rows && j< cols) {

        int min = 256;
        int max = 0;
        for(di = MAX(0, i - 1); di <= MIN(i + 1, rows - 1); di++) 
        {
            for(dj = MAX(0, j - 1); dj <= MIN(j + 1, cols - 1); dj++) 
            {
            if(min>img[di*cols+dj]) {
                min = img[di*cols+dj];
            }

            if(max<img[di*cols+dj]) { 
                max = img[di*cols+dj]; 
            }
            }
        }
        out[i*cols+j] = max-min;
    }
     
 }

/* Programa cria dois vetores e soma eles em GPU */
void blur(imagem *img) {
    
    thrust::host_vector<unsigned char> values_cpu(img->total_size);

    for (int i=0; i<values_cpu.size(); i++){
        values_cpu[i] = (int)img->pixels[i];
    }

    thrust::device_vector<unsigned char> values_gpu (values_cpu);

    thrust::device_vector<unsigned char> out_gpu(values_cpu);
  
    dim3 dimGrid(ceil(img->rows/16.0), ceil(img->cols/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    // add_one<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(values_gpu.data()), thrust::raw_pointer_cast(out_gpu.data()), img->rows, img->cols);

    edge_filter<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(values_gpu.data()), thrust::raw_pointer_cast(out_gpu.data()), img->rows, img->cols);

    thrust::host_vector<double> new_img (out_gpu);

    for (int i=0; i<new_img.size(); i++){
        img->pixels[i] = new_img[i];
        // cout << new_img[i] << ' ';
    }
   
}


struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};

void check(nvgraphStatus_t status) {
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

float *SSSP(imagem *img, vector<int> seeds) {
    std::priority_queue<custo_caminho, std::vector<custo_caminho>, compare_custo_caminho > Q;
    double *custos = new double[img->total_size];
    int *predecessor = new int[img->total_size];
    bool *analisado = new bool[img->total_size];


    // nvgraph setup
    const size_t  n = img->total_size, vertex_numsets = 1, edge_numsets = 1;
    float *sssp_1_h;
    void** vertex_dim;


    // nvgraph variables
    nvgraphStatus_t status; nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;
    vector<float> weights_h;
    vector<int> destination_offsets_h;
    vector<int> source_indices_h;
    int offset = 0;

    for (int i = 0; i < img->total_size; i++) {
        predecessor[i] =-1;
        custos[i] = __DBL_MAX__;
        analisado[i] = false;
    }


    for (int i=0; i<img->rows; i++){
        for (int j=0; j<img->cols; j++){
            destination_offsets_h.push_back(offset);

            int vertex = j + i * img->cols;

            
            if (find(begin(seeds), end(seeds), vertex) != end(seeds)) {
                source_indices_h.push_back(img->total_size);
                weights_h.push_back(0.0);
                offset++;
            }

            if (i > 0) {
                int acima = vertex - img->cols;
                double custo_acima = get_edge(img, vertex, acima);
                source_indices_h.push_back(acima);
                weights_h.push_back(custo_acima);
                offset++;
            }

            if (i < img->rows - 1) {
                int abaixo = vertex + img->cols;
                double custo_abaixo = get_edge(img, vertex, abaixo);
                source_indices_h.push_back(abaixo);
                weights_h.push_back(custo_abaixo);
                offset++;
            }


            if (j < img->cols - 1) {
                int direita = vertex + 1;
                double custo_direita = get_edge(img, vertex, direita);
                source_indices_h.push_back(direita);
                weights_h.push_back(custo_direita);
                offset++;
            }

            if (j > 0) {
                int esquerda = vertex - 1;
                double custo_esquerda = get_edge(img, vertex, esquerda);
                source_indices_h.push_back(esquerda);
                weights_h.push_back(custo_esquerda);
                offset++;
            }
        }
    }

    const int nnz = source_indices_h.size();
    destination_offsets_h.push_back(offset);


    // Init host data
    sssp_1_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_1_h; vertex_dimT[0] = CUDA_R_32F;
    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr (handle, &graph));
    CSC_input->nvertices = img->total_size; CSC_input->nedges = nnz;
    CSC_input->destination_offsets = &destination_offsets_h[0];
    CSC_input->source_indices = &source_indices_h[0];

    // Set graph connectivity and properties (tranfers)
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void*)&weights_h[0], 0));

    // Solve
    check(nvgraphSssp(handle, graph, 0,  &img->total_size, 0));

    // Get and print result
    check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));


    //Clean 
    free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));


    
    
    return sssp_1_h;
}


int main(int argc, char **argv) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time=0.0, seg_img_time=0.0, sssp_time=0.0, graph_time=0.0;

    if (argc < 3) {
        cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }

    std::string path(argv[1]);
    std::string path_output(argv[2]);
    imagem *img = read_pgm(path);

    
    int n_fg, n_bg;
    vector<int> seeds_fg, seeds_bg;
    int x, y;
    
    std::cin >> n_fg >> n_bg;

    for (int k = 0; k < n_fg; k++) {
        std::cin >> x >> y;
        int seed_fg = y * img->cols + x;
        seeds_fg.push_back(seed_fg);
    }
     
    for (int k = 0; k < n_bg; k++) {  
        std::cin >> x >> y;
        int seed_bg = y * img->cols + x;
        seeds_bg.push_back(seed_bg);
    }

    
    cudaEventRecord(start);
    cout << "detecting edges..." << endl;
    blur(img);


    cout << "calculating SSSP for fg seeds..." << endl;
    float *fg = SSSP(img, seeds_fg);

    cout << "calculating SSSP for bg seeds..." << endl;
    float *bg = SSSP(img, seeds_bg);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&graph_time, start, stop); 
    
    
    imagem *saida = new_image(img->rows, img->cols);
    
    cout << "creating new image..." << endl;

    cudaEventRecord(start);

    for (int k = 0; k < saida->total_size; k++) {
        if (fg[k] > bg[k]) {
            saida->pixels[k] = 0;
        } else {
            saida->pixels[k] = 255;
        }
    }

 
    write_pgm(saida, path_output);   
       
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&seg_img_time, start, stop); 


    printf("Total time: %fs\n", graph_time/1000+seg_img_time/1000);
    printf("Graph + Solution time: %fs\n", graph_time/1000);
    printf("Image creation time: %fs\n", seg_img_time/1000);


    return 0;
}