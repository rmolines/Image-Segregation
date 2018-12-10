#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>
#include <chrono>

#include "imagem.h"

typedef std::pair<double, int> custo_caminho;

typedef std::pair<double *, int *> result_sssp;

typedef std::chrono::high_resolution_clock Time;

using namespace std;

struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};

result_sssp SSSP(imagem *img, vector <int> seeds, double &sssp_time, double &graph_time) {
    Time::time_point t1, t2;

    std::priority_queue<custo_caminho, std::vector<custo_caminho>, compare_custo_caminho > Q;
    double *custos = new double[img->total_size];
    int *predecessor = new int[img->total_size];
    bool *analisado = new bool[img->total_size];

    t1 = Time::now();
    result_sssp res(custos, predecessor);
    
    for (int i = 0; i < img->total_size; i++) {
        predecessor[i] =-1;
        custos[i] = __DBL_MAX__;
        analisado[i] = false;
    };

    for (int i = 0; i< seeds.size(); i++){
        Q.push(custo_caminho(0.0, seeds[i]));
        predecessor[seeds[i]] = seeds[i];
        custos[seeds[i]] = 0.0;
    }

    while (!Q.empty()) {
        custo_caminho cm = Q.top();
        Q.pop();

        int vertex = cm.second;
        if (analisado[vertex]) continue; // já tem custo mínimo calculado
        analisado[vertex] = true;
        double custo_atual = cm.first;
        assert(custo_atual == custos[vertex]);

        int acima = vertex - img->cols;
        if (acima >= 0) {
            double custo_acima = custo_atual + get_edge(img, vertex, acima);
            if (custo_acima < custos[acima]) {
                custos[acima] = custo_acima;
                Q.push(custo_caminho(custo_acima, acima));
                predecessor[acima] = vertex;
            }
        }

        int abaixo = vertex + img->cols;
        if (abaixo < img->total_size) {
            double custo_abaixo = custo_atual + get_edge(img, vertex, abaixo);
            if (custo_abaixo < custos[abaixo]) {
                custos[abaixo] = custo_abaixo;
                Q.push(custo_caminho(custo_abaixo, abaixo));
                predecessor[abaixo] = vertex;
            }
        }


        int direita = vertex + 1;
        if (direita < img->total_size) {
            double custo_direita = custo_atual + get_edge(img, vertex, direita);
            if (custo_direita < custos[direita]) {
                custos[direita] = custo_direita;
                Q.push(custo_caminho(custo_direita, direita));
                predecessor[direita] = vertex;
            }
        }

        int esquerda = vertex - 1;
        if (esquerda >= 0) {
            double custo_esquerda = custo_atual + get_edge(img, vertex, esquerda);
            if (custo_esquerda < custos[esquerda]) {
                custos[esquerda] = custo_esquerda;
                Q.push(custo_caminho(custo_esquerda, esquerda));
                predecessor[esquerda] = vertex;
            }
        }
    }

    t2 = Time::now();
    sssp_time += std::chrono::duration_cast<std::chrono::duration<double>> (t2-t1).count();
    
    delete[] analisado;
    
    return res;
}


int main(int argc, char **argv) { 
    if (argc < 3) {
        std::cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }
    std::string path(argv[1]);
    std::string path_output(argv[2]);
    imagem *img = read_pgm(path);

    Time::time_point t1, t2;
    double sssp_time, seg_img_time;
    
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
    
    cout << "calculating SSSP for fg seeds..." << endl;
    result_sssp fg = SSSP(img, seeds_fg, sssp_time, seg_img_time);

    cout << "calculating SSSP for bg seeds..." << endl;
    result_sssp bg = SSSP(img, seeds_bg, sssp_time, seg_img_time);
    
    cout << "creating new image..." << endl;
    t1 = Time::now();
    imagem *saida = new_image(img->rows, img->cols);
    
    for (int k = 0; k < saida->total_size; k++) {
        if (fg.first[k] > bg.first[k]) {
            saida->pixels[k] = 0;
        } else {
            saida->pixels[k] = 255;
        }
    }
    
    write_pgm(saida, path_output);    
    t2 = Time::now();
    seg_img_time = std::chrono::duration_cast<std::chrono::duration<double>> (t2-t1).count();

    cout << "Total time: " << sssp_time+seg_img_time << "s" << endl;
    cout << "Solution time: " << sssp_time << "s" << endl;
    cout << "Image creation time: " << seg_img_time << "s" << endl;

    return 0;
}
