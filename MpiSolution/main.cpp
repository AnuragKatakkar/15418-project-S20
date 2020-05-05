#include <fstream>
#include <ctime>
#include <iomanip>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <mpi.h>

#include "CycleTimer.h"
#include "mpiMatMul.h"

using namespace std;

string generateImageFilename()
{
    return NET_FOLDER + "sparse-images-" + std::to_string(NUM_NEURONS) + ".tsv";
}

string generateTruthFilename()
{
    return NET_FOLDER + "neuron" + std::to_string(NUM_NEURONS) + "-l" + std::to_string(FILE_NUM_LAYERS) + "-categories.tsv";
}

int main(int argc, char **argv) {
    MPI_Init(&argc,&argv);

    // Get rank, will be needed later.
    int rank;
    int num_processors;
    int img_idx_start = -1;
    int img_idx_end = -1;
    int image_chunk_size = -1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);

    // Determine what subset of images this thread will process. 
    if (num_processors >= NUM_IMAGES && rank < NUM_IMAGES)
    {
        img_idx_start = rank;
        img_idx_end = rank + 1;
        image_chunk_size = 1;
    }
    else
    {
        image_chunk_size = (NUM_IMAGES + (NUM_IMAGES % num_processors)) / num_processors;
        img_idx_start = image_chunk_size * rank;
        img_idx_end = image_chunk_size * (rank + 1);

        if (img_idx_end > NUM_IMAGES)
        {
            img_idx_end = NUM_IMAGES;
        }
    }

    if (rank == MPI_MASTER)
    {
        cout << "Running inference for " << NUM_IMAGES << " images with CUDA matrix multiplication on " << NUM_LAYERS << " layers, each size " << NUM_NEURONS << endl;
        cout << "Loading data..." << flush;
    }

    // We won't use GPU for this part, because it's not timed. 
    string images_file = generateImageFilename();
    string truth_file = generateTruthFilename();


    // Initialize the array that will hold our results.
    float *images = new float[image_chunk_size * NUM_NEURONS];

    // Initialize all values in images to 0.
    // images is going to store our image.
    for (int i = 0; i < image_chunk_size * NUM_NEURONS; i++) {
        images[i] = 0;
    }

    // Read images file into images.
    ifstream ifs_images(images_file);
    if (ifs_images.fail()) {
        cout << "\nError opening image file" << endl;
        return 0;
    }

    string line;
    while (getline(ifs_images, line)) {
        stringstream ss(line);
        int row = 0;
        int col = 0;
        float value = 0;
        int i = 0;
        string tmp;
        while (getline(ss, tmp, '\t')) {
            if (i == 0) {
                row = (int)atoi(tmp.c_str());
            } else if (i == 1) {
                col = (int)atoi(tmp.c_str());
            } else {
                stringstream buffer;
                buffer << tmp;
                buffer >> value;
            }
            i++;
        }

        if (col >=  0 && row > img_idx_start && row <= img_idx_end) {
            images[(row - img_idx_start - 1) * NUM_NEURONS + (col - 1)] = value;
        }
    }


    // Initialize the array holding our truth categories for all images.
    bool *truth_categories = new bool[image_chunk_size];
    for (int i = 0; i < image_chunk_size; i++) {
        truth_categories[i] = false;
    }

    // Read truth categories file into truth_categories.
    ifstream ifs_truth(truth_file);
    if (ifs_truth.fail()) {
        cout << "\nError opening truth categories file" << endl;
        return 0;
    }

    while (getline(ifs_truth, line)) {
        stringstream ss(line);
        int image_idx = 0;
        string tmp;
        while (getline(ss, tmp, '\t')) {
            image_idx = (int)atoi(tmp.c_str());
        }

        if (image_idx > img_idx_start && image_idx <= img_idx_end) {
            truth_categories[image_idx - img_idx_start - 1] = true;
        }
    }

    // Initialize the array that will hold layer weights.
    float **weights = new float *[NUM_LAYERS];
    for (int i = 0; i < NUM_LAYERS; i++) {
        weights[i] = new float[NUM_NEURONS * NUM_NEURONS];
        for (int j = 0; j < NUM_NEURONS * NUM_NEURONS; j++) {
            weights[i][j] = 0;
        }
    }

    // Read in weight matrices for all layers.
    for (int layer_idx = 1; layer_idx <= NUM_LAYERS; layer_idx++) {
        string layer_file = NET_FOLDER + "n" + std::to_string(NUM_NEURONS) + "-l" + std::to_string(layer_idx) + ".tsv";

        ifstream ifs(layer_file);
        if (ifs.fail()) {
            cout << "\nError opening weight file for layer " << layer_idx << endl;
            return 0;
        }

        while (getline(ifs, line)) {
            stringstream ss(line);
            int row = 0;
            int col = 0;
            float value = 0;
            int i = 0;
            string tmp;
            while (getline(ss, tmp, '\t')) {
                if (i == 0) {
                    row = (int)atoi(tmp.c_str());
                } else if (i == 1) {
                    col = (int)atoi(tmp.c_str());
                } else {
                    stringstream buffer;
                    buffer << tmp;
                    buffer >> value;
                }
                i++;
            }

            // We transpose weight matrices so that multiplication will be easier.
            weights[layer_idx - 1][(col - 1) * NUM_NEURONS + (row - 1)] = value;
        }
    }

    bool *truth_results = new bool[image_chunk_size];

    if (rank == MPI_MASTER)
    {
        cout << "DONE" << endl;
    }

    float accuracy = 0;
    float accuracy_sum = 0;

    if (img_idx_start < img_idx_end)
    {
        sparseMatMulCuda(images, weights, truth_results, img_idx_start, img_idx_end);

        for (int i = 0; i < image_chunk_size; i++) {
            if (truth_results[i] == truth_categories[i]) {
                accuracy += 1;
            }
        }
    }

    // Sum all accuracy counts. 
    MPI_Reduce(&accuracy, &accuracy_sum, 1, MPI_FLOAT, MPI_SUM, MPI_MASTER, MPI_COMM_WORLD);

    // MPI_Barrier(MPI_COMM_WORLD);
    if (rank == MPI_MASTER)
    {
        cout << "Accuracy : " << (accuracy_sum / NUM_IMAGES) * 100 << "%\n";
    }

    MPI_Finalize();
    return 0;
}