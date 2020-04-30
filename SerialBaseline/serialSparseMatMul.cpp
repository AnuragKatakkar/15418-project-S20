#include <ctime>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include "serialMatMul.h"

using namespace std;

SparseMatrix sparseMatMul(SparseMatrix &matrix_data, SparseMatrix &matrix_weights)
{
    SparseMatrix results;
    vector<matrix_entry> data = *matrix_data.data; 
    vector<matrix_entry> weights = *matrix_weights.data;

    uint data_pos = 0;
    uint weight_pos = 0;
    while(data_pos < data.size())
    {
        int row = get<0>(data[data_pos]);
        weight_pos = 0;

        while(weight_pos < weights.size())
        {
            int col = get<0>(weights[weight_pos]);

            uint data_pos_temp = data_pos;
            uint weights_pos_temp = weight_pos;

            // We want to add the bias to our result.
            float sum = BIAS;

            while (data_pos_temp < data.size() && get<0>(data[data_pos_temp]) == row
                && weights_pos_temp < weights.size() && get<0>(weights[weights_pos_temp]) == col)
            {
                if (get<1>(data[data_pos_temp]) < get<1>(weights[weights_pos_temp]))
                {
                    data_pos_temp++;
                }
                else if (get<1>(data[data_pos_temp]) > get<1>(weights[weights_pos_temp]))
                {
                    weights_pos_temp++;
                }
                else
                {
                    sum += get<2>(data[data_pos_temp++]) * get<2>(weights[weights_pos_temp++]);
                }
            }

            if (sum > RELU_MIN)
            {
                if (sum < RELU_MAX)
                {
                    results.insert(row, col, sum);
                }
                else 
                {
                    results.insert(row, col, RELU_MAX);
                }
            }

            while(weight_pos < weights.size() && get<0>(weights[weight_pos]) == col)
            {
                weight_pos++;
            }
        }

        while(data_pos < data.size() && get<0>(data[data_pos]) == row)
        {
            data_pos++;
        }

    }

    return results;
}

void sparseSerialInference(SparseMatrix &matrix_data, SparseMatrix* layers, bool* truth_results)
{
    // Start inference timer.
    cout << "Starting inference..." << flush;
    clock_t inference_start = clock();

    // Begin inference, going layer by layer.
    for (int layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++) {
        matrix_data = sparseMatMul(matrix_data, layers[layer_idx]);
    }

    // Inference finished, stop timer.
    double inference_duration =
            (clock() - inference_start) / (double)CLOCKS_PER_SEC;
    cout << "DONE\nDetermining truth categories..." << flush;

    // Start truth category timer.
    clock_t truth_start = clock();

    for(vector<matrix_entry>::iterator it = matrix_data.data->begin(); it != matrix_data.data->end(); ++it)
    {
        if (get<2>(*it) > 0)
        {
            truth_results[get<0>(*it)] = true;
        }
    }

    // End truth category timer. 
    double truth_duration = (clock() - truth_start) / (double)CLOCKS_PER_SEC;
    cout << "DONE \n" << endl;

    // Print all duration info. 
    cout << "Inference Duration: " << setprecision(TIMING_PRECISION) << inference_duration << " s" << endl;
    cout << "Truth Determination Duration: " << setprecision(TIMING_PRECISION) << truth_duration << " s" << endl;
    cout << "Total time taken: " << setprecision(TIMING_PRECISION) << inference_duration + truth_duration << " s" << endl;
}

int runSparseMatMul()
{
    cout << "Running inference for " << NUM_IMAGES << " images with sparse matrix multiplication on " << NUM_LAYERS << " layers, each size " << NUM_NEURONS << endl;
    string images_file = generateImageFilename();
    string truth_file = generateTruthFilename();

    cout << "Loading data..." << flush;

    SparseMatrix image_matrix;

    // Read images file into image_matrix.
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

        if (row >= 0 && col >= 0 && row <= NUM_IMAGES) {
            image_matrix.insert(row - 1, col - 1, value);
        }
    }
    image_matrix.sort();

    // Initialize the array holding our truth categories for all images.
    bool *truth_categories = new bool[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; i++) {
        truth_categories[i] = false;
    }
    
    if (TEST_ACCURACY) {
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

            if (image_idx <= NUM_IMAGES) {
                truth_categories[image_idx - 1] = true;
            }
        }
    }

    // Initialize the matrix that will hold layer weights.
    SparseMatrix* weights = new SparseMatrix[NUM_LAYERS];

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
            
            // Note: weight matrices are transposed for easier multiplication.
            weights[layer_idx - 1].insert(col - 1, row - 1, value);
        }

        weights[layer_idx - 1].sort();
    }

    cout << "DONE" << endl;

    bool *result_categories = new bool[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        result_categories[i] = false;
    }
    sparseSerialInference(image_matrix, weights, result_categories);

    if (TEST_ACCURACY)
    {
        float accuracy = 0;
        for (int i = 0; i < NUM_IMAGES; i++) {
            if (result_categories[i] == truth_categories[i]) {
                accuracy += 1;
            }
            else
            {
                /// cout << "Incorrect for " << i << endl;
            }
        }
        cout << "Accuracy : " << (accuracy / NUM_IMAGES) * 100 << "%\n";
    }

    return 0;
}