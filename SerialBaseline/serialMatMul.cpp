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

void singleValueMult(int row, int col, float **image, float **weights, float **results) {
    results[row][col] = 0;

    for (int i = 0; i < NUM_NEURONS; i++) {
        results[row][col] += image[row][i] * weights[i][col];
    }

    results[row][col] += BIAS;

    if (results[row][col] < RELU_MIN) {
        results[row][col] = RELU_MIN;
    } else if (results[row][col] > RELU_MAX) {
        results[row][col] = RELU_MAX;
    }
}

void serialMatMul(float **prev, float **weights, float **results) {
    for (int input_idx = 0; input_idx < NUM_IMAGES; input_idx++) {
        for (int neuron_idx = 0; neuron_idx < NUM_NEURONS; neuron_idx++) {
            singleValueMult(input_idx, neuron_idx, prev, weights, results);
        }
    }
}

void serialInference(float **results_A, float **results_B, float ***layers, bool *truth_results) {
    // Start inference timer.
    bool old_results_is_A = true;
    cout << "Starting inference..." << flush;
    clock_t inference_start = clock();

    // Begin inference, going layer by layer.
    for (int layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++) {
        if (old_results_is_A) {
            serialMatMul(results_A, layers[layer_idx], results_B);
        } else {
            serialMatMul(results_B, layers[layer_idx], results_A);
        }

        // Flip so that we begin copying into the other results array.
        old_results_is_A = !old_results_is_A;
    }

    // Inference finished, stop timer.
    double inference_duration =
            (clock() - inference_start) / (double)CLOCKS_PER_SEC;
    cout << "DONE\nDetermining truth categories..." << flush;

    // Start truth category timer.
    clock_t truth_start = clock();

    float **final_results = old_results_is_A ? results_A : results_B;
    for (int i = 0; i < NUM_IMAGES; i++) {
        truth_results[i] = false;
        for (int j = 0; j < NUM_NEURONS; j++) {
            if (final_results[i][j] > 0)
            {
                truth_results[i] = true;
                break;
            }
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

int runDenseMatMul()
{
    cout << "Running inference for " << NUM_IMAGES << " images with dense matrix multiplication on " << NUM_LAYERS << " layers, each size " << NUM_NEURONS << endl;
    string images_file = generateImageFilename();
    string truth_file = generateTruthFilename();

    cout << "Loading data..." << flush;

    // Initialize the arrays that will hold our results.
    // We need two in order to run multiple layers.
    float **results_A = new float *[NUM_IMAGES];
    float **results_B = new float *[NUM_IMAGES];

    // Initialize all values in results_A to 0.
    // results_A is going to store our image.
    for (int i = 0; i < NUM_IMAGES; i++) {
        results_A[i] = new float[NUM_NEURONS];
        results_B[i] = new float[NUM_NEURONS];
        for (int j = 0; j < NUM_NEURONS; j++) {
            results_A[i][j] = 0;
        }
    }

    // Read images file into results_A.
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
            results_A[row - 1][col - 1] = value;
        }
    }

    // Initialize the array holding our truth categories for all images.
    bool *truth_categories = new bool[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; i++) {
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

        if (image_idx <= NUM_IMAGES) {
            truth_categories[image_idx - 1] = true;
        }
    }

    // Initialize the array that will hold layer weights.
    float ***weights = new float **[NUM_LAYERS];
    for (int i = 0; i < NUM_LAYERS; i++) {
        weights[i] = new float *[NUM_NEURONS];
        for (int j = 0; j < NUM_NEURONS; j++) {
            weights[i][j] = new float[NUM_NEURONS];
            for (int k = 0; k < NUM_NEURONS; k++) {
                weights[i][j][k] = 0;
            }
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

            weights[layer_idx - 1][row - 1][col - 1] = value;
        }
    }

    cout << "DONE" << endl;

    bool *result_categories = new bool[NUM_IMAGES];
    serialInference(results_A, results_B, weights, result_categories);

    float accuracy = 0;
    for (int i = 0; i < NUM_IMAGES; i++) {
        if (result_categories[i] == truth_categories[i]) {
            accuracy += 1;
        }
        else
        {
            cout << "Incorrect for " << i << endl;
        }
    }

    cout << "Accuracy : " << (accuracy / NUM_IMAGES) * 100 << "%\n";
    return 0;
}