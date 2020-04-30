#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "serialMatMul.h"

using namespace std;

string generateImageFilename()
{
    return NET_FOLDER + "sparse-images-" + std::to_string(NUM_NEURONS) + ".tsv";
}

string generateTruthFilename()
{
    return NET_FOLDER + "neuron" + std::to_string(NUM_NEURONS) + "-l" + std::to_string(NUM_LAYERS) + "-categories.tsv";
    //return "./categories/neuron" + std::to_string(NUM_NEURONS) + "-l" + std::to_string(NUM_LAYERS) + "-categories.tsv";
}

int main() {
    if (IS_SPARSE)
    {
        return runSparseMatMul();
    }

    return runDenseMatMul();
}