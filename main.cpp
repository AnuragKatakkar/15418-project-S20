//First attempt at reading in a single weight matrix and 
//the images file
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

#define NUM_NEURONS 1024
#define NUM_IMAGES 60000

void simplemutMulCuda(short *img, float  wts[][1024], float * res);

int main() {

    char * weight_fname = "./n1024-l1.tsv";
    char * images_file = "./sparse-images-1024.tsv";

    float weights[NUM_NEURONS][NUM_NEURONS];
    // for (int i = 0; i < NUM_NEURONS ; i++){
    //     weights[i] = new float[NUM_NEURONS];
    // }
    for (int i = 0; i < NUM_NEURONS ; i++){
        for (int j = 0; j < NUM_NEURONS; j++) {
            weights[i][j] = 0.0;
        }
    }
    short *images = new short[NUM_NEURONS];
    for (int j = 0; j < NUM_NEURONS; j++) {
        images[j] = 0;
    }

    ifstream ifs(weight_fname);
	if (ifs.fail()) {
		cout << "error" << endl;
		return 0;
	} else {
        cout << "Successfully opened file\n";
    }

    string line;
    while (getline(ifs, line)) {
        stringstream ss(line);
        int row;
        int col;
        float value;
        int i = 0;
        string tmp;
        while (getline(ss, tmp, '\t')) {
            if (i == 0) {
                row = (int)atoi(tmp.c_str());
            } else if (i == 1) {
                col = (int)atoi(tmp.c_str());
            } else {
                // cout << tmp <<endl;
                stringstream buffer;
                buffer << tmp;
                buffer >> value;
                // value = (float)atof(tmp.c_str());
            }
            i++;
        }
        weights[row - 1][col - 1] = value;
    }

    //Read Images File
    ifstream ifs_images(images_file);
	if (ifs_images.fail()) {
		cout << "error" << endl;
		return 0;
	} else {
        cout << "Successfully opened file\n";
    }

    while (getline(ifs_images, line)) {
        stringstream ss(line);
        int row;
        int col;
        // float value;
        int i = 0;
        string tmp;
        while (getline(ss, tmp, '\t')) {
            if (i == 0) {
                row = (int)atoi(tmp.c_str());
            } else if (i == 1) {
                col = (int)atoi(tmp.c_str());
            } else {
                // cout << tmp <<endl;
                // stringstream buffer;
                // buffer << tmp;
                // buffer >> value;
                // value = (float)atof(tmp.c_str());
            }
            i++;
        }
        if (row == 1){
            images[col - 1] = 1;
        } else if (row == 2){
            //Read in second row
        }
    }

    float *results = new float[NUM_NEURONS];
    simplemutMulCuda(images, weights, results);

    for (int j = 0; j < NUM_NEURONS ; j++) {
        cout<<results[j] << "\t";
    }
    cout << endl;

    return 0;
}
