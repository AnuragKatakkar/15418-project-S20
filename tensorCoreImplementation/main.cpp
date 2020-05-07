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

#define NUM_NEURONS 4096
#define NUM_IMAGES 60000
#define NUM_LAYERS 64

//  void simplemutMulCuda(float *img, float **wts, float * res, short *truth);
void tensorMatMul(float *img, float **wts, float * res, short *truth);
int main() {

    short * truth = new short[NUM_IMAGES];

    string images_file = "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/akatakka+inf/akatakkaData/neuron4096/sparse-images-4096.tsv";

    float **weights = new float* [NUM_LAYERS];
    for (int i = 0; i < NUM_LAYERS ; i++){
        weights[i] = new float[NUM_NEURONS * NUM_NEURONS];
    }
    for (int k = 0 ; k < NUM_LAYERS ; k ++ ) {
        for (int i = 0; i < NUM_NEURONS * NUM_NEURONS ; i++){
            weights[k][i] = 0.0;
        }
    }
    float *images = new float[NUM_NEURONS * NUM_IMAGES];
    for (int j = 0; j < NUM_NEURONS * NUM_IMAGES; j++) {
        images[j] = 0;
    }


    string line;
    for (int layer = 0; layer < NUM_LAYERS ; layer ++ ){
        string weight_fname = "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/akatakka+inf/akatakkaData/neuron4096/n4096-l" + to_string(layer + 1) + ".tsv";
        cout << "Reading weights : "<< weight_fname <<"\n";
        
        ifstream ifs(weight_fname);
        if (ifs.fail()) {
            cout << "error" << endl;
            return 0;
        } else {
            cout << "Successfully opened file\n";
        }

        while (getline(ifs, line)) {
            stringstream ss(line);
            int row = 0;
            int col = 0;
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
            // cout << i;
            // cout << row;
            // cout << col;
            // cout << "Here\n";
            weights[layer][(col - 1) * NUM_NEURONS + row - 1] = value;
        }
    }

    // for (int k = 0 ; k < NUM_LAYERS ; k ++ ) {
    //     for (int i = 0; i < NUM_NEURONS * NUM_NEURONS ; i++){
    //             cout << weights[k][i] << "\t";
    //     }
    //     cout<<endl;
    // }

    //Read Images File
    ifstream ifs_images(images_file);
	if (ifs_images.fail()) {
		cout << "error" << endl;
		return 0;
	} else {
        cout << "Successfully opened file\n";
    }

    cout << "Reading Images\n";
    while (getline(ifs_images, line)) {
        stringstream ss(line);
        int row = 0;
        int col = 0;
        // float value;
        int i = 0;
        string tmp;
        while (getline(ss, tmp, '\t')) {
            if (i == 0) {
                row = (int)atoi(tmp.c_str());
            } else if (i == 1) {
                col = (int)atoi(tmp.c_str());
            }
            i++;
        }
        //Control how many images to read in
        if (row <= NUM_IMAGES) {
            // cout << row <<endl;
            // cout << col <<endl;
            images[((col - 1)*NUM_IMAGES) +  row - 1] = 1;
        }
        
    }

    float *results = new float[NUM_IMAGES * NUM_NEURONS];

    // simplemutMulCuda(images, weights, results, truth);
    tensorMatMul(images, weights, results, truth);

    // for (int i = 0 ; i < NUM_IMAGES ; i ++) {
    //     // if (i != 0){
    //     //     continue;
    //     // }
    //     cout<<"Image Number : "<< i <<endl;
    //     for (int j = 0; j < NUM_NEURONS ; j++) {
    //         cout<<results[i + NUM_IMAGES * j] << "\t";
    //         // if (results[i + NUM_NEURONS * j] != 0) {
    //         //     cout << "Ohhhh\n";
    //         // }
    //     }
    //     cout << endl;
    // }

    unsigned short * ground_truth = new unsigned short[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES ; i++) {
        ground_truth[i] = 0;
    }

    string ground_truth_file = "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/akatakka+inf/akatakkaData/neuron4096/neuron4096-l" + to_string(NUM_LAYERS) + "-categories.tsv";
    ifstream ifs_gt(ground_truth_file);
	if (ifs_gt.fail()) {
		cout << "error" << endl;
		return 0;
	} else {
        cout << "Successfully opened file\n";
    }

    cout << "Reading Ground Truth\n";
    while (getline(ifs_gt, line)) {
        stringstream ss(line);
        int img_number = 0;
        // float value;
        int i = 0;
        string tmp;
        while (getline(ss, tmp, '\t')) {
            if (i == 0) {
                img_number = (int)atoi(tmp.c_str());
            }
            i++;
        }
        //Control how many images to read in
        if (img_number <= NUM_IMAGES)
            ground_truth[img_number - 1] = 1;
        
    }


    cout<<"Calculating Accuracy\n";
    float accuracy = 0;
    for (int i = 0 ; i < NUM_IMAGES; i++){
        if (truth[i] == ground_truth[i]){
            accuracy += 1;
        }
    }
    cout << "Acc : "<< accuracy << endl;

    cout << "Accuracy : "<< (accuracy/NUM_IMAGES) * 100 <<"%\n";

    for (int i = 0; i < NUM_LAYERS ; i++){
        delete weights[i];
    }
    delete weights;

    return 0;
}
