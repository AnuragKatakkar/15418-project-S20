#include <string>
#include <vector>
#include <tuple>

#define MPI_MASTER 0
#define RELU_MAX 32
#define RELU_MIN 0

#define NUM_NEURONS 1024
#define NUM_LAYERS 120
#define FILE_NUM_LAYERS 120
#define BIAS -0.3f
#define NUM_IMAGES 30000
#define THREADS_PER_BLOCK 1024 // Must evenly divide NUM_NEURONS.

const std::string NET_FOLDER = "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/akatakka+inf/akatakkaData/n1024/";

std::string generateImageFilename();
std::string generateTruthFilename();

double sparseMatMulCuda(float *images, int** weights_rows, float** weights_vals, bool *truth, int img_start_idx, int img_end_idx);
