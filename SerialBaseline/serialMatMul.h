#include <string>
#include <vector>
#include <tuple>

#define RELU_MAX 32
#define RELU_MIN 0
#define TIMING_PRECISION 20

#define NUM_NEURONS 4096
#define NUM_LAYERS 120
#define BIAS -0.35f
#define TEST_ACCURACY false
#define IS_SPARSE true
#define NUM_IMAGES 60000

const std::string NET_FOLDER = "/afs/cs.cmu.edu/academic/class/15418-s20/public/projects/akatakka+inf/akatakkaData/neuron4096/";

typedef std::tuple<int, int, float> matrix_entry;

class SparseMatrix{
public:
    std::vector<matrix_entry>* data;
    SparseMatrix();
    void insert(int row, int col, float val);
    void sort();
};

std::string generateImageFilename();
std::string generateTruthFilename();

int runDenseMatMul();
int runSparseMatMul();
