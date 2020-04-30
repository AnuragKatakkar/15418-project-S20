#include <stdlib.h>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include "serialMatMul.h"

using namespace std;
SparseMatrix::SparseMatrix()
{
    data = new vector<matrix_entry>();
}

void SparseMatrix::insert(int row, int col, float val)
{
    data->push_back(make_tuple(row, col, val));
}

void SparseMatrix::sort()
{
    std::sort(data->begin(), data->end());
}
