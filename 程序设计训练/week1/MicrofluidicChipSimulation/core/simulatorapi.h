
#ifndef SIMULATORAPI
#define SIMULATORAPI


#include <vector>
using namespace std;


vector<double> caluconspeed(int num, const vector<double>&length, int i1, int i2, int o1, int o2, int o3);

vector<double> evolve(const vector<double> &targetFlow, int n, const vector<int> &inputCol, const vector<int> &outputCol);

double decodeSingleLength(const vector<bool> &code);

vector<bool> encodeSingleLength(double length);

vector<double> decodeLengths(const vector<bool> &code);

double calcLoss(const vector<double> &targetFlow, const vector<double> &predictFlow);

template<typename T>
void print(vector<vector<T>> &vec);

template<typename T>
void print(const vector<T> &vec);

template < typename T>
vector<size_t> sort_indexes(const vector<T>  & v);

vector<bool> initPredictCode(int size);

bool isNANorINFincluded(const vector<double> &vec);

#endif // SIMULATORAPI
