#include "workerevolve.h"
#include "core/simulatorapi.h"

#define FOR(i,s,e) for(int i=s;i<e;i++)
#define ROF(i,s,e) for(int i=s;i>e;i--)

#include <iostream>


WorkerEvolve::WorkerEvolve(QObject *parent) : QThread (parent)
{
    isStop = false;
}
void WorkerEvolve::init(const vector<double> &targetFlow, int n, const vector<int> &inputCol, const vector<int> &outputCol)
{
    this->targetFlow = targetFlow;
    this->n = n;
    this->inputCol = inputCol;
    this->outputCol = outputCol;
}
void WorkerEvolve::run()
{
    int round = 20;
    int GENE_LENGTH = (2 * n*n - 2 * n) * 7;
    int VALID_GENE = GENE_LENGTH;
    int DIRECTLY_COPY_NUM = 3;
    int SELECTED_NUM = 20;
    const int POPULATION_NUM = 100;
    double THRESHOLD = 3;
    vector<vector<bool>> population(POPULATION_NUM);
    for (int i = 0; i < POPULATION_NUM; i++)
    {
        population[i] = initPredictCode(VALID_GENE);
    }
    FOR(rnd, 0, round)
    {
        if(isStop) return;
        // calculate loss
        vector<double> losses(POPULATION_NUM);
        FOR(j, 0, POPULATION_NUM)
        {
            if(isStop) return;
            double loss;
            vector<double> lengths = decodeLengths(population[j]);
            lengths.insert(lengths.end(), 5, 1);

            vector<double> predictFlow = caluconspeed(n, lengths,
                                                      inputCol[0], inputCol[1], outputCol[0], outputCol[1], outputCol[2]);
            //print(predictFlow);
            if (isNANorINFincluded(predictFlow))
            {
                loss = 100000000;
            }
            else
            {
                loss = calcLoss(targetFlow, predictFlow);
            }
            losses[j] = loss;
        }

        // pick the top 15 out to generate the new generation
        vector<vector<bool>> seleted(SELECTED_NUM);
        vector<size_t> idx = sort_indexes(losses);

        FOR(i, 0, SELECTED_NUM)
        {
            seleted[i] = population[idx[i]];
            std::cout << (losses[idx[i]]) << ' ';
        }
        if (losses[idx[0]] < THRESHOLD)
        {
            vector<double> lengths = decodeLengths(population[idx[0]]);
            lengths.insert(lengths.end(), 5, 1);
            isStop = true;
            emit lengthsFound(lengths);
        }
        FOR(i, 0, DIRECTLY_COPY_NUM)
        {
            population[i] = seleted[i];
            if (!(rand() % 3)) {
                FOR(times, 0, 5)
                {
                    int variationPos = rand() % (VALID_GENE);
                    population[i][variationPos] = !population[i][variationPos];
                }
            }
        }
        FOR(i, DIRECTLY_COPY_NUM, POPULATION_NUM)
        {
            vector<bool> mother = seleted[rand() % SELECTED_NUM];
            vector<bool> father = seleted[rand() % SELECTED_NUM];
            int split = rand() % (VALID_GENE);
            FOR(j, 0, population[i].size())
            {
                population[i][j] = rand() % 2 ? mother[j] : father[j];
            }
            if (!(rand() % 3)) {
                FOR(times, 0, 10)
                {
                    int variationPos = rand() % (VALID_GENE);
                    population[i][variationPos] = !population[i][variationPos];
                }
            }
        }
    }
    vector<double> lengths = decodeLengths(population[0]);
    lengths.insert(lengths.end(), 5, 1);

    emit lengthsFound(lengths);
}
void WorkerEvolve::stopImmediately()
{
    isStop = true;
}
