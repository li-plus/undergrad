#pragma once

#include <iostream>
#include <cassert>
using namespace std;


static inline void print(int ** board, int M, int N)
{
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            cout << board[row][col] << ' ';
        }

        cout << endl;
    }
}

static inline void print(const int * v, int n)
{
    for (int i = 0; i < n; i++)
        cout << v[i] << ' ';

    cout << endl;
}

template<typename T>
static inline int** new2D(int M, int N)
{
    T** mat = new T*[M];
    mat[0] = new T[M * N];

    for (int i = 1; i < M; i++)
        mat[i] = mat[i - 1] + N;

    return mat;
}

template <typename T>
static inline void delete2D(T ** mat)
{
    delete[] mat[0];
    delete[] mat;
}

template<typename T>
static inline void memmove2D(T ** dst, const T ** src, int M, int N)
{
    memmove(*dst, *src, sizeof(T) * M * N);
}

static inline int randInt(int start, int end)
{
    assert(start < end);
    return start + (rand() % (end - start));
}