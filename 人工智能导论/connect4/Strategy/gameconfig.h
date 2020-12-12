#pragma once



namespace GameConfig
{

enum GameSide { NONE = 0, USER = 1, MACHINE = 2 };
static int noX = -1;
static int noY = -1;
static int M = 0;
static int N = 0;
static int lastX = -1;
static int lastY = -1;
static int** board = nullptr;
static int** boardBackup = nullptr;
static GameSide mySide = MACHINE;
static GameSide enemySide = USER;
static int top[16];
static const int * topBackup = nullptr;
static inline GameSide oppositeSide(GameSide side)
{
    return (side == USER) ? MACHINE : USER;
}

static inline bool isValidMove(int moveX, int moveY)
{
    return (0 <= moveX && moveX < M && 0 <= moveY && moveY < N);
}
static inline bool isNewGame()
{
    int cnt = 0;
    for (int row = M - 2; row < M; row++)
        for (int col = 0; col < N; col++)
            cnt += board[row][col];

    if (cnt == 1 || cnt == 0)
        return true;

    return false;
}
}