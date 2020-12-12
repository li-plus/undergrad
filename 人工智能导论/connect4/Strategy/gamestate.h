#pragma once
#include <iostream>
#include <cassert>
#include <algorithm>
#include "Judge.h"
#include "print.h"
#include "utils.h"
#include "gameconfig.h"
using namespace std;

/**
* the state is created by _side
* the opposite side is about to move
*/

class GameState
{
public:
    GameState() = default;
    GameState(GameConfig::GameSide side, int moveX, int moveY) : _side(side), _moveX(moveX), _moveY(moveY)
    {
        selectState();
        initTerminal(GameConfig::top);

        if (_isTerminal)
            return;

        for (int col = 0; col < GameConfig::N; col++)
        {
            if (GameConfig::top[col] > 0)
                _validChoiceSize++;
        }
    }
    bool isSelected()
    {
        return !GameConfig::isValidMove(_moveX, _moveY) || GameConfig::board[_moveX][_moveY] == _side;
    }
    /*
    * add chess to the board
    */
    void selectState()
    {
        if (!GameConfig::isValidMove(_moveX, _moveY))
            return;

        GameConfig::board[_moveX][_moveY] = _side;
        GameConfig::top[_moveY] = _moveX;

        if (_moveY == GameConfig::noY && _moveX - 1 == GameConfig::noX)
            GameConfig::top[_moveY]--;
    }
    /*
    * remove chess from the board
    */
    void unselectState()
    {
        if (_moveX < 0 || _moveY < 0 || GameConfig::M <= _moveX || GameConfig::N <= _moveY)
            return;

        GameConfig::board[_moveX][_moveY] = GameConfig::NONE;
    }

    GameState nextRandomStateForRollOut()
    {
        int col = randInt(0, GameConfig::N);

        while (GameConfig::top[col] == 0)
            col = (col + 1) % GameConfig::N;

        return GameState(GameConfig::oppositeSide(_side), GameConfig::top[col] - 1, col);
    }
    bool isTerminal() const
    {
        return _isTerminal;
    }
    GameConfig::GameSide winner() const
    {
        assert(_isTerminal);
        return _winner;
    }
    void initTerminal(const int* const top)
    {
        if (_moveX < 0 || _moveY < 0 || GameConfig::M <= _moveX || GameConfig::N <= _moveY)
            return;

        switch (_side)
        {
            case GameConfig::MACHINE:
                if (machineWin(_moveX, _moveY, GameConfig::M, GameConfig::N, GameConfig::board))
                {
                    _winner = GameConfig::MACHINE;
                    _isTerminal = true;
                    return;
                }

                break;

            case GameConfig::USER:
                if(userWin(_moveX, _moveY, GameConfig::M, GameConfig::N, GameConfig::board))
                {
                    _winner = GameConfig::USER;
                    _isTerminal = true;
                    return;
                }

                break;

            default:
                throw std::runtime_error("invalid side");
        }

        if (isTie(GameConfig::N, &top[0]))
        {
            _winner = GameConfig::NONE;
            _isTerminal = true;
            return;
        }
    }

    bool winStep(const int* const top, int & y)
    {
        for (int col = 0; col < GameConfig::N; col++)
        {
            if (top[col] == 0)
                continue;

            GameConfig::board[top[col] - 1][col] = GameConfig::oppositeSide(_side);

            if (_side == GameConfig::USER && machineWin(top[col] - 1, col, GameConfig::M, GameConfig::N, GameConfig::board)
                    || _side == GameConfig::MACHINE && userWin(top[col] - 1, col, GameConfig::M, GameConfig::N, GameConfig::board))
            {
                y = col;
                GameConfig::board[top[col] - 1][col] = GameConfig::NONE;
                return true;
            }

            GameConfig::board[top[col] - 1][col] = GameConfig::NONE;
        }

        return false;
    }
    bool stopEnemyWinning(const int* const top, int lastX, int lastY, int& y)
    {
        if (lastX < 0 || GameConfig::M <= lastX || lastY < 0 || GameConfig::N <= lastY)
            return false;

        int leftmost = std::max(0, lastY - 3);
        int rightmost = std::min(GameConfig::N, lastY + 4);

        for (int col = leftmost; col < rightmost; col++)
        {
            if (top[col] == 0)
                continue;

            GameConfig::board[top[col] - 1][col] = _side;

            if ((_side == GameConfig::USER && userWin(top[col] - 1, col, GameConfig::M, GameConfig::N, GameConfig::board))
                    || (_side == GameConfig::MACHINE && machineWin(top[col] - 1, col, GameConfig::M, GameConfig::N, GameConfig::board)))
            {
                y = col;
                GameConfig::board[top[col] - 1][col] = GameConfig::NONE;
                return true;
            }

            GameConfig::board[top[col] - 1][col] = GameConfig::NONE;
        }

        return false;
    }
    friend ostream& operator<< (ostream & out, const GameState & state)
    {
        cout << "board" << endl;
        print(GameConfig::board, GameConfig::M, GameConfig::N);
        return cout << "move(" << state._moveX << ',' << state._moveY << ')';
    }

public:
    int _moveX;
    int _moveY;
    GameConfig::GameSide _side;
    int _validChoiceSize = 0;
    bool _isTerminal = false;
    GameConfig::GameSide _winner;
};
