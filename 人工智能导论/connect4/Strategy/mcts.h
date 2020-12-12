#pragma once

#include <iostream>
#include <ctime>
#include <string>
#include <climits>
#include <cmath>
#include "print.h"
#include "gamestate.h"
using namespace std;


#define MAX_NODE_NUM		5000000

#ifdef _DEBUG
#define MIN_RESERVE_SIZE	MAX_NODE_NUM - 100
#else
#define MIN_RESERVE_SIZE	1000000
#endif

#define MAX_TIME_SPENT		2500
#define EXPAND_C			0.70710678
#define MAX_ITER_TIMES		0x00FFFFFF

namespace MCTS
{
class Node
{
public:
    Node()
    {
        _children.reserve(GameConfig::N);
    }
    Node(const GameState & state) : _state(state)
    {
        _children.reserve(GameConfig::N);
    }

    bool isAllExpand()
    {
        return _children.size() == _state._validChoiceSize;
    }

    friend ostream & operator<<(ostream & out, const Node & node)
    {
        return out << "state:" << node._state << " wins:" << node._wins << " visitTimes:" << node._visits;
    }

#ifdef _DEBUG
    void printAll()
    {
        int curRow = 0;
        vector<pair<Node *, int> > q;
        q.emplace_back(this, curRow);

        while (!q.empty())
        {
            auto p = q.front();
            q.erase(q.begin());
            Node * curNode = p.first;

            if (curRow < p.second)
            {
                cout << endl;
                curRow++;
            }

            for (auto & child : curNode->_children)
            {
                q.emplace_back(child, p.second + 1);
            }
        }
    }
#endif // _DEBUG

public:
    vector<Node*> _children;
    Node * _parent = nullptr;
    double _visits = 0;
    double _wins = 0;
    GameState _state;
};

namespace NodePool
{
static Node nodePool[MAX_NODE_NUM + 150];
static int size = 0;
static void clear()
{
    size = 0;
}
template <typename ...Args>
static Node* newNode(Args ...args)
{
    nodePool[size] = Node(args...);
    return &nodePool[size++];
}
}

static Node * selectedNode = nullptr;

static GameState nextState(Node * node)
{
    int nextY = randInt(0, GameConfig::N);

    while (true)
    {
        if (std::find_if(node->_children.begin(), node->_children.end(), [ = ](const Node * n)
		{
			return n->_state._moveY == nextY;
		}) == node->_children.end() && GameConfig::top[nextY] > 0) // not found
        {
            break;
        }
        nextY = (nextY + 1) % GameConfig::N;
    }

    return GameState(GameConfig::oppositeSide(node->_state._side), GameConfig::top[nextY] - 1, nextY);
}

static Node * expandChild(Node * node)
{
#ifdef _DEBUG
    cout << "----------------expand-------------------" << endl;
#endif // _DEBUG
    GameState newState = nextState(node);
    Node* newChild = NodePool::newNode(newState);
    node->_children.emplace_back(newChild);
    newChild->_parent = node;
#ifdef _DEBUG
    cout << "expand:: next state" << endl;
    cout << newChild->_state << endl;
#endif // _DEBUG
    return newChild;
}
/*
* selection & expansion
* select the child with max ucb score.
* expand the untapped child first.
*/
Node * selectBestChild(Node * node, bool isExploration)
{
    Node * bestSubNode = nullptr;
    double c = (isExploration ? EXPAND_C : 0);
    double maxScore = INT_MIN;

    for (auto &child : node->_children)	// choose the node with max UCB score
    {
        // UCB = w/n + c * sqrt(2ln(N)/n)
        double score = (double)child->_wins / child->_visits + c * sqrt(2 * log(node->_visits) / child->_visits);

        if (score > maxScore)
        {
            maxScore = score;
            bestSubNode = child;
        }
    }

    bestSubNode->_state.selectState();
    return bestSubNode;
}
static Node * treePolicy(Node * node)
{
#ifdef _DEBUG
    cout << "----------------tree policy-------------------" << endl;
#endif // _DEBUG

    while (!node->_state.isTerminal())
    {
#ifdef _DEBUG
        cout << "selecting child" << endl;
#endif // _DEBUG

        if (node->isAllExpand())
            node = selectBestChild(node, true);
        else
            return expandChild(node);
    }
    return node;
}
/*
* simulation
* randomly select next state of game till the game is over.
*/
static GameConfig::GameSide defaultPolicy(Node * node)
{
#ifdef _DEBUG
    cout << "----------------default policy-------------------" << endl;
#endif // _DEBUG
    GameState curState = node->_state;

    while (!curState.isTerminal())
    {
        curState = curState.nextRandomStateForRollOut();
#ifdef _DEBUG
        cout << "default policy:: next random state" << endl;
        cout << curState << endl;
#endif // _DEBUG
    }

    return curState.winner();
}
/*
* back propagation.
* update all its ancestors' visit times and win times.
*/
static void backup(Node * node, GameConfig::GameSide winner)
{
#ifdef _DEBUG
    cout << "----------------backup-------------------" << endl;
#endif // _DEBUG

    while (node)
    {
        node->_visits++;

        if (winner != GameConfig::NONE)
        {
            if (winner == node->_state._side)
                node->_wins++;
            else
                node->_wins--;
        }

#ifdef _DEBUG
        cout << "backup:: state" << endl;
        cout << node->_state << endl;
        cout << node->_wins << '/' << node->_visits << endl;
#endif // _DEBUG
        node = node->_parent;
    }
}
static Node * newRoot(int lastX, int lastY)
{
    if (!selectedNode)
        return nullptr;

    for (auto & child : selectedNode->_children)
    {
        if(child->_state._moveX == lastX && child->_state._moveY == lastY)
            return child;
    }

    return nullptr;
}
static Node * bestNextStep(Node * s, const clock_t &clkStart)
{
#ifdef _DEBUG
    cout << "-------------------- best next step --------------------" << endl;
#endif // _DEBUG
    int i = 0;

    for (; i < MAX_ITER_TIMES; i++)
    {
        if (clock() - clkStart > MAX_TIME_SPENT || NodePool::size >= MAX_NODE_NUM)
            break;

        // init next iteration
        memmove2D(GameConfig::board, (const int**)GameConfig::boardBackup, GameConfig::M, GameConfig::N);
        memmove(GameConfig::top, GameConfig::topBackup, sizeof(int) * GameConfig::N);
        // general MCTS algorithm
        Node * expandNode = treePolicy(s);
        auto winner = defaultPolicy(expandNode);
        backup(expandNode, winner);
    }

    selectedNode = selectBestChild(s, false);
#ifdef _DEBUG
    cout << "expand num: " << i << "  node usage: " << NodePool::size << endl;
#endif // _DEBUG
    return selectedNode;
}
};


