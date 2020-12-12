#include "macro.h"
#include <iostream>
#include "Point.h"
#include "Strategy.h"
#include "mcts.h"
#include "gamestate.h"
#include "gameconfig.h"
#include "utils.h"


using namespace std;

/*
	���Ժ����ӿ�,�ú������Կ�ƽ̨����,ÿ�δ��뵱ǰ״̬,Ҫ�����������ӵ�,�����ӵ������һ��������Ϸ��������ӵ�,��Ȼ�Կ�ƽ̨��ֱ����Ϊ��ĳ�������

	input:
		Ϊ�˷�ֹ�ԶԿ�ƽ̨ά����������ɸ��ģ����д���Ĳ�����Ϊconst����
		M, N : ���̴�С M - ���� N - ���� ����0��ʼ�ƣ� ���Ͻ�Ϊ����ԭ�㣬����x��ǣ�����y���
		top : ��ǰ����ÿһ���ж���ʵ��λ��. e.g. ��i��Ϊ��,��_top[i] == M, ��i������,��_top[i] == 0
		_board : ���̵�һά�����ʾ, Ϊ�˷���ʹ�ã��ڸú����տ�ʼ���������Ѿ�����ת��Ϊ�˶�ά����board
				��ֻ��ֱ��ʹ��board���ɣ����Ͻ�Ϊ����ԭ�㣬�����[0][0]��ʼ��(����[1][1])
				board[x][y]��ʾ��x�С���y�еĵ�(��0��ʼ��)
				board[x][y] == 0/1/2 �ֱ��Ӧ(x,y)�� ������/���û�����/�г������,�������ӵ㴦��ֵҲΪ0
		lastX, lastY : �Է���һ�����ӵ�λ��, ����ܲ���Ҫ�ò�����Ҳ������Ҫ�Ĳ������ǶԷ�һ����
				����λ�ã���ʱ��������Լ��ĳ����м�¼�Է������ಽ������λ�ã�����ȫȡ�������Լ��Ĳ���
		noX, noY : �����ϵĲ������ӵ�(ע:��ʵ���������top�Ѿ����㴦���˲������ӵ㣬Ҳ����˵���ĳһ��
				������ӵ�����ǡ�ǲ������ӵ㣬��ôUI�����еĴ�����Ѿ������е�topֵ�ֽ�����һ�μ�һ������
				��������Ĵ�����Ҳ���Ը�����ʹ��noX��noY��������������ȫ��Ϊtop������ǵ�ǰÿ�еĶ�������,
				��Ȼ�������ʹ��lastX,lastY�������п��ܾ�Ҫͬʱ����noX��noY��)
		���ϲ���ʵ���ϰ����˵�ǰ״̬(M N _top _board)�Լ���ʷ��Ϣ(lastX lastY),��Ҫ���ľ�������Щ��Ϣ�¸������������ǵ����ӵ�
	output:
		������ӵ�Point
*/


/*
	my side			(machine)	2
	opposite side	(user)		1
*/
extern "C" __declspec(dllexport) Point* getPoint(const int M, const int N, const int* top, const int* _board,
        const int lastX, const int lastY, const int noX, const int noY)
{
    clock_t clkStart = clock();
    srand((unsigned)time(nullptr));
    // init param
    GameConfig::M = M;
    GameConfig::N = N;
    GameConfig::noX = noX;
    GameConfig::noY = noY;
    GameConfig::lastX = lastX;
    GameConfig::lastY = lastY;
    // init board
    GameConfig::board = new2D<int>(M, N);
    GameConfig::boardBackup = new2D<int>(M, N);
    memmove2D(GameConfig::board, &_board, M, N);
    memmove2D(GameConfig::boardBackup, &_board, M, N);
    // init top
    GameConfig::topBackup = top;
    memmove(GameConfig::top, top, sizeof(int) * N);
    // init state
    GameState initState(GameConfig::enemySide, lastX, lastY);
#ifdef _DEBUG
    cout << "board" << endl;
    print(GameConfig::board, M, N);
    cout << "top " << endl;
    print(top, N);
    cout << "last move(" << lastX << ',' << lastY << ')' << endl;
    cout << "init state\n" << initState << endl;
#endif // _DEBUG

    MCTS::NodePool::clear();
	MCTS::Node * initNode = MCTS::NodePool::newNode(initState);
    MCTS::Node * nextState = MCTS::bestNextStep(initNode, clkStart);
#ifdef _DEBUG
    cout << "\nnext state \n" << *nextState << endl;
#endif // _DEBUG
    int x = nextState->_state._moveX;
    int y = nextState->_state._moveY;

    if (top[y] - 1 != x)
    {
        cout << "error move(" << x << ',' << y << ')' << endl;
    }

    delete2D(GameConfig::board);
    delete2D(GameConfig::boardBackup);
#ifdef _DEBUG
    cout << "time spent " << clock() - clkStart << endl;
#endif // _DEBUG
    return new Point(x, y);
}


/*
	getPoint�������ص�Pointָ�����ڱ�dllģ���������ģ�Ϊ��������Ѵ���Ӧ���ⲿ���ñ�dll�е�
	�������ͷſռ䣬����Ӧ�����ⲿֱ��delete
*/
extern "C" __declspec(dllexport) void clearPoint(Point* p)
{
    delete p;
    return;
}

