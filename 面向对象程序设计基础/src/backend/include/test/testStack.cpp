#include <iostream>
#include "../stack.h"



int main()
{
    Stack<int>  stk;
    cout << "pushing"<<endl;
    for(int i=0;i<16;i++)
    {
        stk.push(i);
    }
    cout << stk<<endl;
    cout << "popping"<<endl;
    for(int i=0; i< 8;i++)
    {
        cout << stk.pop() << ' ';
    }
    cout << endl ;
    cout << stk <<endl;
    return 0;
}