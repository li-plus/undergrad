#pragma once

#include <iostream>
#include <vector>
#include <algorithm>

/**
 * @brief Stack class organizes elements as a FILO queue.
 * @tparam T Data type
 * @details
 * The std::stack is inconvenient, so a custom one is implemented.
 * It provides all methods of a standard stack. What is more?
 * You can access the element on top while popping it out.
 * Elements inside the stack could be accessed.
 * Standard output method is implemented.
 */
template<typename T>
class Stack : public std::vector<T>
{
public:
    Stack() = default;
    /**
     * @brief Push in an element at back
     * @param val A value
     */
    void push(const T &val)
    { std::vector<T>::emplace_back(val); }
    /**
     * @brief Pop out the top element and return it.
     * @return The popped element.
     */
    T pop()
    {
        auto backup = std::vector<T>::back();
        std::vector<T>::pop_back();
        return backup;
    }
    /**
     * @brief Get the element on top.
     * @return The element on top.
     */
    T &top()
    { return std::vector<T>::back(); }
    /**
     * @brief Standard output function
     * @param out Output stream
     * @param stk The stack to output
     * @return Output stream
     */
    friend std::ostream &operator<<(std::ostream &out, const Stack<T> &stk)
    {
        out << "Stack( ";
        for (auto &e: stk)
            out << e << ' ';
        return out << ')';
    }
};
