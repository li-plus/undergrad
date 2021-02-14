#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <regex>
#include <algorithm>
#include "datastream.h"

/**
 * @brief This class provide methods for date operation.
 */
class Date
{
public:
    /**
     * @brief Default trivial constructor.
     */
    Date() = default;

    /**
     * @brief Construct date with year, month and day.
     * @param year
     * @param month
     * @param day
     */
    Date(int year, int month, int day) : _year(year), _month(month), _day(day)
    {}

    /**
     * @brief Construct a date with a string of "YYYY-MM-DD" format.
     * @param str Input string.
     */
    Date(const std::string &str)
    {
        std::smatch sm;
        if (!regex_match(str, sm, std::regex(R"((\d{4})-(\d{2})-(\d{2}))")))
            return;
        _year = std::stoi(sm[1].str());
        _month = std::stoi(sm[2].str());
        _day = std::stoi(sm[3].str());
        if (_year < 0 || _year > 9999 || _month < 1 || 12 < _month || _day < 1 || 31 < _day)
            _year = _month = _day = 0;
    }

    /**
     * @brief Determine whether two dates equal.
     * @param other Another date.
     * @return Whether two dates equal.
     */
    bool operator==(const Date &other) const
    { return value() == other.value(); }

    /**
     * @brief Determine whether this is less than another
     * @param other Another date.
     * @return Whether this is less than another
     */
    bool operator<(const Date &other) const
    { return value() < other.value(); }

    /**
     * @brief Determine whether this is greater than another
     * @param other Another date
     * @return Whether this is greater than another
     */
    bool operator>(const Date &other) const
    { return value() > other.value(); }

    /**
     * @brief Add date by the given interval days.
     * @param days The interval days.
     * @return The operation result.
     */
    Date operator+(int days) const
    { return Date(_year, _month, _day + days).carry(); }

    /**
     * @brief Standard output function
     * @param out The output stream
     * @param date The date to output
     * @return The output stream
     */
    friend std::ostream &operator<<(std::ostream &out, const Date &date)
    {
        return out << std::setw(4) << std::setfill('0') << date._year << '-'
                   << std::setw(2) << std::setfill('0') << date._month << '-'
                   << std::setw(2) << std::setfill('0') << date._day;
    }

    /**
     * @brief Binary output function
     * @param out Binary output stream
     * @param date The date to output
     * @return Binary output stream
     */
    friend DataStream &operator<<(DataStream &out, const Date &date)
    { return out << date._year << date._month << date._day; }

    /**
     * @brief Binary input stream
     * @param in Binary input stream.
     * @param date The destination date.
     * @return Binary input stream.
     */
    friend DataStream &operator>>(DataStream &in, Date &date)
    { return in >> date._year >> date._month >> date._day; }

protected:
    /**
     * @brief Get the int value for comparison between two dates.
     * @return The int value.
     * @details
     * A date is less than / equal to / greater than another date if and only if
     * its value is less than / equal to / greater than that of another date
     */
    int value() const
    { return _year * 400 + _month * 40 + _day; }

    /**
     * @brief Solve days overflow.
     * @return The valid result.
     */
    Date &carry()
    {
        static std::vector<int> maxDays = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        while (_day > maxDays[_month])
        {
            _day -= maxDays[_month];
            if (_month++ > 12)
            {
                _month = 1;
                _year++;
            }
        }
        return *this;
    }

protected:
    int _year = 0;
    int _month = 0;
    int _day = 0;
};

/**
 * @brief This class provide methods for time operation.
 */
class Time
{
public:
    /**
     * @brief Default trivial constructor.
     */
    Time() = default;

    /**
     * @brief Construct time with hour, minute and second value.
     * @param hour
     * @param minute
     * @param second
     */
    Time(int hour, int minute, int second) : _hour(hour), _minute(minute), _second(second)
    { carry(); }

    /**
     * @brief Construct time with string of "HH-MM-SS" format.
     * @param str Input string.
     */
    Time(const std::string &str)
    {
        std::smatch sm;
        if (!regex_match(str, sm, std::regex(R"((\d{2}):(\d{2}):(\d{2}))")))
            return;
        _hour = std::stoi(sm[1].str());
        _minute = std::stoi(sm[2].str());
        _second = std::stoi(sm[3].str());
        carry();
    }

    /**
     * @brief Add time by specific interval seconds.
     * @param secs Interval seconds.
     * @return The addition result.
     */
    Time operator+(int secs) const
    { return {_hour, _minute, _second + secs}; }

    /**
     * @brief Determine whether this time is equal to another.
     * @param other Another Time instance.
     * @return Whether this time is equal to another.
     */
    bool operator==(const Time &other) const
    { return totalSeconds() == other.totalSeconds(); }

    /**
     * @brief Determine whether this time is less than another.
     * @param other Another time instance.
     * @return True iff this time is less than another.
     */
    bool operator<(const Time &other) const
    { return totalSeconds() < other.totalSeconds(); }

    /**
     * @brief Determine whether this time is greater than another.
     * @param other Another time instance.
     * @return True iff this time is greater than another.
     */
    bool operator>(const Time &other) const
    { return totalSeconds() > other.totalSeconds(); }

    /**
     * @brief Standard output method.
     * @param out Standard output stream
     * @param t A time to output
     * @return Standard output streamm.
     */
    friend std::ostream &operator<<(std::ostream &out, const Time &t)
    {
        return out << std::setw(2) << std::setfill('0') << t._hour << ':'
                   << std::setw(2) << std::setfill('0') << t._minute << ':'
                   << std::setw(2) << std::setfill('0') << t._second;
    }

    /**
     * @brief Binary output stream.
     * @param out Binary output stream.
     * @param t A time to output
     * @return Binary output stream.
     */
    friend DataStream &operator<<(DataStream &out, const Time &t)
    { return out << t._hour << t._minute << t._second; }

    /**
     * @brief Binary input stream.
     * @param in Binary input stream.
     * @param t The destination time.
     * @return Binary input stream.
     */
    friend DataStream &operator>>(DataStream &in, Time &t)
    { return in >> t._hour >> t._minute >> t._second; }

protected:
    /**
     * @brief Get the corresponding total seconds of the time.
     * @return The total seconds.
     */
    int totalSeconds() const
    { return _hour * 3600 + _minute * 60 + _second; }

    /**
     * @brief Solve overflow.
     * @return The result time.
     */
    Time &carry()
    {
        if (_second > 59)
        {
            _minute += _second / 60;
            _second %= 60;
        }
        if (_minute > 59)
        {
            _hour += _minute / 60;
            _minute %= 60;
        }
        _hour %= 24;
        return *this;
    }

protected:
    int _hour = 0;      ///< Hour value. Valid range is [0,24).
    int _minute = 0;    ///< Minute value. Valid range is [0,60).
    int _second = 0;    ///< Second value. Valid range is [0,60).
};
