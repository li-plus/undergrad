#include "variant.h"
#include <cmath>

/**
 * @brief Calculate abs(v).
 * @param v
 * @return Variant of the same type of v.
 */
static inline Variant abs(const Variant &v)
{
    Variant result = std::abs(v.convertTo(Variant::DOUBLE).toDouble());
    return result.convertTo(v.type());
}

/**
 * @brief Calculate exp(v).
 * @param v
 * @return Variant of double type by default.
 */
static inline Variant exp(const Variant &v)
{
    return std::exp(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate floor(v).
 * @param v
 * @return Variant of int type by default.
 */
static inline Variant floor(const Variant &v)
{
    return (int) std::floor(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate ceil(v).
 * @param v
 * @return Variant of int type by default.
 */
static inline Variant ceil(const Variant &v)
{
    return (int) std::ceil(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate ln(v).
 * @param v
 * @return Variant of double type by default.
 */
static inline Variant ln(const Variant &v)
{
    return std::log(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate log10(v).
 * @param v
 * @return Variant of double type by default.
 */
static inline Variant log10(const Variant &v)
{
    return std::log10(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate sin(v).
 * @param v
 * @return Variant of double type by default.
 */
static inline Variant sin(const Variant &v)
{
    return std::sin(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate cos(v)
 * @param v
 * @return Variant of double type by default.
 */
static inline Variant cos(const Variant &v)
{
    return std::cos(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate tan(v).
 * @param v
 * @return Variant of double type by default.
 */
static inline Variant tan(const Variant &v)
{
    return std::tan(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate asin(v).
 * @param v
 * @return Variant of double type by default.
 */
static inline Variant asin(const Variant &v)
{
    return std::asin(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate acos(v).
 * @param v
 * @return Variant of double type by default.
 */
static inline Variant acos(const Variant &v)
{
    return std::acos(v.convertTo(Variant::DOUBLE).toDouble());
}

/**
 * @brief Calculate atan(v).
 * @param v
 * @return Variant of double type by default.
 */
static inline Variant atan(const Variant &v)
{
    return std::atan(v.convertTo(Variant::DOUBLE).toDouble());
}

