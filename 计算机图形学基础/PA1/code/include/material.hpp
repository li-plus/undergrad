#ifndef MATERIAL_H
#define MATERIAL_H

#include <cassert>
#include <vecmath.h>

#include "ray.hpp"
#include "hit.hpp"
#include <iostream>

// TODO: Implement Shade function that computes Phong introduced in class.
class Material {
public:

    explicit Material(const Vector3f &d_color, const Vector3f &s_color = Vector3f::ZERO, float s = 0) :
            diffuseColor(d_color), specularColor(s_color), shininess(s) {

    }

    virtual ~Material() = default;

    virtual Vector3f getDiffuseColor() const {
        return diffuseColor;
    }

    static float relu(float x) {
        return x > 0 ? x : 0;
    }

    Vector3f Shade(const Ray &ray, const Hit &hit,
                   const Vector3f &dirToLight, const Vector3f &lightColor) {
        auto L = dirToLight.normalized();
        auto N = hit.getNormal().normalized();
        Vector3f reflectedDir = 2 * Vector3f::dot(N, L) * N - L;
        Vector3f shaded = (relu(Vector3f::dot(L, N)) * diffuseColor +
                           pow(relu(Vector3f::dot(-ray.getDirection(), reflectedDir)), shininess) * specularColor
                          ) * lightColor;
        return shaded;
    }

protected:
    Vector3f diffuseColor;
    Vector3f specularColor;
    float shininess;
};


#endif // MATERIAL_H
