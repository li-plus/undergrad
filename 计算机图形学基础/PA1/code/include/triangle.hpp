#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "plane.hpp"
#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>
#include <iostream>

using namespace std;

// TODO: implement this class and add more fields as necessary,
class Triangle : public Object3D {

public:
    Triangle() = delete;

    // a b c are three vertex positions of the triangle
    Triangle(const Vector3f &a, const Vector3f &b, const Vector3f &c, Material *m) : Object3D(m) {
        normal = Vector3f::cross(b - a, c - a).normalized();
        _a = a;
        _b = b;
        _c = c;
        float d = Vector3f::dot(normal, a);
        _plane = Plane(normal, d, material);
    }

    bool intersect(const Ray &ray, Hit &hit, float tmin) override {
        bool planeIntersect = _plane.intersect(ray, hit, tmin);
        if (!planeIntersect) { return false; }
        auto pos = ray.pointAtParameter(hit.getT());
        return (Vector3f::dot(Vector3f::cross(_b - _a, pos - _a), Vector3f::cross(_c - _a, pos - _a)) <= 0 &&
                Vector3f::dot(Vector3f::cross(_a - _b, pos - _b), Vector3f::cross(_c - _b, pos - _b)) <= 0);
    }

    Vector3f normal;
    Vector3f _a;
    Vector3f _b;
    Vector3f _c;
protected:
    Plane _plane;
};

#endif //TRIANGLE_H
