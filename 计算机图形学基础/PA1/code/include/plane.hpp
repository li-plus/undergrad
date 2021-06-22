#ifndef PLANE_H
#define PLANE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

// TODO: Implement Plane representing an infinite plane
// function: ax+by+cz=d
// choose your representation , add more fields and fill in the functions

class Plane : public Object3D {
public:
    Plane() {
        _n = Vector3f(0, 0, 1);
        _d = 0;
    }

    Plane(const Vector3f &normal, float d, Material *m) : Object3D(m) {
        _n = normal.normalized();
        _d = d;
    }

    ~Plane() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        float dirNorm = r.getDirection().length();
        auto unitDirect = r.getDirection().normalized();
        float norm = Vector3f::dot(unitDirect, _n);
        if (norm == 0) { return false; }
        float t = (_d - Vector3f::dot(r.getOrigin(), _n)) / norm;
        t /= dirNorm;
        if (t < tmin) { return false; }
        h.set(t, material, _n);
        return true;
    }

protected:
    Vector3f _n;
    float _d;
};

#endif //PLANE_H
		

