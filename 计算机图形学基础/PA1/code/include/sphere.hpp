#ifndef SPHERE_H
#define SPHERE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

// TODO: Implement functions and add more fields as necessary

class Sphere : public Object3D {
public:
    Sphere() {
        // unit ball at the center
        _center = Vector3f::ZERO;
        _radius = 1;
    }

    Sphere(const Vector3f &center, float radius, Material *material) : Object3D(material) {
        _center = center;
        _radius = radius;
    }

    ~Sphere() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        float dirNorm = r.getDirection().length();
        auto unitDirect = r.getDirection().normalized();
        float r2 = _radius * _radius;
        Vector3f rayToCenter = _center - r.getOrigin();
        float dist = rayToCenter.length();
        if (dist == _radius) {
            // intersection is on sphere, ignore
            return false;
        }
        auto tp = Vector3f::dot(rayToCenter, unitDirect);
        if (dist > _radius && tp < 0) {
            return false;
        }
        float d2 = dist * dist - tp * tp;
        if (d2 > r2) {
            return false;
        }
        auto delta_t = sqrt(r2 - d2);
        float t;
        if (dist < _radius) {
            // inside the sphere
            t = tp + delta_t;
        } else {
            // outside the sphere
            t = tp - delta_t;
        }
        t /= dirNorm;
        if (t < tmin) {
            return false;
        }
        auto n = (r.pointAtParameter(t) - _center).normalized();
        h.set(t, material, n);
        return true;
    }

protected:
    Vector3f _center;
    float _radius;
};


#endif
