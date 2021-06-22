#ifndef CAMERA_H
#define CAMERA_H

#include "ray.hpp"
#include <vecmath.h>
#include <float.h>
#include <cmath>


class Camera {
public:
    Camera(const Vector3f &center, const Vector3f &direction, const Vector3f &up, int imgW, int imgH) {
        this->center = center;
        this->direction = direction.normalized();
        this->horizontal = Vector3f::cross(this->direction, up);
        this->horizontal.normalize();
        this->up = Vector3f::cross(this->horizontal, this->direction);
        this->width = imgW;
        this->height = imgH;
    }

    // Generate rays for each screen-space coordinate
    virtual Ray generateRay(const Vector2f &point) = 0;
    virtual ~Camera() = default;

    int getWidth() const { return width; }
    int getHeight() const { return height; }

protected:
    // Extrinsic parameters
    Vector3f center;
    Vector3f direction;
    Vector3f up;
    Vector3f horizontal;
    // Intrinsic parameters
    int width;
    int height;
};

// TODO: Implement Perspective camera
// You can add new functions or variables whenever needed.
class PerspectiveCamera : public Camera {
    float _focal_len;
public:
    PerspectiveCamera(const Vector3f &center, const Vector3f &direction, const Vector3f &up,
                      int imgW, int imgH, float angle) : Camera(center, direction, up, imgW, imgH) {
        // angle is in radian.
        _focal_len = imgH / 2. / tan(angle / 2.);
    }

    Ray generateRay(const Vector2f &point) override {
        Vector3f dirCam(point.x() - width / 2, point.y() - height / 2, _focal_len);
        Matrix3f camRotation(horizontal, up, direction);
        Vector3f dirWorld = (camRotation * dirCam).normalized();
        Ray ray(center, dirWorld);
        return ray;
    }
};

#endif //CAMERA_H
