#ifndef GROUP_H
#define GROUP_H


#include "object3d.hpp"
#include "ray.hpp"
#include "hit.hpp"
#include <iostream>
#include <vector>


// TODO: Implement Group - add data structure to store a list of Object*
class Group : public Object3D {

public:

    Group() {

    }

    explicit Group(int num_objects) {
        _objs.assign(num_objects, nullptr);
    }

    ~Group() override {
        for (auto &obj: _objs) {
            delete obj;
        }
    }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        bool hasIntersect = false;
        for (auto &obj: _objs) {
            Hit hit;
            auto ret = obj->intersect(r, hit, tmin);
            hasIntersect |= ret;
            if (ret && hit.getT() < h.getT()) {
                h = hit;
            }
        }
        return hasIntersect;
    }

    void addObject(int index, Object3D *obj) {
        _objs[index] = obj;
    }

    int getGroupSize() {
        return _objs.size();
    }

private:
    std::vector<Object3D *> _objs;
};

#endif

