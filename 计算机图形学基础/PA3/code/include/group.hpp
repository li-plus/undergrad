#ifndef GROUP_H
#define GROUP_H


#include "object3d.hpp"
#include "ray.hpp"
#include "hit.hpp"
#include <iostream>
#include <vector>


// TODO (PA2): Implement Group - copy from PA1
class Group : public Object3D {

public:

    Group() {

    }

    explicit Group (int num_objects) {
        _objs.assign(num_objects, nullptr);
    }

    ~Group() override {
        for (auto &obj: _objs) {
            delete obj;
        }
    }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        return false;
    }

    void drawGL() override {
        for (auto &obj: _objs) {
            obj->drawGL();
        }
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
	
