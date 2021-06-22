#ifndef MESH_H
#define MESH_H

#include <vector>
#include "object3d.hpp"
#include "triangle.hpp"
#include "Vector2f.h"
#include "Vector3f.h"


class Mesh : public Object3D {

public:
    Mesh(const char *filename, Material *m);

    struct TriangleIndex {
        TriangleIndex() {
            x[0] = 0; x[1] = 0; x[2] = 0;
        }
        int &operator[](const int i) { return x[i]; }
        // By Computer Graphics convention, counterclockwise winding is front face
        int x[3]{};
    };

    std::vector<Vector3f> v;
    std::vector<TriangleIndex> t;
    std::vector<Vector3f> n;
    bool intersect(const Ray &r, Hit &h, float tmin) override;

    void drawGL() override {
        // TODO (PA2): Call drawGL for each individual triangle.
        for (size_t i = 0; i < n.size(); i++) {
            Object3D::drawGL();
            glBegin(GL_TRIANGLES);
            Vector3f norm = n[i];
            glNormal3fv(norm);
            auto tri = t[i].x;
            glVertex3fv(v[tri[0]]);
            glVertex3fv(v[tri[1]]);
            glVertex3fv(v[tri[2]]);
            glEnd();
        }
    }

private:

    // Normal can be used for light estimation
    void computeNormal();
};

#endif
