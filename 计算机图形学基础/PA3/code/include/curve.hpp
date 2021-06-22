#ifndef CURVE_HPP
#define CURVE_HPP

#include "object3d.hpp"
#include <vecmath.h>
#include <vector>
#include <utility>

#include <algorithm>

// TODO (PA3): Implement Bernstein class to compute spline basis function.
//       You may refer to the python-script for implementation.

// The CurvePoint object stores information about a point on a curve
// after it has been tesselated: the vertex (V) and the tangent (T)
// It is the responsiblility of functions that create these objects to fill in all the data.
struct CurvePoint {
    Vector3f V; // Vertex
    Vector3f T; // Tangent  (unit)
};

// De Boor's algorithm: https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
class DeBoor {
public:
    /**
     * Compute curve point at given position.
     * @param x Position.
     * @param t Array of knot positions, needs to be padded as described above.
     * @param c Array of control points.
     * @param p Degree of B-spline.
     * @return Curve point at position x.
     */
    static Vector3f evaluate(float x, const std::vector<float> &t, const std::vector<Vector3f> &c, int p) {
        int k = (x == t[0]) ? (int) (std::upper_bound(t.begin(), t.end(), x) - t.begin()) - 1
                            : (int) (std::lower_bound(t.begin(), t.end(), x) - t.begin()) - 1;
        std::vector<Vector3f> d(p + 1);
        for (int j = 0; j <= p; j++) {
            d[j] = c[j + k - p];
        }
        for (int r = 1; r <= p; r++) {
            for (int j = p; j >= r; j--) {
                float alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p]);
                d[j] = (1.f - alpha) * d[j - 1] + alpha * d[j];
            }
        }
        return d[p];
    }

    template<class T>
    static void makePadding(std::vector<T> &arr, int p) {
        arr.insert(arr.begin(), p, arr.front());
        arr.insert(arr.end(), p, arr.back());
    }
};

class Curve : public Object3D {
protected:
    std::vector<Vector3f> controls;
public:
    explicit Curve(std::vector<Vector3f> points) : controls(std::move(points)) {}

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        return false;
    }

    std::vector<Vector3f> &getControls() {
        return controls;
    }

    virtual void discretize(int resolution, std::vector<CurvePoint> &data) = 0;

    void drawGL() override {
        Object3D::drawGL();
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glDisable(GL_LIGHTING);
        glColor3f(1, 1, 0);
        glBegin(GL_LINE_STRIP);
        for (auto &control : controls) { glVertex3fv(control); }
        glEnd();
        glPointSize(4);
        glBegin(GL_POINTS);
        for (auto &control : controls) { glVertex3fv(control); }
        glEnd();
        std::vector<CurvePoint> sampledPoints;
        discretize(30, sampledPoints);
        glColor3f(1, 1, 1);
        glBegin(GL_LINE_STRIP);
        for (auto &cp : sampledPoints) { glVertex3fv(cp.V); }
        glEnd();
        glPopAttrib();
    }
};

class BezierCurve : public Curve {
public:
    explicit BezierCurve(const std::vector<Vector3f> &points) : Curve(points) {
        if (points.size() < 4 || points.size() % 3 != 1) {
            printf("Number of control points of BezierCurve must be 3n+1!\n");
            exit(0);
        }
    }

    void discretize(int resolution, std::vector<CurvePoint> &data) override {
        data.clear();
        // TODO (PA3): fill in data vector
        dp(resolution, data);
    }

protected:
    // brute force
    void bf(int resolution, std::vector<CurvePoint> &data) const {
        int N = (int) controls.size() - 1;
        std::vector<double> comb_n_1;
        std::vector<double> comb(N + 1, 1);
        for (int n = 2; n <= N; n++) {
            for (int i = n - 1; i > 0; i--) {
                comb[i] += comb[i - 1];
            }
            if (n == N - 1) {
                comb_n_1 = comb;
            }
        }
        for (int r = 0; r <= resolution; r++) {
            CurvePoint pos;
            float t = (float) r / (float) resolution;
            // compute value
            for (int i = 0; i <= N; i++) {
                pos.V += (float) (comb[i] * pow(t, i) * pow(1 - t, N - i)) * controls[i];
            }
            // compute derivative
            for (int i = 0; i < N; i++) {
                pos.T += (float) (comb_n_1[i] * pow(t, i) * pow(1 - t, N - 1 - i)) * (controls[i + 1] - controls[i]);
            }
            pos.T = (pos.T * (float) N).normalized();
            data.emplace_back(std::move(pos));
        }
    }

    void dp(int resolution, std::vector<CurvePoint> &data) const {
        // curve control points
        int n = (int) controls.size() - 1;
        int p = n;
        std::vector<Vector3f> c = controls;
        DeBoor::makePadding(c, p);
        auto t = makeKnots(n);

        // derivative control points
        std::vector<Vector3f> dc(n);
        for (int i = 0; i < n; i++) {
            dc[i] = (float) p * (controls[i + 1] - controls[i]);
        }
        DeBoor::makePadding(dc, p);
        std::vector<float> dt(t.begin() + 1, t.end() - 1);

        for (int r = 0; r <= resolution; r++) {
            CurvePoint pos;
            float x = (float) r / (float) resolution;
            pos.V = DeBoor::evaluate(x, t, c, p);
            pos.T = DeBoor::evaluate(x, dt, dc, p - 1).normalized();
            data.emplace_back(std::move(pos));
        }
    }

protected:
    static std::vector<float> makeKnots(int n) {
        std::vector<float> t(n + n + 2);
        for (int i = 0; i <= n; i++) {
            t[i] = 0;
        }
        for (int i = n + 1; i <= n + n + 1; i++) {
            t[i] = 1;
        }
        DeBoor::makePadding(t, n);
        return t;
    }
};

class BsplineCurve : public Curve {
public:
    BsplineCurve(const std::vector<Vector3f> &points) : Curve(points) {
        if (points.size() < 4) {
            printf("Number of control points of BspineCurve must be more than 4!\n");
            exit(0);
        }
    }

    void discretize(int resolution, std::vector<CurvePoint> &data) override {
        data.clear();
        // TODO (PA3): fill in data vector

        // curve control points
        int n = (int) controls.size() - 1;
        int p = 3;
        std::vector<Vector3f> c = controls;
        DeBoor::makePadding(c, p);
        auto t = makeKnots(n, p);

        // derivative control points
        std::vector<Vector3f> dc(n);
        for (int i = 0; i < n; i++) {
            dc[i] = (float) p / (t[i + p + 1] - t[i + 1]) * (controls[i + 1] - controls[i]);
        }
        DeBoor::makePadding(dc, p);
        std::vector<float> dt(t.begin() + 1, t.end() - 1);

        float x = t[p + p];
        float step = (t[p + n + 1] - t[p + p]) / (float) (resolution * (n + 1 - p));
        for (int r = 0; r <= resolution * (n + 1 - p); r++) {
            CurvePoint pos;
            pos.V = DeBoor::evaluate(x, t, c, p);
            pos.T = DeBoor::evaluate(x, dt, dc, p - 1).normalized();
            data.emplace_back(std::move(pos));
            x += step;
        }
    }

protected:
    /**
     * Build knot vector for De Boor's algorithm
     * @param n Number of control points minus 1
     * @param p Degree of B-spline
     * @return Knot vector
     */
    static std::vector<float> makeKnots(int n, int p) {
        std::vector<float> t(n + p + 2);
        for (int i = 0; i <= n + p + 1; i++) {
            t[i] = (float) i / (float) (n + p + 1);
        }
        DeBoor::makePadding(t, p);
        return t;
    }
};

#endif // CURVE_HPP
