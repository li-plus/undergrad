#pragma once

#include <image.hpp>
#include <algorithm>
#include <queue>
#include <stack>
#include <cstdio>

class Element {
public:
    virtual void draw(Image &img) = 0;
    virtual ~Element() = default;
};

class Line : public Element {

public:
    int xA, yA;
    int xB, yB;
    Vector3f color;
    void draw(Image &img) override {
        // TODO: Implement Bresenham Algorithm
        printf("Draw a line from (%d, %d) to (%d, %d) using color (%f, %f, %f)\n", xA, yA, xB, yB,
                color.x(), color.y(), color.z());

        int dx = xB - xA;
        int dy = yB - yA;
        int abs_dx = abs(dx);
        int abs_dy = abs(dy);
        int step = ((dx > 0) ^ (dy > 0)) ? -1 : 1;

        if (abs_dy > abs_dx) {
            // If |k| > 1, iterate across y axis
            if (yA > yB) {
                std::swap(yA, yB);
                std::swap(xA, xB);
            }
            int x = xA;
            int error = -abs_dy;
            for (int y = yA; y <= yB; y++) {
                if (0 <= x && x < img.Width() && 0 <= y && y < img.Height()) {
                    img.SetPixel(x, y, color);
                }
                error += 2 * abs_dx;
                if (error >= 0) {
                    x += step;
                    error -= 2 * abs_dy;
                }
            }
        } else {
            // If |k| <= 1, iterate across x axis
            if (xA > xB) {
                std::swap(xA, xB);
                std::swap(yA, yB);
            }
            int y = yA;
            int error = -abs_dx;
            for (int x = xA; x <= xB; x++) {
                if (0 <= x && x < img.Width() && 0 <= y && y < img.Height()) {
                    img.SetPixel(x, y, color);
                }
                error += 2 * abs_dy;
                if (error >= 0) {
                    y += step;
                    error -= 2 * abs_dx;
                }
            }
        }
    }
};

class Circle : public Element {

public:
    int cx, cy;
    int radius;
    Vector3f color;
    void draw(Image &img) override {
        // TODO: Implement Algorithm to draw a Circle
        printf("Draw a circle with center (%d, %d) and radius %d using color (%f, %f, %f)\n", cx, cy, radius,
               color.x(), color.y(), color.z());

        int sq_radius = radius * radius;
        int x = 0;
        int y = radius;
        int d = 5 - 4 * radius;
        while (x <= y) {
            for (auto & pos: std::vector<std::pair<int, int>>{
                {cx + x, cy + y}, {cx + x, cy - y}, {cx - x, cy + y}, {cx - x, cy - y},
                {cx + y, cy + x}, {cx + y, cy - x}, {cx - y, cy + x}, {cx - y, cy - x}}) {
                int pos_x = pos.first;
                int pos_y = pos.second;
                if (0 <= pos_x && pos_x < img.Width() && 0 <= pos_y && pos_y < img.Height()) {
                    img.SetPixel(pos_x, pos_y, color);
                }
            }

            if (d < 0) {
                d += 4 * (2 * x + 3);
            } else {
                d += 4 * (2 * (x - y) + 5);
                y--;
            }
            x++;
        }
    }
};

class Fill : public Element {

public:
    int cx, cy;
    Vector3f color;
    void draw(Image &img) override {
        // TODO: Flood fill
        printf("Flood fill source point = (%d, %d) using color (%f, %f, %f)\n", cx, cy,
                color.x(), color.y(), color.z());

        if (!(0 <= cx && cx < img.Width() && 0 <= cy && cy < img.Height())) {
            fprintf(stderr, "Fill source point out of range\n");
            return;
        }

        auto old_color = img.GetPixel(cx, cy);
        std::stack<Vector2f> stk;
        stk.emplace(cx, cy);
        while (!stk.empty()) {
            auto pos = stk.top();
            stk.pop();
            // scan right
            int x;
            int y = pos.y();
            for (x = pos.x(); x < img.Width() && img.GetPixel(x, y) == old_color; x++) {
                img.SetPixel(x, y, color);
            }
            int x_right = x - 1;
            // scan left
            for (x = pos.x() - 1; x >= 0 && img.GetPixel(x, y) == old_color; x--) {
                img.SetPixel(x, y, color);
            }
            int x_left = x + 1;
            // handle upper line
            if (y > 0) {
                int up_x = x_left;
                int up_y = y - 1;
                while (up_x <= x_right) {
                    while (up_x <= x_right && img.GetPixel(up_x, up_y) != old_color) {
                        up_x++;
                    }
                    if (up_x <= x_right) {
                        stk.emplace(up_x, up_y);
                    }
                    while (up_x <= x_right && img.GetPixel(up_x, up_y) == old_color) {
                        up_x++;
                    }
                }
            }
            // handle lower line
            if (y < img.Height() - 1){
                int down_x = x_left;
                int down_y = y + 1;
                while (down_x <= x_right) {
                    while (down_x <= x_right && img.GetPixel(down_x, down_y) != old_color) {
                        down_x++;
                    }
                    if (down_x <= x_right) {
                        stk.emplace(down_x, down_y);
                    }
                    while (down_x <= x_right && img.GetPixel(down_x, down_y) == old_color) {
                        down_x++;
                    }
                }
            }
        }
    }
};