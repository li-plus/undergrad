#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

#include "scene_parser.hpp"
#include "image.hpp"
#include "camera.hpp"
#include "group.hpp"
#include "light.hpp"

#include <string>

using namespace std;

int main(int argc, char *argv[]) {
    for (int argNum = 1; argNum < argc; ++argNum) {
        std::cout << "Argument " << argNum << " is: " << argv[argNum] << std::endl;
    }

    if (argc != 3) {
        cout << "Usage: ./bin/PA1 <input scene file> <output bmp file>" << endl;
        return 1;
    }
    string inputFile = argv[1];
    string outputFile = argv[2];  // only bmp is allowed.

    // TODO: Main RayCasting Logic
    // First, parse the scene using SceneParser.
    // Then loop over each pixel in the image, shooting a ray
    // through that pixel and finding its intersection with
    // the scene.  Write the color at the intersection to that
    // pixel in your output image.
    cout << "Hello! Computer Graphics!" << endl;

    SceneParser sceneParser(inputFile.c_str());
    Camera *camera = sceneParser.getCamera();

    Image img(camera->getWidth(), camera->getHeight());

    // Loop over all pixels
    for (int x = 0; x < camera->getWidth(); ++x) {
        for (int y = 0; y < camera->getHeight(); ++y) {
            // Compute the ray from the current pixel (x,y)
            Ray camRay = camera->generateRay(Vector2f(x, y));
            Group * baseGroup = sceneParser.getGroup();
            Hit hit;
            // Find out whether camRay intersects with the scene,
            // and store the nearest intersection in hit.
            bool isIntersect = baseGroup->intersect(camRay, hit, 0);
            if (isIntersect) {
                // Found intersection: accumulate illumination from all lights.
                Vector3f finalColor = Vector3f::ZERO;
                for (int li = 0; li < sceneParser.getNumLights(); ++li) {
                    Light * light = sceneParser.getLight(li);
                    Vector3f L, lightColor;
                    // Light illumination
                    light->getIllumination(camRay.pointAtParameter(hit.getT()), L, lightColor);
                    // Local illumination
                    finalColor += hit.getMaterial()->Shade(camRay, hit, L, lightColor);
                }
                img.SetPixel(x, y, finalColor);
            } else {
                // No intersection: return background color
                img.SetPixel(x, y, sceneParser.getBackgroundColor());
            }
        }
    }
    img.SaveBMP(outputFile.c_str());
    return 0;
}

