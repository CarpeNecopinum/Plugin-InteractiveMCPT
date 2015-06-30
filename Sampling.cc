#include "Sampling.hh"

using namespace ACG;

namespace Sampling
{


std::vector<Vec3d> randomDirectionCosPowerTheta(int number, Vec3d n, double exponent)
{
    std::vector<Vec3d> dirs;
    Vec3d  x_dir, y_dir;
    generateTangentSystem(n, x_dir, y_dir);

    while (number--) {
        double xi1 = random();
        double costheta = std::pow(xi1, 1.0 / (exponent + 1.0));
        double theta = std::acos(costheta);
        double xi2 = random();
        double phi = 2.0 * M_PI * xi2;

        Vec3d direction =
                x_dir * std::cos(phi) * std::sin(theta) +
                y_dir * std::sin(phi) * std::sin(theta) +
                n     * std::cos(theta);
        dirs.push_back(direction);
    }
    return dirs;
}


std::vector<Vec3d> randomDirectionsCosTheta(int number, Vec3d n) {
    return randomDirectionCosPowerTheta(number, n, 1.0);
    /*std::vector<DirectionSample> dirs;
    Vec3d  x_dir, y_dir;
    generateTangentSystem(n, x_dir, y_dir);

    while(number--){
        double xi1 = random();
        double costheta = std::sqrt(xi1);
        double theta = std::acos(costheta);
        double phi = 2.0 * M_PI * random();
        Vec3d direction =
                x_dir * std::cos(phi) * std::sin(theta) +
                y_dir * std::sin(phi) * std::cos(theta) +
                n     * std::cos(theta);
        double weight = costheta * std::sin(theta) * M_PI;

        dirs.push_back({direction, weight});
    }
    return dirs;*/
}

void testWeight()
{
    /*
    auto samples = randomDirectionCosPowerTheta(100000, Vec3d(0.0, 0.0, 1.0), 2.0);
    double weightsum = 0.0;
    for (auto sample : samples)
    {
        weightsum += 1.0 / sample.density;
    }
    std::cout << weightsum / double(samples.size()) << std::endl;
    */
}

std::vector<Vec3d> randomDirectionsCosThetaOld(unsigned int number, Vec3d n) {
    std::vector<Vec3d> dirs;

    Vec3d  x_dir, y_dir, dir;
    Vec3d sample;

    n.normalize();
    // local coord system
    y_dir = clampToAxis(n);

    x_dir = n % y_dir;
    y_dir = x_dir % n;
    x_dir.normalize();
    y_dir.normalize();

    for (unsigned int s = 0; s < number; ++s)
    {
        Vec3d dir;
        while(true) {
            // method from siggraph 03 course
            double z1 = ((double)rand())/((double)RAND_MAX);
            double z2 = ((double)rand())/((double)RAND_MAX);

            double x = sqrt(1.0 - z1) * cos(2*M_PI*z2);
            double y = sqrt(1.0 - z1) * sin(2*M_PI*z2);
            double z = sqrt(z1);
            dir = x * x_dir + y * y_dir + z * n;
            dir.normalize();
            if((double)(dir|n) >= 0.0) break;
        }
        sample = dir;

        dirs.push_back(sample);
    }
    return dirs;
}

void generateTangentSystem(Vec3d &n, Vec3d &x, Vec3d &y)
{
    n.normalize();
    y = clampToAxis(n);
    x = n % y;
    y = x % n;
    x.normalize();
    y.normalize();
}


Vec3d clampToAxis(const Vec3d &n) {
    Vec3d res;
    if(std::fabs(n[0]) > std::fabs(n[1]) && std::fabs(n[0]) > std::fabs(n[2])) {
        res = Vec3d(0,0,1);
    } else if(std::fabs(n[1]) > std::fabs(n[0]) && std::fabs(n[1]) > std::fabs(n[2])) {
        res = Vec3d(1,0,0);
    } else { //(std::fabs(n[2]) > std::fabs(n[0]) && std::fabs(n[2]) > std::fabs(n[1])) {
        res = Vec3d(0,1,0);
    }
    return res;
}


double brightness(Vec4f color) {
    // Convert to Y (Yuv Color space)
    return (0.299f * color[0] + 0.587f * color[1] + 0.114f * color[2]) * color[3];
}

double densityCosPowerTheta(Vec3d z, double exponent, Vec3d direction)
{
    double costheta = z | direction;
    double theta = std::acos(costheta);
    return (exponent + 1.0) * std::pow(costheta, exponent) * std::sin(theta);
}

double densityCosTheta(Vec3d z, Vec3d direction)
{
    return densityCosPowerTheta(z, 1.0, direction);
    /*
    double costheta = z | direction;
    double theta = std::acos(costheta);
    return costheta * std::sin(theta);*/
}

}
