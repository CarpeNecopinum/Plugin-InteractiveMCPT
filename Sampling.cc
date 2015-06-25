#include "Sampling.hh"

using namespace ACG;

std::vector<Vec3d> Sampling::randomDirectionCosPowerTheta(int number, Vec3d n, double exponent)
{
    std::vector<Vec3d> dirs(number, Vec3d(1.0, 0.0, 0.0));
    Vec3d  x_dir, y_dir;
    generateTangentSystem(n, x_dir, y_dir);

    int counter = 0;
    while(counter < number){
        double xi1 = random();
        double theta = std::acos(std::pow(xi1, 1.0 / (exponent + 1.0)));
        double xi2 = random();
        double phi = 2.0 * M_PI * xi2;

        dirs.push_back(
                    x_dir * std::cos(phi) * std::sin(theta) +
                    y_dir * std::sin(phi) * std::cos(theta) +
                    n     * std::cos(theta)
                );
        counter++;
    }
    return dirs;
}


std::vector<Vec3d> Sampling::randomDirectionsCosTheta(int number, Vec3d n) {
    std::vector<Vec3d> dirs(number, Vec3d(1.0, 0.0, 0.0));
    // Hint: Generate random number between -1 and 1 by "(((double)rand()) / ((double)RAND_MAX) - 0.5) * 2.0"
    // Note: Insert a generated direction "d" into vector dirs by "dirs[s] = d"
    Vec3d  x_dir, y_dir;
    generateTangentSystem(n, x_dir, y_dir);
    /// --- start strip --- ///
    int counter = 0;
    while(counter < number){
        double x = randomSymmetric();
        double y = randomSymmetric();
        double r2 = x * x + y * y;
        if(r2 < 1.0){
            dirs[counter] = x * x_dir + y * y_dir + sqrt(1 - r2) * n;
            counter++;
        }
    }
    /// --- end strip --- ///
    return dirs;
}


void Sampling::generateTangentSystem(Vec3d &n, Vec3d &x, Vec3d &y)
{
    n.normalize();
    y = clampToAxis(n);
    x = n % y;
    y = x % n;
    x.normalize();
    y.normalize();
}


Vec3d Sampling::clampToAxis(const Vec3d &n) {
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
