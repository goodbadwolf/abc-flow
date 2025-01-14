#pragma once

#include <array>
#include <chrono>
#include <functional>
#include <vector>

#include <vtkActor.h>
#include <vtkContourFilter.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

template <typename F, typename... Args>
auto measureTime(F &&func, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto result = std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    return std::make_pair(result, duration);
}

// Structure to hold 3D vector
struct Vector3d
{
    double x, y, z;

    Vector3d(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    Vector3d operator+(const Vector3d &other) const
    {
        return Vector3d(x + other.x, y + other.y, z + other.z);
    }

    Vector3d operator*(double scalar) const
    {
        return Vector3d(x * scalar, y * scalar, z * scalar);
    }
};

class FTLEComputer
{
public:
    FTLEComputer(int resolution, std::string format);
    void computeFTLE();
    void saveToVDB(const std::string &filename);
    void saveToVTK(const std::string &filename);
    vtkSmartPointer<vtkImageData> getGrid() { return grid; }
    void setABCParameters(double a, double b, double c);
    void renderWithSave();
    std::string getFormat() { return format; }
    bool toggleContour();
    void toggleActiveField();

private:
    Vector3d abcFlow(const Vector3d &pos, double t);
    Vector3d unsteadyABCFlow(const Vector3d &pos, double t);
    std::vector<Vector3d> seedParticles();
    std::vector<Vector3d> advectParticles(const std::vector<Vector3d> &particles,
                                          double t0, double tf, double dt, bool forward);
    std::vector<double> computeFTLEField(const std::vector<Vector3d> &initial,
                                         const std::vector<Vector3d> &advected,
                                         double dt);
    void resetContourFilter();
    int resolution;
    double cellSpacing;
    std::vector<double> fftle;
    std::vector<double> bftle;
    double A, B, C;
    std::string format;
    bool showContour = false;
    int numContours = 10;
    std::string activeField = "FFTLE";
    vtkSmartPointer<vtkImageData> grid;
    vtkSmartPointer<vtkActor> cubeActor;
    vtkSmartPointer<vtkActor> contourActor;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkContourFilter> contourFilter;
};