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

struct ParticleAdvectionParams
{
    double t0;
    double tf;
    double dt;
    int numCheckpoints;
};

class FTLEComputer
{
public:
    FTLEComputer(int resolution, std::string format);

    vtkSmartPointer<vtkImageData> getGrid() { return grid; }
    std::string getActiveFieldName() { return activeField; }

    int getNumIsos() { return numContours; }
    void setNumIsos(int numIsos) { numContours = numIsos; }

    void setABCParameters(double a, double b, double c);

    std::string getFormat() { return format; }

    bool toggleContour();
    void toggleActiveField();

    void setAdvectionParams(const ParticleAdvectionParams &params)
    {
        advectionParams = params;
        currentCheckpoint = -1;
    }

    void advanceCheckpoint();
    void backtrackCheckpoint();

    void render();

    void saveGrid();

private:
    Vector3d unsteadyABCFlow(const Vector3d &pos, double t);
    std::vector<Vector3d> seedParticles();
    std::vector<Vector3d> advectParticles(const std::vector<Vector3d> &particles,
                                          double t0, double tf, double dt, bool forward);
    void computeFTLE();
    std::vector<double> computeFTLEField(const std::vector<Vector3d> &initial,
                                         const std::vector<Vector3d> &advected,
                                         double dt);
    void updateGrid(const std::vector<double> &fftle, const std::vector<double> &bftle);
    void resetContourFilter();
    void saveParameters(const std::string &datasetFileName);
    void saveToVDB(const std::string &filename);
    void saveToVTK(const std::string &filename);
    void updateScene();

    int resolution;
    double cellSpacing;
    std::vector<double> fftle;
    std::vector<double> bftle;
    double A, B, C;
    std::string format;
    ParticleAdvectionParams advectionParams;
    int currentCheckpoint = -1;
    bool showContour = false;
    int numContours = 10;
    std::string activeField = "FFTLE";
    vtkSmartPointer<vtkImageData> grid;
    vtkSmartPointer<vtkActor> cubeActor;
    vtkSmartPointer<vtkActor> contourActor;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkContourFilter> contourFilter;
};