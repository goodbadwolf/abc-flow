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
#include <vtkRenderWindow.h>
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
    FTLEComputer(int gridResolution);

    void setABCParameters(double a, double b, double c);
    void setAdvectionParams(const ParticleAdvectionParams &params);
    void setNumContours(int numContours);
    void setOutputFormat(const std::string &outputFormat);

    void toggleShowContours();
    void toggleActiveField();

    void advanceCheckpoint();
    void backtrackCheckpoint();

    void render();

    void saveGrid();

private:
    std::vector<Vector3d> seedParticles();
    void computeFTLE();

    std::vector<Vector3d> advectParticles(const std::vector<Vector3d> &particles,
                                          double t0, double tf, double dt, bool forward);
    Vector3d unsteadyABCFlow(const Vector3d &pos, double t);
    std::vector<double> computeFTLEField(const std::vector<Vector3d> &initial,
                                         const std::vector<Vector3d> &advected,
                                         double dt);
    void updateGrid(const std::vector<double> &fftle, const std::vector<double> &bftle);

    void updateScene();

    void saveToVDB(const std::string &filename);
    void saveToVTK(const std::string &filename);
    void saveSimParameters(const std::string &datasetFileName);

    int gridResolution;
    std::vector<double> fftle;
    std::vector<double> bftle;
    double A, B, C;
    std::string outputFormat;
    ParticleAdvectionParams advectionParams;
    int currentCheckpoint = -1;
    bool showContour = false;
    int numContours = 10;
    std::string activeFieldName = "FFTLE";
    vtkSmartPointer<vtkImageData> grid;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
};