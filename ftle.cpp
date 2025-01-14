#include "ftle.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <execution>
#include <filesystem>
#include <iostream>
#include <string>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>

#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkCallbackCommand.h>
#include <vtkDataSetMapper.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkRectilinearGrid.h>
#include <vtkRectilinearGridWriter.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkScalarBarActor.h>
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataWriter.h>

std::string ToLower(const std::string &str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c)
                   { return std::tolower(c); });
    return result;
}

std::string FindNextAvailableFileName(std::function<std::string(int counter)> fileNameGenerator)
{
    int counter = 0;
    std::string fileName = fileNameGenerator(counter);
    while (std::filesystem::exists(fileName))
    {
        fileName = fileNameGenerator(++counter);
    }
    return fileName;
}

void KeypressCallbackFunction(vtkObject *caller, long unsigned int eventId, void *clientData, void *callData)
{
    auto computer = static_cast<FTLEComputer *>(clientData);
    auto grid = computer->getGrid();
    auto interactor = static_cast<vtkRenderWindowInteractor *>(caller);

    std::string key = interactor->GetKeySym();
    if (key == "s")
    {
        std::cout << "Using format: " << computer->getFormat() << std::endl;
        std::string fileName = FindNextAvailableFileName([&computer](int counter)
                                                         { 
                                    std::string ext = computer->getFormat() == "vtk" ? ".vti" : ".vdb";
                                    return "ftle_" + ToLower(computer->getActiveFieldName()) + "_" + std::to_string(counter) + ext; });
        if (computer->getFormat() == "vtk")
        {
            computer->saveToVTK(fileName);
        }
        else
        {
            computer->saveToVDB(fileName);
        }
    }
    else if (key == "i")
    {
        computer->toggleContour();
    }
    else if (key == "f")
    {
        computer->toggleActiveField();
    }
    else if (key == "q")
    {
        interactor->GetRenderWindow()->Finalize();
        interactor->TerminateApp();
    }
}

FTLEComputer::FTLEComputer(int res, std::string format) : resolution(res), format(format), A(1.0), B(1.0), C(1.0)
{
    cellSpacing = 2.0 * M_PI / (resolution - 1);
    grid = vtkSmartPointer<vtkImageData>::New();
    grid->SetDimensions(resolution, resolution, resolution);
    grid->SetSpacing(cellSpacing, cellSpacing, cellSpacing);
    grid->SetOrigin(0, 0, 0);
}

Vector3d FTLEComputer::unsteadyABCFlow(const Vector3d &pos, double t)
{
    double uA = A + 0.5 * t * std::sin(M_PI * t);
    double uB = B;
    double uC = C;

    return Vector3d(
        uA * std::sin(pos.z) + uB * std::cos(pos.y),
        uB * std::sin(pos.x) + uC * std::cos(pos.z),
        uC * std::sin(pos.y) + uA * std::cos(pos.x));
}

std::vector<Vector3d> FTLEComputer::seedParticles()
{
    std::vector<Vector3d> particles;
    particles.reserve(resolution * resolution * resolution);
    auto origin = grid->GetOrigin();

    for (int k = 0; k < resolution; ++k)
    {
        for (int j = 0; j < resolution; ++j)
        {
            for (int i = 0; i < resolution; ++i)
            {
                particles.emplace_back(
                    origin[0] + i * cellSpacing,
                    origin[1] + j * cellSpacing,
                    origin[2] + k * cellSpacing);
            }
        }
    }
    return particles;
}

std::vector<Vector3d> FTLEComputer::advectParticles(
    const std::vector<Vector3d> &particles,
    double t0, double tf, double dt, bool forward)
{

    std::vector<Vector3d> advectedParticles = particles;
    int numSteps = std::abs(tf - t0) / dt;

    std::for_each(std::execution::par_unseq,
                  advectedParticles.begin(),
                  advectedParticles.end(),
                  [&](Vector3d &particle)
                  {
                      for (int step = 0; step < numSteps; ++step)
                      {
                          double t = forward ? t0 + step * dt : t0 - step * dt;
                          particle = particle + unsteadyABCFlow(particle, t) * dt;
                      }
                  });

    return advectedParticles;
}

std::vector<double> FTLEComputer::computeFTLEField(
    const std::vector<Vector3d> &initial,
    const std::vector<Vector3d> &advected,
    double dt)
{

    std::vector<double> ftle(initial.size());

    std::transform(std::execution::par_unseq,
                   initial.begin(), initial.end(),
                   advected.begin(),
                   ftle.begin(),
                   [dt](const Vector3d &init, const Vector3d &adv)
                   {
                       double dx = adv.x - init.x;
                       double dy = adv.y - init.y;
                       double dz = adv.z - init.z;
                       double dF = std::sqrt(dx * dx + dy * dy + dz * dz);
                       return std::log(dF) / dt;
                   });

    return ftle;
}

void FTLEComputer::computeFTLE()
{
    auto [particles, seedTime] = measureTime(&FTLEComputer::seedParticles, this);
    std::cout << "Particle seeding time: " << seedTime << " seconds\n";

    double t0_forward = 0.0;
    double tf_forward = 10.0;
    double dt = 0.1;

    auto [fwdParticles, fwdTime] = measureTime(
        &FTLEComputer::advectParticles, this,
        particles, t0_forward, tf_forward, dt, true);
    std::cout << "Forward advection time: " << fwdTime << " seconds\n";

    auto [fftleResult, fftleTime] = measureTime(
        &FTLEComputer::computeFTLEField, this,
        particles, fwdParticles, tf_forward - t0_forward);
    fftle = std::move(fftleResult);
    std::cout << "Forward FTLE computation time: " << fftleTime << " seconds\n";

    double t0_backward = 10.0;
    double tf_backward = 0.0;

    auto [bwdParticles, bwdTime] = measureTime(
        &FTLEComputer::advectParticles, this,
        particles, t0_backward, tf_backward, dt, false);
    std::cout << "Backward advection time: " << bwdTime << " seconds\n";

    auto [bftleResult, bftleTime] = measureTime(
        &FTLEComputer::computeFTLEField, this,
        particles, bwdParticles, tf_backward - t0_backward);
    bftle = std::move(bftleResult);
    std::cout << "Backward FTLE computation time: " << bftleTime << " seconds\n";

    auto fftleArray = vtkSmartPointer<vtkDoubleArray>::New();
    fftleArray->SetName("FFTLE");
    fftleArray->SetNumberOfComponents(1);
    fftleArray->SetNumberOfTuples(grid->GetNumberOfPoints());
    std::copy(fftle.begin(), fftle.end(), fftleArray->GetPointer(0));

    auto bftleArray = vtkSmartPointer<vtkDoubleArray>::New();
    bftleArray->SetName("BFTLE");
    bftleArray->SetNumberOfComponents(1);
    bftleArray->SetNumberOfTuples(grid->GetNumberOfPoints());
    std::copy(bftle.begin(), bftle.end(), bftleArray->GetPointer(0));

    grid->GetPointData()->AddArray(fftleArray);
    grid->GetPointData()->AddArray(bftleArray);

    this->activeField = "FFTLE";
    grid->GetPointData()->SetActiveScalars(activeField.c_str());
}

void FTLEComputer::saveToVDB(const std::string &fileName)
{
    openvdb::initialize();

    auto fftle_grid = openvdb::FloatGrid::create();
    auto bftle_grid = openvdb::FloatGrid::create();

    fftle_grid->setName("FFTLE");
    bftle_grid->setName("BFTLE");

    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(cellSpacing);
    fftle_grid->setTransform(transform);
    bftle_grid->setTransform(transform);

    openvdb::tools::Dense<float>
        fftle_dense(openvdb::Coord(resolution, resolution, resolution));
    openvdb::tools::Dense<float>
        bftle_dense(openvdb::Coord(resolution, resolution, resolution));

    for (int k = 0; k < resolution; ++k)
    {
        for (int j = 0; j < resolution; ++j)
        {
            for (int i = 0; i < resolution; ++i)
            {
                int idx = k * resolution * resolution + j * resolution + i;
                fftle_dense.setValue(i, j, k, fftle[idx]);
                bftle_dense.setValue(i, j, k, bftle[idx]);
            }
        }
    }

    openvdb::tools::copyFromDense(fftle_dense, *fftle_grid, 0.0f, true);
    openvdb::tools::copyFromDense(bftle_dense, *bftle_grid, 0.0f, true);

    openvdb::GridPtrVec grids;
    if (this->activeField == "FFTLE")
    {
        grids.push_back(fftle_grid);
    }
    else
    {
        grids.push_back(bftle_grid);
    }

    openvdb::io::File file(fileName);
    file.write(grids);
    file.close();
    saveParameters(fileName);

    std::cout << "Saved OpenVDB file to: " << fileName << "\n";
    std::cout << "Grid statistics:\n";
    if (this->activeField == "FFTLE")
    {
        std::cout << "FFTLE grid active voxels: " << fftle_grid->activeVoxelCount() << "\n";
    }
    else
    {
        std::cout << "BFTLE grid active voxels: " << bftle_grid->activeVoxelCount() << "\n";
    }
}

void FTLEComputer::saveToVTK(const std::string &fileName)
{
    vtkSmartPointer<vtkXMLImageDataWriter> writer =
        vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(fileName.c_str());
    writer->SetInputData(grid);
    writer->Write();
    saveParameters(fileName);
}

void FTLEComputer::saveParameters(const std::string &datasetFileName)
{
    std::string fileName = datasetFileName.substr(0, datasetFileName.find_last_of('.')) + "_params.txt";
    std::ofstream paramFile(fileName);
    paramFile << "Resolution: " << resolution << "\n";
    paramFile << "A: " << A << "\n";
    paramFile << "B: " << B << "\n";
    paramFile << "C: " << C << "\n";
    paramFile.close();
}

void FTLEComputer::setABCParameters(double a, double b, double c)
{
    A = a;
    B = b;
    C = c;
}

bool FTLEComputer::toggleContour()
{
    showContour = !showContour;
    this->cubeActor->SetVisibility(!showContour);
    this->contourActor->SetVisibility(showContour);
    this->cubeActor->GetMapper()->Update();
    this->contourActor->GetMapper()->Update();
    return showContour;
}

void FTLEComputer::resetContourFilter()
{
    contourFilter = vtkSmartPointer<vtkContourFilter>::New();
    contourFilter->SetInputData(grid);
    contourFilter->GenerateValues(this->numContours, grid->GetPointData()->GetScalars()->GetRange());
}

void FTLEComputer::renderWithSave()
{
    if (!grid || !grid->GetPointData() || !grid->GetPointData()->GetScalars())
    {
        std::cerr << "Error: No scalar data available for rendering.\n";
        return;
    }

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(grid);
    mapper->SetScalarRange(grid->GetPointData()->GetScalars()->GetRange());
    this->cubeActor = vtkSmartPointer<vtkActor>::New();
    this->cubeActor->SetMapper(mapper);

    this->resetContourFilter();
    vtkSmartPointer<vtkDataSetMapper> contourMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    contourMapper->SetInputConnection(contourFilter->GetOutputPort());
    this->contourActor = vtkSmartPointer<vtkActor>::New();
    this->contourActor->SetMapper(contourMapper);
    this->contourActor->SetVisibility(showContour);

    this->renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(this->cubeActor);
    renderer->AddActor(this->contourActor);
    renderer->SetBackground(0.1, 0.2, 0.4);
    renderer->ResetCamera();

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(1920, 1080);
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    renderWindowInteractor->SetInteractorStyle(style);

    vtkSmartPointer<vtkScalarBarActor> scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar->SetLookupTable(mapper->GetLookupTable());
    scalarBar->SetTitle("FTLE");
    scalarBar->SetNumberOfLabels(4);
    renderer->AddActor2D(scalarBar);

    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
    vtkSmartPointer<vtkOrientationMarkerWidget> axesWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    axesWidget->SetOutlineColor(0.9300, 0.5700, 0.1300);
    axesWidget->SetOrientationMarker(axes);
    axesWidget->SetInteractor(renderWindowInteractor);
    axesWidget->SetViewport(0.0, 0.0, 0.2, 0.2);
    axesWidget->SetEnabled(1);
    axesWidget->InteractiveOff();

    vtkSmartPointer<vtkCallbackCommand> keypressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    keypressCallback->SetCallback(KeypressCallbackFunction);
    keypressCallback->SetClientData(this);
    renderWindowInteractor->AddObserver(vtkCommand::KeyPressEvent, keypressCallback);

    renderWindow->Render();
    renderWindowInteractor->Start();
}

void FTLEComputer::toggleActiveField()
{
    auto pointData = grid->GetPointData();
    if (!pointData || !pointData->GetScalars())
    {
        std::cerr << "No scalar data available to toggle.\n";
        return;
    }
    std::string currentScalar = pointData->GetScalars()->GetName();
    this->activeField = (currentScalar == "FFTLE") ? "BFTLE" : "FFTLE";
    pointData->SetActiveScalars(this->activeField.c_str());
    std::cout << "Toggled active scalar to: " << this->activeField << "\n";

    cubeActor->GetMapper()->SetScalarRange(grid->GetPointData()->GetScalars()->GetRange());
    cubeActor->GetMapper()->Update();
    this->resetContourFilter();
    contourActor->GetMapper()->SetScalarRange(grid->GetPointData()->GetScalars()->GetRange());
    contourActor->GetMapper()->Update();

    this->renderer->ResetCamera();
    this->renderer->ResetCameraScreenSpace();
}