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
#include <vtkActorCollection.h>
#include <vtkActor2D.h>
#include <vtkActor2DCollection.h>
#include <vtkAxesActor.h>
#include <vtkCallbackCommand.h>
#include <vtkDataSetMapper.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLookupTable.h>
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
        computer->saveGrid();
    }
    else if (key == "i")
    {
        computer->toggleContour();
    }
    else if (key == "f")
    {
        computer->toggleActiveField();
    }
    else if (key == "a")
    {
        computer->advanceCheckpoint();
    }
    else if (key == "b")
    {
        computer->backtrackCheckpoint();
    }
    else if (key == "q")
    {
        interactor->GetRenderWindow()->Finalize();
        interactor->TerminateApp();
    }
}

template <typename F, typename... Args>
auto measureTime(F &&func, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto result = std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    return std::make_pair(result, duration);
}

void FTLEComputer::saveGrid()
{
    std::cout << "Using format: " << this->getFormat() << std::endl;
    std::string fileName = FindNextAvailableFileName([this](int counter)
                                                     { 
                                    std::string ext = this->getFormat() == "vtk" ? ".vti" : ".vdb";
                                    return ToLower(this->getActiveFieldName()) + "_" + std::to_string(counter) + ext; });
    if (this->getFormat() == "vtk")
    {
        this->saveToVTK(fileName);
    }
    else
    {
        this->saveToVDB(fileName);
    }
}

FTLEComputer::FTLEComputer(int res, std::string format) : resolution(res), format(format), A(1.0), B(1.0), C(1.0),
                                                          advectionParams({0.0, 10.0, 0.1, 1}), currentCheckpoint(-1)
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

void FTLEComputer::advanceCheckpoint()
{
    if (currentCheckpoint >= advectionParams.numCheckpoints - 1)
    {
        std::cout << "Already at the last checkpoint\n";
        return;
    }
    currentCheckpoint++;
    this->computeFTLE();
}

void FTLEComputer::backtrackCheckpoint()
{
    if (currentCheckpoint <= 0)
    {
        std::cout << "Already at the first checkpoint\n";
        return;
    }
    currentCheckpoint--;
    this->computeFTLE();
}

void FTLEComputer::computeFTLE()
{
    auto [particles, seedTime] = measureTime(&FTLEComputer::seedParticles, this);
    std::cout << "Particle seeding time: " << seedTime << " seconds\n";

    double totalDuration = advectionParams.tf - advectionParams.t0;
    double checkpointDuration = totalDuration / advectionParams.numCheckpoints;

    double t_start = advectionParams.t0 + currentCheckpoint * checkpointDuration;
    double t_end = advectionParams.t0 + (currentCheckpoint + 1) * checkpointDuration;
    std::cout << "Computing FTLE for checkpoint " << currentCheckpoint << "\n";
    std::cout << "Time interval: [" << t_start << ", " << t_end << "]\n";

    auto [fwdParticles, fwdTime] = measureTime(
        &FTLEComputer::advectParticles, this,
        particles, t_start, t_end, advectionParams.dt, true);
    std::cout << "Forward advection time: " << fwdTime << " seconds\n";

    auto [fftleResult, fftleTime] = measureTime(
        &FTLEComputer::computeFTLEField, this,
        particles, fwdParticles, std::abs(t_end - t_start));
    fftle = std::move(fftleResult);
    std::cout << "Forward FTLE computation time: " << fftleTime << " seconds\n";

    double t0_backward = 10.0;
    double tf_backward = 0.0;

    auto [bwdParticles, bwdTime] = measureTime(
        &FTLEComputer::advectParticles, this,
        particles, t_end, t_start, advectionParams.dt, false);
    std::cout << "Backward advection time: " << bwdTime << " seconds\n";

    auto [bftleResult, bftleTime] = measureTime(
        &FTLEComputer::computeFTLEField, this,
        particles, bwdParticles, std::abs(t_end - t_start));
    bftle = std::move(bftleResult);
    std::cout << "Backward FTLE computation time: " << bftleTime << " seconds\n";

    this->updateGrid(fftle, bftle);
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

void FTLEComputer::updateGrid(const std::vector<double> &fftle, const std::vector<double> &bftle)
{
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

    auto pointData = grid->GetPointData();
    if (pointData->HasArray("FFTLE"))
    {
        pointData->RemoveArray("FFTLE");
    }

    if (pointData->HasArray("BFTLE"))
    {
        pointData->RemoveArray("BFTLE");
    }
    pointData->AddArray(fftleArray);
    pointData->AddArray(bftleArray);
    pointData->SetActiveScalars(this->activeField.c_str());
    pointData->Modified();
    grid->Modified();
}

void FTLEComputer::updateScene()
{
    auto actors = renderer->GetActors();
    for (int i = 0; i < actors->GetNumberOfItems(); ++i)
    {
        renderer->RemoveActor(actors->GetNextActor());
    }
    auto actors2D = renderer->GetActors2D();
    for (int i = 0; i < actors2D->GetNumberOfItems(); ++i)
    {
        renderer->RemoveActor2D(actors2D->GetNextActor2D());
    }

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(grid);
    mapper->SetScalarRange(grid->GetPointData()->GetScalars()->GetRange());
    this->cubeActor = vtkSmartPointer<vtkActor>::New();
    this->cubeActor->SetMapper(mapper);
    renderer->AddActor(cubeActor);

    this->resetContourFilter();
    vtkSmartPointer<vtkDataSetMapper> contourMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    contourMapper->SetInputConnection(contourFilter->GetOutputPort());

    double scalarRange[2];
    grid->GetPointData()->GetScalars()->GetRange(scalarRange);

    // Create a lookup table with 256 table values.
    vtkSmartPointer<vtkLookupTable> contourLUT = vtkSmartPointer<vtkLookupTable>::New();
    int tableSize = 256;
    contourLUT->SetNumberOfTableValues(tableSize);
    contourLUT->SetRange(scalarRange);
    contourLUT->Build();

    // Fill the lookup table. In this example we use a blue-to-red color map.
    // The alpha values are ramped from 0 at the low end to 1 at the high end.
    for (int i = 0; i < tableSize; i++)
    {
        double t = static_cast<double>(i) / (tableSize - 1); // normalized [0,1]
        double alpha = t;                                    // lowest (t=0) gets 0 alpha; highest (t=1) gets 1 alpha

        // Example: blue-to-red color interpolation:
        // At t=0, color = blue (0, 0, 1); at t=1, color = red (1, 0, 0)
        double r = t;
        double g = 0.0;
        double b = 1.0 - t;

        contourLUT->SetTableValue(i, r, g, b, alpha);
    }

    // Assign the lookup table to the contour mapper.
    contourMapper->SetLookupTable(contourLUT);
    contourMapper->SetScalarRange(scalarRange);

    this->contourActor = vtkSmartPointer<vtkActor>::New();
    this->contourActor->SetMapper(contourMapper);
    this->contourActor->SetVisibility(showContour);

    vtkSmartPointer<vtkScalarBarActor> scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar->SetLookupTable(mapper->GetLookupTable());
    scalarBar->SetTitle("FTLE");
    scalarBar->SetNumberOfLabels(4);
    renderer->AddActor2D(scalarBar);

    renderer->SetBackground(0.1, 0.2, 0.4);
    renderer->ResetCamera();
}

void FTLEComputer::render()
{
    if (!grid || !grid->GetPointData() || !grid->GetPointData()->GetScalars())
    {
        std::cerr << "Error: No scalar data available for rendering.\n";
        return;
    }

    this->renderer = vtkSmartPointer<vtkRenderer>::New();
    this->updateScene();

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(1920, 1080);
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    renderWindowInteractor->SetInteractorStyle(style);

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