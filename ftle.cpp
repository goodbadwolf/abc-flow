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
#include <vtkCamera.h>
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
    auto interactor = static_cast<vtkRenderWindowInteractor *>(caller);

    std::string key = interactor->GetKeySym();
    if (key == "s")
    {
        computer->saveGrid();
    }
    else if (key == "i")
    {
        computer->toggleShowContours();
    }
    else if (key == "t")
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

FTLEComputer::FTLEComputer(int gridResolution) : gridResolution(gridResolution), outputFormat("vdb"), A(1.0), B(1.0), C(1.0),
                                                 advectionParams({0.0, 10.0, 0.1, 1, -1})
{
    double cellSpacing = 2.0 * M_PI / (gridResolution - 1);
    grid = vtkSmartPointer<vtkImageData>::New();
    grid->SetDimensions(gridResolution, gridResolution, gridResolution);
    grid->SetSpacing(cellSpacing, cellSpacing, cellSpacing);
    grid->SetOrigin(0, 0, 0);
}

void FTLEComputer::setABCParameters(double a, double b, double c)
{
    A = a;
    B = b;
    C = c;
}

void FTLEComputer::setAdvectionParams(const ParticleAdvectionParams &params)
{
    advectionParams = params;
}

void FTLEComputer::setOutputFormat(const std::string &outputFormat)
{
    this->outputFormat = outputFormat;
}

void FTLEComputer::setNumContours(int numContours)
{
    this->numContours = numContours;
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
    particles.reserve(gridResolution * gridResolution * gridResolution);
    auto origin = grid->GetOrigin();
    auto spacing = grid->GetSpacing();

    for (int k = 0; k < gridResolution; ++k)
    {
        for (int j = 0; j < gridResolution; ++j)
        {
            for (int i = 0; i < gridResolution; ++i)
            {
                particles.emplace_back(
                    origin[0] + i * spacing[0],
                    origin[1] + j * spacing[1],
                    origin[2] + k * spacing[2]);
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
    if (advectionParams.currentCheckpoint >= advectionParams.numCheckpoints - 1)
    {
        std::cout << "Already at the last checkpoint\n";
        return;
    }
    advectionParams.currentCheckpoint++;
    this->computeFTLE();
    if (this->renderer)
    {
        this->updateScene();
        this->renderer->GetRenderWindow()->Render();
    }
}

void FTLEComputer::backtrackCheckpoint()
{
    if (advectionParams.currentCheckpoint <= 0)
    {
        std::cout << "Already at the first checkpoint\n";
        return;
    }
    advectionParams.currentCheckpoint--;
    this->computeFTLE();
    if (this->renderer)
    {
        this->updateScene();
        this->renderer->GetRenderWindow()->Render();
    }
}

void FTLEComputer::computeFTLE()
{
    auto [particles, seedTime] = measureTime(&FTLEComputer::seedParticles, this);
    std::cout << "Particle seeding time: " << seedTime << " seconds\n";

    double totalDuration = advectionParams.tf - advectionParams.t0;
    double checkpointDuration = totalDuration / advectionParams.numCheckpoints;

    double t_start = advectionParams.t0 + advectionParams.currentCheckpoint * checkpointDuration;
    double t_end = advectionParams.t0 + (advectionParams.currentCheckpoint + 1) * checkpointDuration;
    std::cout << "Computing FTLE for checkpoint " << advectionParams.currentCheckpoint << "\n";
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
    std::cout << "FFTLE range: [" << *std::min_element(fftle.begin(), fftle.end()) << ", "
              << *std::max_element(fftle.begin(), fftle.end()) << "]\n";

    auto [bwdParticles, bwdTime] = measureTime(
        &FTLEComputer::advectParticles, this,
        particles, t_end, t_start, advectionParams.dt, false);
    std::cout << "Backward advection time: " << bwdTime << " seconds\n";

    auto [bftleResult, bftleTime] = measureTime(
        &FTLEComputer::computeFTLEField, this,
        particles, bwdParticles, std::abs(t_end - t_start));
    bftle = std::move(bftleResult);
    std::cout << "Backward FTLE computation time: " << bftleTime << " seconds\n";
    std::cout << "BFTLE range: [" << *std::min_element(bftle.begin(), bftle.end()) << ", "
              << *std::max_element(bftle.begin(), bftle.end()) << "]\n";

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
        openvdb::math::Transform::createLinearTransform(grid->GetSpacing()[0]);
    fftle_grid->setTransform(transform);
    bftle_grid->setTransform(transform);

    openvdb::tools::Dense<float>
        fftle_dense(openvdb::Coord(gridResolution, gridResolution, gridResolution));
    openvdb::tools::Dense<float>
        bftle_dense(openvdb::Coord(gridResolution, gridResolution, gridResolution));

    for (int k = 0; k < gridResolution; ++k)
    {
        for (int j = 0; j < gridResolution; ++j)
        {
            for (int i = 0; i < gridResolution; ++i)
            {
                int idx = k * gridResolution * gridResolution + j * gridResolution + i;
                fftle_dense.setValue(i, j, k, fftle[idx]);
                bftle_dense.setValue(i, j, k, bftle[idx]);
            }
        }
    }

    openvdb::tools::copyFromDense(fftle_dense, *fftle_grid, 0.0f, true);
    openvdb::tools::copyFromDense(bftle_dense, *bftle_grid, 0.0f, true);

    openvdb::GridPtrVec grids;
    if (this->activeFieldName == "FFTLE")
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

    std::cout << "Saved OpenVDB file to: " << fileName << "\n";
    std::cout << "Grid statistics:\n";
    if (this->activeFieldName == "FFTLE")
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
}

void FTLEComputer::saveSimParameters(const std::string &fileName)
{
    std::ofstream paramFile(fileName);
    paramFile << "gridResolution: " << gridResolution << "x" << gridResolution << "x" << gridResolution << "\n";
    paramFile << "A: " << A << "\n";
    paramFile << "B: " << B << "\n";
    paramFile << "C: " << C << "\n";
    paramFile << "t0: " << advectionParams.t0 << "\n";
    paramFile << "tf: " << advectionParams.tf << "\n";
    paramFile << "dt: " << advectionParams.dt << "\n";
    paramFile << "currentCheckpoint: " << advectionParams.currentCheckpoint << "\n";
    paramFile << "numCheckpoints: " << advectionParams.numCheckpoints << "\n";
    paramFile.close();
}

void FTLEComputer::toggleShowContours()
{
    showContour = !showContour;
    if (renderer)
    {
        this->updateScene();
        renderer->GetRenderWindow()->Render();
    }
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
    pointData->AddArray(fftleArray);
    pointData->AddArray(bftleArray);
    pointData->SetActiveScalars(this->activeFieldName.c_str());
    pointData->Modified();
    grid->Modified();
}

void FTLEComputer::updateScene()
{
    bool retainCamera = false;
    double cameraPos[3], cameraFocal[3], cameraViewUp[3];

    if (this->renderWindow && this->renderWindow->HasRenderer(renderer))
    {
        retainCamera = true;
        renderer->GetActiveCamera()->GetPosition(cameraPos);
        renderer->GetActiveCamera()->GetFocalPoint(cameraFocal);
        renderer->GetActiveCamera()->GetViewUp(cameraViewUp);

        this->renderWindow->RemoveRenderer(renderer);
    }
    this->renderer = vtkSmartPointer<vtkRenderer>::New();
    this->renderWindow->AddRenderer(renderer);

    auto cubeMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    cubeMapper->SetInputData(grid);
    cubeMapper->SetScalarRange(grid->GetPointData()->GetScalars()->GetRange());
    auto cubeActor = vtkSmartPointer<vtkActor>::New();
    cubeActor->SetMapper(cubeMapper);
    cubeActor->SetVisibility(!showContour);
    renderer->AddActor(cubeActor);

    auto contourFilter = vtkSmartPointer<vtkContourFilter>::New();
    contourFilter->SetInputData(grid);
    contourFilter->GenerateValues(this->numContours, grid->GetScalarRange());

    auto contourMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    contourMapper->SetInputConnection(contourFilter->GetOutputPort());
    contourMapper->SetScalarRange(grid->GetScalarRange());
    auto contourActor = vtkSmartPointer<vtkActor>::New();
    contourActor->SetMapper(contourMapper);
    contourActor->SetVisibility(showContour);
    renderer->AddActor(contourActor);

    auto lut = showContour ? contourMapper->GetLookupTable() : cubeMapper->GetLookupTable();
    auto scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar->SetLookupTable(lut);
    scalarBar->SetTitle("FTLE");
    scalarBar->SetNumberOfLabels(10);
    renderer->AddActor2D(scalarBar);

    renderer->SetBackground(0.1, 0.2, 0.4);
    renderer->ResetCamera();

    if (retainCamera)
    {
        renderer->GetActiveCamera()->SetPosition(cameraPos);
        renderer->GetActiveCamera()->SetFocalPoint(cameraFocal);
        renderer->GetActiveCamera()->SetViewUp(cameraViewUp);
    }
}

void FTLEComputer::render()
{
    if (!grid || !grid->GetPointData() || !grid->GetPointData()->GetScalars())
    {
        std::cerr << "Error: No scalar data available for rendering.\n";
        return;
    }

    renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(1920, 1080);

    this->updateScene();

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
    this->activeFieldName = (this->activeFieldName == "FFTLE") ? "BFTLE" : "FFTLE";
    pointData->SetActiveScalars(this->activeFieldName.c_str());
    std::cout << "Toggled active scalar to: " << this->activeFieldName << "\n";

    if (renderer)
    {
        this->updateScene();
        renderer->GetRenderWindow()->Render();
    }
}

void FTLEComputer::saveGrid()
{
    std::cout << "Saving grid using format: " << this->outputFormat << std::endl;
    std::string fileName = FindNextAvailableFileName([this](int counter)
                                                     { 
                                    std::string ext = this->outputFormat == "vtk" ? ".vti" : ".vdb";
                                    return ToLower(this->activeFieldName) + "_" + std::to_string(counter) + ext; });
    if (this->outputFormat == "vtk")
    {
        this->saveToVTK(fileName);
    }
    else
    {
        this->saveToVDB(fileName);
    }

    std::string paramsFileName = fileName.substr(0, fileName.find_last_of('.')) + "_params.txt";
    this->saveSimParameters(paramsFileName);
}
