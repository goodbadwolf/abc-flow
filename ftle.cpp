#include "ftle.h"
#include <cmath>
#include <execution>
#include <iostream>
#include <vtkXMLImageDataWriter.h>
#include <vtkRectilinearGrid.h>
#include <vtkRectilinearGridWriter.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCallbackCommand.h>
#include <vtkSmartPointer.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkScalarBarActor.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>

void KeypressCallbackFunction(vtkObject *caller, long unsigned int eventId, void *clientData, void *callData)
{
    static int counter = 0; // For saving with unique names
    auto computer = static_cast<FTLEComputer *>(clientData);
    auto grid = computer->getGrid();
    auto interactor = static_cast<vtkRenderWindowInteractor *>(caller);

    if (interactor->GetKeySym() == std::string("s"))
    {
        std::cout << "Using format: " << computer->getFormat() << std::endl;
        std::string filename = "ftle_output_" + std::to_string(counter++) + (computer->getFormat() == "vtk" ? ".vti" : ".vdb");
        if (computer->getFormat() == "vtk")
        {
            computer->saveToVTK(filename);
        }
        else
        {
            computer->saveToVDB(filename);
        }
        std::cout << "Saved file: " << filename << std::endl;
        /*
        std::string filename = "ftle_output_" + std::to_string(counter++) + ".vti";
        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetFileName(filename.c_str());
        writer->SetInputData(grid);
        writer->Write();
        std::cout << "Saved file: " << filename << std::endl;
        */
    }
}

FTLEComputer::FTLEComputer(int res, std::string format) : resolution(res), format(format), A(1.0), B(1.0), C(1.0)
{
    cellSpacing = 2.0 * M_PI / (resolution - 1);

    // Initialize VTK grid
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

    for (int k = 0; k < resolution; ++k)
    {
        for (int j = 0; j < resolution; ++j)
        {
            for (int i = 0; i < resolution; ++i)
            {
                particles.emplace_back(
                    i * cellSpacing,
                    j * cellSpacing,
                    k * cellSpacing);
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

    // Parallel execution of particle advection
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
    // Seed particles
    auto [particles, seedTime] = measureTime(&FTLEComputer::seedParticles, this);
    std::cout << "Particle seeding time: " << seedTime << " seconds\n";

    // Forward FTLE computation
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

    // Backward FTLE computation
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

    // Add FTLE data to grid
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
    grid->GetPointData()->SetActiveScalars("FFTLE"); // Ensure active scalar is set
}

void FTLEComputer::saveToVDB(const std::string &filename)
{
    openvdb::initialize();

    // Create grids for forward and backward FTLE
    auto fftle_grid = openvdb::FloatGrid::create();
    auto bftle_grid = openvdb::FloatGrid::create();

    fftle_grid->setName("FFTLE");
    bftle_grid->setName("BFTLE");

    // Create transform
    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(cellSpacing);
    fftle_grid->setTransform(transform);
    bftle_grid->setTransform(transform);

    // Create dense grids from the data
    openvdb::tools::Dense<float>
        fftle_dense(openvdb::Coord(resolution, resolution, resolution));
    openvdb::tools::Dense<float>
        bftle_dense(openvdb::Coord(resolution, resolution, resolution));

    // Copy data to dense grids
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

    // Convert dense grids to sparse VDB grids
    openvdb::tools::copyFromDense(fftle_dense, *fftle_grid, 0.0f, true);
    openvdb::tools::copyFromDense(bftle_dense, *bftle_grid, 0.0f, true);

    // Create file and write grids
    openvdb::GridPtrVec grids;
    grids.push_back(fftle_grid);
    grids.push_back(bftle_grid);

    openvdb::io::File file(filename);
    file.write(grids);
    file.close();

    std::cout << "Saved OpenVDB file to: " << filename << "\n";
    std::cout << "Grid statistics:\n";
    std::cout << "FFTLE grid active voxels: " << fftle_grid->activeVoxelCount() << "\n";
    std::cout << "BFTLE grid active voxels: " << bftle_grid->activeVoxelCount() << "\n";
}

void FTLEComputer::saveToVTK(const std::string &filename)
{
    vtkSmartPointer<vtkXMLImageDataWriter> writer =
        vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(grid);
    writer->Write();
}

void FTLEComputer::setABCParameters(double a, double b, double c)
{
    A = a;
    B = b;
    C = c;
}

void FTLEComputer::renderWithSave()
{
    // Check if grid has scalar data
    if (!grid || !grid->GetPointData() || !grid->GetPointData()->GetScalars())
    {
        std::cerr << "Error: No scalar data available for rendering.\n";
        return;
    }

    // Create a mapper and actor
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(grid);
    mapper->SetScalarRange(grid->GetPointData()->GetScalars()->GetRange());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Create a renderer and render window
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);
    renderer->SetBackground(0.1, 0.2, 0.4);
    renderer->ResetCamera();

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(1920, 1080);
    renderWindow->AddRenderer(renderer);

    // Create an interactor
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    renderWindowInteractor->SetInteractorStyle(style);

    // Add scalar bar
    vtkSmartPointer<vtkScalarBarActor> scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar->SetLookupTable(mapper->GetLookupTable());
    scalarBar->SetTitle("FTLE");
    scalarBar->SetNumberOfLabels(4);
    renderer->AddActor2D(scalarBar);

    // Add axes
    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
    vtkSmartPointer<vtkOrientationMarkerWidget> axesWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    axesWidget->SetOutlineColor(0.9300, 0.5700, 0.1300);
    axesWidget->SetOrientationMarker(axes);
    axesWidget->SetInteractor(renderWindowInteractor);
    axesWidget->SetViewport(0.0, 0.0, 0.2, 0.2);
    axesWidget->SetEnabled(1);
    axesWidget->InteractiveOff();

    // Add keypress callback
    vtkSmartPointer<vtkCallbackCommand> keypressCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    keypressCallback->SetCallback(KeypressCallbackFunction);
    keypressCallback->SetClientData(this);
    renderWindowInteractor->AddObserver(vtkCommand::KeyPressEvent, keypressCallback);

    // Start interaction
    renderWindow->Render();
    renderWindowInteractor->Start();
}