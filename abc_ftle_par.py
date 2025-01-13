import argparse
import vtk
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import sys

# Constants for the ABC flow
A = 1.0
B = 1.0
C = 1.0
omega = 0.1


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result, execution_time

    return wrapper


# Time-dependent ABC flow equations
def abc_flow(x, y, z, t=0):
    u = A * np.sin(z + omega * t) + C * np.cos(y + omega * t)
    v = B * np.sin(x + omega * t) + A * np.cos(z + omega * t)
    w = C * np.sin(y + omega * t) + B * np.cos(x + omega * t)
    return np.array([u, v, w])


def unsteady_abc_flow(x, y, z, t=0):
    uA = 1.7320 + 0.5 * t * np.sin(math.pi * t)
    uB = 1.4142
    uC = 1.0

    u = uA * np.sin(z) + uB * np.cos(y)
    v = uB * np.sin(x) + uC * np.cos(z)
    w = uC * np.sin(y) + uA * np.cos(x)
    return np.array([u, v, w])


# Create a uniform grid
def create_uniform_grid(dimensions, spacing):
    grid = vtk.vtkImageData()
    grid.SetDimensions(dimensions)
    grid.SetSpacing(spacing)
    return grid


# Seed particles at each grid point
@timer
def seed_particles(grid):
    dimensions = grid.GetDimensions()
    spacing = grid.GetSpacing()
    origin = grid.GetOrigin()

    particles = []
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            for k in range(dimensions[2]):
                x = origin[0] + i * spacing[0]
                y = origin[1] + j * spacing[1]
                z = origin[2] + k * spacing[2]
                particles.append([x, y, z])
    return np.array(particles)


def advect_single_particle(args):
    particle, t0, tf, dt, forward = args
    num_steps = int(abs(tf - t0) / dt)
    for step in range(num_steps):
        t = t0 + step * dt if forward else t0 - step * dt
        particle += unsteady_abc_flow(*particle, t) * dt
    return particle


# Advect particles through the flow - parallel version
@timer
def advect_particles_parallel(particles, t0, tf, dt, forward=True, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    args_list = [(particle.copy(), t0, tf, dt, forward) for particle in particles]

    pool = multiprocessing.Pool(num_workers)
    advected_particles = list(pool.map(advect_single_particle, args_list))

    return np.array(advected_particles)


def compute_ftle_single(args):
    initial_particle, advected_particle, dt = args
    dF = np.linalg.norm(advected_particle - initial_particle)
    return (1 / dt) * np.log(dF)


# Compute FTLE - parallel version
@timer
def compute_ftle_parallel(initial_particles, advected_particles, dt, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    args_list = [
        (initial_particles[i], advected_particles[i], dt)
        for i in range(len(initial_particles))
    ]

    pool = multiprocessing.Pool(num_workers)
    ftle = list(pool.map(compute_ftle_single, args_list))

    return np.array(ftle)


def save_grid_to_vtk(grid, filename):
    def create_rectilinear_grid(image_data):
        dimensions = image_data.GetDimensions()
        origin = image_data.GetOrigin()
        spacing = image_data.GetSpacing()

        x_coords = vtk.vtkDoubleArray()
        y_coords = vtk.vtkDoubleArray()
        z_coords = vtk.vtkDoubleArray()

        for i in range(dimensions[0]):
            x_coords.InsertNextValue(origin[0] + i * spacing[0])
        for j in range(dimensions[1]):
            y_coords.InsertNextValue(origin[1] + j * spacing[1])
        for k in range(dimensions[2]):
            z_coords.InsertNextValue(origin[2] + k * spacing[2])

        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(dimensions)
        grid.SetXCoordinates(x_coords)
        grid.SetYCoordinates(y_coords)
        grid.SetZCoordinates(z_coords)
        grid.GetPointData().DeepCopy(image_data.GetPointData())

        return grid

    def write_rectilinear_grid(grid, output_file):
        writer = vtk.vtkRectilinearGridWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(grid)
        writer.SetFileTypeToASCII()
        writer.Write()

    rectilinear_grid = create_rectilinear_grid(grid)
    write_rectilinear_grid(rectilinear_grid, filename)


def create_isosurface(grid, scalar_name, isovalue):
    contour = vtk.vtkContourFilter()
    contour.SetInputData(grid)
    contour.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_name
    )
    contour.SetValue(0, isovalue)
    contour.Update()

    return contour.GetOutput()


def visualize_ftle(grid):
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(grid)
    mapper.SetScalarRange(grid.GetPointData().GetScalars().GetRange())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)

    # Add scalar bar (legend)
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(mapper.GetLookupTable())
    scalar_bar.SetTitle("FTLE")
    scalar_bar.SetNumberOfLabels(4)

    # Adjust size and appearance
    scalar_bar.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    scalar_bar.SetPosition(0.05, 0.4)
    scalar_bar.SetWidth(0.2)  # Width as a fraction of the window width
    scalar_bar.SetHeight(0.5)  # Height as a fraction of the window height

    renderer.AddActor2D(scalar_bar)

    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(1.0, 1.0, 1.0)
    axes.SetShaftTypeToCylinder()
    axes.SetCylinderRadius(0.01)
    axes.SetConeRadius(0.1)

    orientation_marker = vtk.vtkOrientationMarkerWidget()
    orientation_marker.SetOutlineColor(0.9300, 0.5700, 0.1300)
    orientation_marker.SetOrientationMarker(axes)
    orientation_marker.SetInteractor(interactor)
    orientation_marker.SetViewport(
        0.0, 0.0, 0.2, 0.2
    )  # Position in the bottom left corner
    orientation_marker.SetEnabled(True)
    orientation_marker.InteractiveOff()

    # Fit camera to all actors
    renderer.ResetCamera()

    render_window.Render()
    interactor.Start()


def visualize_ftle_vol(grid):
    grid.GetPointData().SetActiveScalars("FFTLE")

    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(grid)
    volume_mapper.SetBlendModeToComposite()
    volume_mapper.SetScalarModeToUsePointData()

    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    # Create color transfer function
    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 1.0)  # Blue for low values
    color_transfer_function.AddRGBPoint(0.5, 0.0, 1.0, 0.0)  # Green for mid values
    color_transfer_function.AddRGBPoint(1.0, 1.0, 0.0, 0.0)  # Red for high values
    volume_property.SetColor(color_transfer_function)

    # Create opacity transfer function
    opacity_transfer_function = vtk.vtkPiecewiseFunction()

    def add_point(p, o):
        print(f"Adding point: {p:>.2f}, {o:>.2f}")
        opacity_transfer_function.AddPoint(p, o)

    points = [(0.1, 1.0), (0.5, 1.0), (0.9, 1.0)]
    add_point(0.00, 0.0)
    for p, o in points:
        add_point(p - 0.01, 0.0)
        add_point(p, o)
        add_point(p + 0.1, o)
        add_point(p + 0.11, 0.0)
    add_point(1.00, 0.0)
    volume_property.SetScalarOpacity(opacity_transfer_function)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.2, 0.4)

    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)

    axes = vtk.vtkAxesActor()
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(interactor)
    axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    axes_widget.SetEnabled(1)
    axes_widget.InteractiveOff()

    # Add a color bar
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(color_transfer_function)
    scalar_bar.SetTitle("FTLE")
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetPosition(0.85, 0.05)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.9)
    renderer.AddActor2D(scalar_bar)

    # Fit camera to all actors
    renderer.ResetCamera()

    # Render and start interaction
    render_window.Render()
    interactor.Start()


def visualize_ftle_iso(grid):
    fftle = grid.GetPointData().GetArray("FFTLE")
    bftle = grid.GetPointData().GetArray("BFTLE")
    fftle_isovalue = np.percentile(fftle, 90)
    bftle_isovalue = np.percentile(bftle, 90)

    fftle_iso = create_isosurface(grid, "FFTLE", fftle_isovalue)
    bftle_iso = create_isosurface(grid, "BFTLE", bftle_isovalue)

    fftle_mapper = vtk.vtkPolyDataMapper()
    fftle_mapper.SetInputData(fftle_iso)
    fftle_mapper.SetScalarRange(fftle.GetRange())
    fftle_mapper.ScalarVisibilityOff()

    bftle_mapper = vtk.vtkPolyDataMapper()
    bftle_mapper.SetInputData(bftle_iso)
    bftle_mapper.SetScalarRange(bftle.GetRange())
    bftle_mapper.ScalarVisibilityOff()

    fftle_actor = vtk.vtkActor()
    fftle_actor.SetMapper(fftle_mapper)
    fftle_actor.GetProperty().SetColor(1, 0, 0)  # Red for forward FTLE
    fftle_actor.GetProperty().SetOpacity(0.5)

    bftle_actor = vtk.vtkActor()
    bftle_actor.SetMapper(bftle_mapper)
    bftle_actor.GetProperty().SetColor(0, 0, 1)  # Blue for backward FTLE
    bftle_actor.GetProperty().SetOpacity(0.5)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(fftle_actor)
    renderer.AddActor(bftle_actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # RGB background color

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)  # Set window size

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)

    axes = vtk.vtkAxesActor()
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(interactor)
    axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    axes_widget.SetEnabled(1)
    axes_widget.InteractiveOff()

    # Add a color bar
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(fftle_mapper.GetLookupTable())
    scalar_bar.SetTitle("FTLE")
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetPosition(0.85, 0.05)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.9)
    renderer.AddActor2D(scalar_bar)

    # Fit camera to all actors
    renderer.ResetCamera()

    # Render and start interaction
    render_window.Render()
    interactor.Start()


# Main function to compute FTLE and visualize it
def main():
    parser = argparse.ArgumentParser(description="Compute FTLE for ABC flow")
    parser.add_argument("--input", type=str, help="Input VTK file name")
    parser.add_argument("--res", type=int, default=64, help="Grid resolution")
    parser.add_argument(
        "--vis",
        type=str,
        choices=["none", "iso", "vol"],
        default="none",
        help="Visualization option: none, iso (isosurface), or volume",
    )
    args = parser.parse_args()

    if args.input:
        vtk_file_name = args.input
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(vtk_file_name)
        reader.Update()
        grid = reader.GetOutput()
    else:
        # Grid parameters
        resolution = args.res
        cell_spacing = (2 * math.pi) / (resolution - 1)
        dimensions = (resolution, resolution, resolution)
        spacing = (cell_spacing, cell_spacing, cell_spacing)

        # Create uniform grid
        grid = create_uniform_grid(dimensions, spacing)

        # Seed particles
        initial_particles, seed_time = seed_particles(grid)

        # Advect particles
        t0_forward = 0.0
        tf_forward = 10.0
        dt = 0.1

        advected_particles_fwd, advect_fwd_time = advect_particles_parallel(
            initial_particles, t0_forward, tf_forward, dt, forward=True
        )

        # Compute FTLE
        fftle, fftle_time = compute_ftle_parallel(
            initial_particles, advected_particles_fwd, tf_forward - t0_forward
        )

        # Add FTLE to the grid
        fftle_array = vtk.vtkDoubleArray()
        fftle_array.SetName("FFTLE")
        fftle_array.SetNumberOfComponents(1)
        fftle_array.SetNumberOfTuples(grid.GetNumberOfPoints())

        for i in range(grid.GetNumberOfPoints()):
            fftle_array.SetValue(i, fftle[i])

        grid.GetPointData().AddArray(fftle_array)
        grid.GetPointData().SetActiveScalars("FFTLE")

        t0_backward = 10.0
        tf_backward = 0.0

        advected_particles_bwd, advect_bwd_time = advect_particles_parallel(
            initial_particles, t0_backward, tf_backward, dt, forward=False
        )

        # Compute FTLE
        bftle, bftle_time = compute_ftle_parallel(
            initial_particles, advected_particles_bwd, tf_backward - t0_backward
        )

        # Add FTLE to the grid
        bftle_array = vtk.vtkDoubleArray()
        bftle_array.SetName("BFTLE")
        bftle_array.SetNumberOfComponents(1)
        bftle_array.SetNumberOfTuples(grid.GetNumberOfPoints())

        for i in range(grid.GetNumberOfPoints()):
            bftle_array.SetValue(i, bftle[i])

        grid.GetPointData().AddArray(bftle_array)
        grid.GetPointData().SetActiveScalars("BFTLE")

        # Save the grid to a binary VTK file
        save_grid_to_vtk(grid, "ftle_grid.vtk")

        # Print timing summary
        print("\nTiming Summary:")
        print(f"Particle Seeding Time: {seed_time:.2f} seconds")
        print(f"Forward Advection Time: {advect_fwd_time:.2f} seconds")
        print(f"Backward Advection Time: {advect_bwd_time:.2f} seconds")
        print(f"Forward FTLE Computation Time: {fftle_time:.2f} seconds")
        print(f"Backward FTLE Computation Time: {bftle_time:.2f} seconds")
        print(
            f"Total Computation Time: {seed_time + advect_fwd_time + advect_bwd_time + fftle_time + bftle_time:.2f} seconds"
        )

    if args.vis == "iso":
        visualize_ftle_iso(grid)
    elif args.vis == "vol":
        visualize_ftle_vol(grid)


# Run the main function
if __name__ == "__main__":
    main()
