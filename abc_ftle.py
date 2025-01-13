import vtk
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants for the ABC flow
A = 1.0
B = 1.0
C = 1.0
omega = 0.1


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


# Advect particles through the flow - forward or backward
def advect_particles_serial(particles, t0, tf, dt, forward=True):
    num_steps = int(abs(tf - t0) / dt)
    for step in range(num_steps):
        t = t0 + step * dt if forward else t0 - step * dt
        for i in range(len(particles)):
            particles[i] += unsteady_abc_flow(*particles[i], t) * dt
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}")
    return particles


# Compute FTLE
def compute_ftle_serial(initial_particles, advected_particles, dt):
    dimensions = initial_particles.shape
    ftle = np.zeros(dimensions[0])

    for i in range(dimensions[0]):
        dF = np.linalg.norm(advected_particles[i] - initial_particles[i])
        ftle[i] = (1 / dt) * np.log(dF)

    return ftle


# Save grid to a binary VTK file
def save_grid_to_vtk(grid, filename):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.SetDataModeToBinary()
    writer.Write()


# Main function to compute FTLE and visualize it
def main():
    # Grid parameters
    resolution = 32
    cell_spacing = (2 * math.pi) / (resolution - 1)
    dimensions = (resolution, resolution, resolution)
    spacing = (cell_spacing, cell_spacing, cell_spacing)

    # Create uniform grid
    grid = create_uniform_grid(dimensions, spacing)

    # Seed particles
    initial_particles = seed_particles(grid)

    # Advect particles
    t0_forward = 0.0
    tf_forward = 10.0
    dt = 0.1

    advected_particles = advect_particles_serial(
        initial_particles.copy(), t0_forward, tf_forward, dt, forward=True
    )

    # Compute FTLE
    fftle = compute_ftle_serial(
        initial_particles, advected_particles, tf_forward - t0_forward
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

    advected_particles = advect_particles_serial(
        initial_particles.copy(), t0_backward, tf_backward, dt, forward=False
    )

    # Compute FTLE
    bftle = compute_ftle_serial(
        initial_particles, advected_particles, tf_backward - t0_backward
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

    # Visualize the FTLE
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(grid)
    mapper.SetScalarRange(grid.GetPointData().GetScalars().GetRange())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # RGB background color

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Add interaction style
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)

    # Add scalar bar (legend)
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(mapper.GetLookupTable())
    scalar_bar.SetTitle("FTLE")
    scalar_bar.SetNumberOfLabels(4)

    # Adjust size and appearance
    scalar_bar.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    scalar_bar.SetPosition(0.05, 0.4)  # Position in normalized display coordinates
    scalar_bar.SetWidth(0.2)  # Width as a fraction of the window width
    scalar_bar.SetHeight(0.5)  # Height as a fraction of the window height

    renderer.AddActor2D(scalar_bar)

    # Create and configure the orientation marker widget for fixed axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(1.0, 1.0, 1.0)  # Adjust the length of the axes
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


# Run the main function
if __name__ == "__main__":
    main()
