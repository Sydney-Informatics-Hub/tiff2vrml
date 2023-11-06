# sconverting array to point cloud and saving as mesh

import numpy as np
import pyvista as pv
import meshio
import argparse
import os
from PIL import Image
import vtk

def arrays_to_point_cloud(arrays, spacing = [1,1,1]):
    """
    Convert a stack of 2D binary numpy arrays into a 3D point cloud.

    :param arrays: A 3D numpy array where each 2D slice along the third axis is a binary mask.
    :return: A numpy array of points.
    """
    points = np.argwhere(arrays)  # Get the indices of points where the array is not zero
    # multiply with spacing in 3D 
    points *= spacing
    return points

def point_cloud_to_mesh(points):
    """
    Convert a point cloud to a mesh using Delaunay 3D triangulation.

    :param points: A numpy array of points.
    :return: A PyVista mesh object.
    """
    cloud = pv.PolyData(points)
    volume = cloud.delaunay_3d(alpha=1.0)
    surface = volume.extract_surface()  # Extract the surface from the UnstructuredGrid
    return surface

def save_mesh(mesh, filename, file_format = 'obj'):
    """
    Save a mesh as a VRML file using meshio.

    :param mesh: A PyVista mesh object.
    :param filename: The output file path.
    """

    if file_format.lower() == 'vrml':
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh)
        plotter.show(auto_close=False) 
        vrml_exporter = vtk.vtkVRMLExporter()
        vrml_exporter.SetFileName(filename)
        vrml_exporter.SetRenderWindow(plotter.ren_win)
        vrml_exporter.Write()
        #vrml_exporter.SetInputData(mesh)
        #vrml_exporter.Write()
        plotter.close()
    else:
        cells = [("triangle", mesh.faces.reshape(-1, 4)[:, 1:])]
        meshio.write_points_cells(
            filename,
            mesh.points,
            cells,
            file_format=file_format
        )

def numpy_stack_to_mesh(array_stack, filename, threshold = 0.9, file_format="obj", grid_spacing = (1, 1, 1)):
    """
    Convert a 3D NumPy array stack to a mesh using PyVista's marching cubes algorithm.

    :param array_stack: A 3D numpy array stack.
    :param filename: The filename to save the mesh.
    :param threshold: The threshold level to apply the marching cubes algorithm.
    :param file_format: Format to save the mesh in (e.g., "stl", "ply", "obj", "vtk").
    :param grid_spacing: The grid spacing in each dimension. Defaults to (1, 1, 1).
    """
    # Create a PyVista grid from the NumPy array stack
    grid = pv.UniformGrid()
    grid.dimensions = np.array(array_stack.shape)
    # For a UniformGrid, the "spacing" attribute defines the distance between adjacent points on the grid
    # If your data is isotropic (equal spacing in all directions), it can be left as the default (1.0, 1.0, 1.0)
    # If you have different spacing, you would set it here
    grid.spacing = grid_spacing  # (optional) Adjust this if your data has non-uniform spacing
    
    # Flatten the array_stack and assign it to the grid
    grid.point_data["values"] = array_stack.flatten(order="F")  # Use Fortran order to match PyVista's indexing
    
    # Run the marching cubes algorithm
    contours = grid.contour([threshold])

    # Save the mesh to a file
    save_mesh(contours, filename, file_format)
    # cells = [("triangle", contours.faces.reshape(-1, 4)[:, 1:])]
    # meshio.write_points_cells(
    #     filename,
    #     contours.points,
    #     cells,
    #     file_format=file_format
    # )


def main(array_stack, filename, file_format="obj"):
    points = arrays_to_point_cloud(array_stack)
    surface = point_cloud_to_mesh(points)
    save_mesh(surface, filename, file_format)



def tif_images_to_numpy_array_stack(directory, filename_pattern, image_count=None):
    """
    Convert a stack of TIF images into a NumPy array stack.

    :param directory: Directory where the TIF images are located.
    :param filename_pattern: A common filename pattern where an integer is replaced by '{}'.
    :param image_count: The number of images to load.
    :return: A NumPy array stack of the images.
    """
    # Initialize a list to hold the image data
    images = []

    if image_count is None:
        # list all files with the given pattern in directory
        filename_list = [f for f in os.listdir(directory) if filename_pattern.format(0) in f]
        image_count = len(filename_list)

    # Loop over the image numbers and load each image
    for i in range(image_count):
        # Construct the full filename for each image
        filename = os.path.join(directory, filename_pattern.format(i))
        
        # Open the image and convert to a NumPy array
        image = Image.open(filename)
        image_array = np.array(image)
        
        # Append the image data to the list
        images.append(image_array)
    
    # Stack the individual image arrays into a single 3D NumPy array
    return np.stack(images, axis=-1)


def main(directory, filename_pattern, filename_out, file_format="obj", grid_spacing = (1, 1, 1)):
    array_stack = tif_images_to_numpy_array_stack(directory, filename_pattern)
    #points = arrays_to_point_cloud(array_stack)
    #surface = point_cloud_to_mesh(points)   
    #save_mesh(surface, filename, file_format)
    numpy_stack_to_mesh(array_stack, filename_out, threshold = 0.9, file_format=file_format, grid_spacing=grid_spacing)


### test function

# Test function to create synthetic data and test the pipeline
def test_conversion():
    # Create synthetic data: a stack of 3 10x10 arrays with a simple shape
    arrays = np.zeros((10, 10, 3), dtype=np.uint8)
    arrays[3:7, 3:7, :] = 1  # Create a cube in the center

    # Convert arrays to point cloud
    points = arrays_to_point_cloud(arrays)
    
    # Convert point cloud to mesh
    surface = point_cloud_to_mesh(points)
    
    # Save mesh as VRML
    mesh_filename = 'synthetic_data_mesh.obj'
    save_mesh(surface, mesh_filename, file_format='obj')

    return mesh_filename

def test_tif_images_to_numpy_array_stack():
    directory = 'data'
    filename_pattern = '{}.tif'
    image_count = 5
    array_stack = tif_images_to_numpy_array_stack(directory, filename_pattern, image_count)



def test_main():
    directory = 'data'
    filename_pattern = '{}.tif'
    #filename_out = 'cellmesh_test.obj'
    filename_out = 'cellmesh_test.wrl'
    file_format="vrml"
    grid_spacing = (1, 1, 1)
    main(directory, filename_pattern, filename_out, file_format=file_format, grid_spacing = grid_spacing)


