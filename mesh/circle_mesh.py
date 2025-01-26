import gmsh

factory = gmsh.model.occ

def generate_circle_mesh(radius=1.0, num_points=32, mesh_size=0.1):
    # Initialize GMSH
    gmsh.initialize()

    # Create a new model
    gmsh.model.add("Circle")

    # Define the circle geometry (center at origin, radius = radius)
    factory.addCircle(0, 0, 0, radius)

    # Define the mesh size at the points
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    # Synchronize the model (prepare for meshing)
    factory.synchronize()

    # Generate the 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh to a file
    gmsh.write("circle.msh")

    # Optionally, you can visualize the mesh
    gmsh.fltk.run()

    # Finalize GMSH
    gmsh.finalize()

if __name__ == "__main__":
    generate_circle_mesh(radius=1.0, num_points=32, mesh_size=0.1)
