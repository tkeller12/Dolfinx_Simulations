
import gmsh

def create_box_mesh(width, height, depth):
    # Initialize GMSH
    gmsh.initialize()

    # Define the 2D base of the box (a rectangle)
    # Corner points of the rectangle
    p1 = gmsh.model.geo.addPoint(0, 0, 0)  # Point at (0,0,0)
    p2 = gmsh.model.geo.addPoint(width, 0, 0)  # Point at (width,0,0)
    p3 = gmsh.model.geo.addPoint(width, height, 0)  # Point at (width,height,0)
    p4 = gmsh.model.geo.addPoint(0, height, 0)  # Point at (0,height,0)

    l1 = gmsh.model.geo.addLine(p1,p2)
    l2 = gmsh.model.geo.addLine(p2,p3)
    l3 = gmsh.model.geo.addLine(p3,p4)
    l4 = gmsh.model.geo.addLine(p4,p1)

    # Create the rectangle
    rect = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])  # Define the boundary of the rectangle
    surface = gmsh.model.geo.addPlaneSurface([rect])  # Create the surface

    # Extrude the 2D rectangle into a 3D box
    gmsh.model.geo.extrude([(2,surface)], 0, 0, depth)

    # Synchronize the model
    gmsh.model.geo.synchronize()

    # Generate the mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh to a file
    gmsh.write('box_mesh.msh')
    print("Mesh generated and saved as 'box_mesh.msh'")

    # Finalize GMSH
    gmsh.finalize()

# Define the dimensions of the box
width = 2.0
height = 3.0
depth = 5.0

# Create and save the mesh
create_box_mesh(width, height, depth)
