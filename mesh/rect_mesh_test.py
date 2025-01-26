import gmsh
import math

# Initialize GMSH
gmsh.initialize()

# Define the model
gmsh.model.add("2D_Rectangle")

# Define the rectangle's corner points
x_min, x_max = 0.0, 10.0  # x-axis limits
y_min, y_max = 0.0, 5.0   # y-axis limits

# Points for the rectangle's corners
p1 = gmsh.model.geo.addPoint(x_min, y_min, 0.0, 0.1)
p2 = gmsh.model.geo.addPoint(x_max, y_min, 0.0, 0.1)
p3 = gmsh.model.geo.addPoint(x_max, y_max, 0.0)
p4 = gmsh.model.geo.addPoint(x_min, y_max, 0.0)

# Lines for the edges of the rectangle
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

# Create the surface using the lines
l5 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
s1 = gmsh.model.geo.addSurfaceFilling([l5])

# Apply a variable mesh size function
# The `addPoint` function allows defining mesh sizes for specific points
def mesh_size(x, y):
    """Define mesh size as a function of the position (x, y)."""
    # Example: Mesh size is smaller near the center of the rectangle
    return 0.01 + 1 * math.sqrt((x - 5.0) ** 2 + (y - 2.5) ** 2)

# Iterate through the mesh points and set the size accordingly
for x in range(11):  # x values from 0 to 10 (inclusive)
    for y in range(6):  # y values from 0 to 5 (inclusive)
        lc = mesh_size(x,y)
        print(lc)
        point_tag = gmsh.model.geo.addPoint(x, y, 0.0, lc)
        # Optionally, you could refine the mesh further based on specific conditions

# Synchronize the GMSH model with the geometry
gmsh.model.geo.synchronize()

# Generate the 2D mesh
gmsh.model.mesh.generate(2)

# Optionally, save the mesh to a file
gmsh.write("variable_refinement_mesh.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()
