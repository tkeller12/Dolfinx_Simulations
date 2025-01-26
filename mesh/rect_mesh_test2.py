
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
p1 = gmsh.model.geo.addPoint(x_min, y_min, 0.0)
p2 = gmsh.model.geo.addPoint(x_max, y_min, 0.0)
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

# Define the center point for mesh refinement
center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2

# Create a distance field from the center of the rectangle
field_tag = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(field_tag, "PointsList", [p1, p2, p3, p4])
gmsh.model.mesh.field.setNumber(field_tag, "Distance", 1.0)
gmsh.model.mesh.field.setNumber(field_tag, "Scaling", 1.0)

# Create a threshold field to apply different mesh sizes depending on distance to center
threshold_field_tag = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field_tag, "IField", field_tag)
gmsh.model.mesh.field.setNumber(threshold_field_tag, "LcMin", 0.2)  # Minimum mesh size (fine near the center)
gmsh.model.mesh.field.setNumber(threshold_field_tag, "LcMax", 2.0)  # Maximum mesh size (coarse away from the center)
gmsh.model.mesh.field.setNumber(threshold_field_tag, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(threshold_field_tag, "DistMax", 5.0)

# Apply the field to control mesh size
gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field_tag)

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
