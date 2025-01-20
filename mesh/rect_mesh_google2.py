import gmsh

# Initialize Gmsh
gmsh.initialize()

# Create a 2D square geometry
model = gmsh.model
geo = model.geo

# Add points
p1 = geo.addPoint(0, 0, 0)
p2 = geo.addPoint(1, 0, 0)
p3 = geo.addPoint(1, 1, 0)
p4 = geo.addPoint(0, 1, 0)

# Add lines
l1 = geo.addLine(p1, p2)
l2 = geo.addLine(p2, p3)
l3 = geo.addLine(p3, p4)
l4 = geo.addLine(p4, p1)

# Create a curve loop and a surface
curve_loop = geo.addCurveLoop([l1, l2, l3, l4])
surface = geo.addPlaneSurface([curve_loop])

# Synchronize the model
geo.synchronize()

# Generate the mesh
model.mesh.generate(2)

# Save the mesh
gmsh.write("my_mesh.msh")

# Finalize Gmsh
gmsh.finalize()
