import gmsh

# Initialize Gmsh
gmsh.initialize()

# Create a model and set its name
model = gmsh.model
model.add("Square")

# Create points
p1 = model.geo.addPoint(0, 0, 0, 1)
p2 = model.geo.addPoint(1, 0, 0, 1)
p3 = model.geo.addPoint(1, 1, 0, 1)
p4 = model.geo.addPoint(0, 1, 0, 1)

# Create lines
l1 = model.geo.addLine(p1, p2)
l2 = model.geo.addLine(p2, p3)
l3 = model.geo.addLine(p3, p4)
l4 = model.geo.addLine(p4, p1)

# Create a curve loop
curveLoop = model.geo.addCurveLoop([l1, l2, l3, l4])

# Create a surface
surface = model.geo.addPlaneSurface([curveLoop])

# Synchronize the model
model.geo.synchronize()

# Generate the mesh
model.mesh.generate(2)

# Save the mesh
gmsh.write("square.msh")

# Finalize Gmsh
gmsh.finalize()
