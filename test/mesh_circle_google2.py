import gmsh

gmsh.initialize()

# Create a 2D circle
#circle = gmsh.model.occ.addCircle(0, 0, 0, 1)
circle = gmsh.model.occ.addCircleArc(0, 0, 0, 1)

# Create a surface from the circle
surface = gmsh.model.occ.addPlaneSurface([circle])

# Synchronize the model
gmsh.model.occ.synchronize()

# Generate the mesh
gmsh.model.mesh.generate(2)

# Save the mesh
gmsh.write("mesh_circle.msh")

gmsh.finalize()
