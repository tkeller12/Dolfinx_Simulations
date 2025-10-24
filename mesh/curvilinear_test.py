import gmsh

gmsh.initialize()
gmsh.model.add("second_order_cylinder")

# --- Parameters ---
r = 0.5       # radius
h = 2.0       # height
lc = 0.1      # mesh size

# --- Create a cylinder ---
# Cylinder base at (0,0,0), axis along z, radius r, height h
cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, h, r)

# Synchronize to make geometry entities available in the model
gmsh.model.occ.synchronize()

# --- Define mesh options ---
# Use second order (quadratic) elements
gmsh.model.mesh.setOrder(2)

# Optionally control element type (tetrahedra)
gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)  # smooth curved elements

# --- Generate 3D mesh ---
gmsh.model.mesh.generate(3)

# --- Save mesh ---
gmsh.write("second_order_cylinder.msh")

# --- Optional GUI ---
#if '-nopopup' not in gmsh.option.getString("General.Terminal"):
gmsh.fltk.run()

gmsh.finalize()
