import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("DielectricResonator")

factory = gmsh.model.occ

resonatorOD = 40e-3
resonatorHeight = 60e-3

lc = 0.002 # very fine mesh
#lc = 0.005


resonator = factory.addCylinder(0,0,-resonatorHeight/2,0,0,resonatorHeight,resonatorOD/2)

factory.synchronize()

print(resonator)

gmsh.model.addPhysicalGroup(3, [resonator], 1, "resonator")

# Set mesh order (2 = quadratic tetrahedra)
gmsh.option.setNumber("Mesh.ElementOrder", 2)

# Set maximum element size
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

# Mesh smoothing (number of iterations)
gmsh.option.setNumber("Mesh.Smoothing", 10)

gmsh.model.mesh.generate(3)
#gmsh.model.mesh.refine()

filename = "cylindrical_resonator_test001.msh"
gmsh.write(filename)

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()


