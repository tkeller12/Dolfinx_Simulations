import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add('Waveguide Iris')

factory = gmsh.model.occ

a = 1
b = 0.5
d = 1.5

iris_width = 0.4
iris_length = 0.05

waveguide_length = 0.5


resonator = factory.addBox(-b/2, -d/2, -a/2, b, d, a)

iris = factory.addBox(-b/2, -d/2, -iris_width/2, b, -iris_length, iris_width)

waveguide = factory.addBox(-b/2, -d/2 - iris_length , -a/2, b, -waveguide_length, a)


out = factory.fuse([(3, resonator)], [(3, iris),(3, waveguide)])
#print(out)
result = out[0][0][1]
#print(result)

factory.synchronize()
gmsh.model.addPhysicalGroup(3, [result], tag = 1, name = 'resonator')
gmsh.model.addPhysicalGroup(2, [11], tag = 2, name = 'waveguide port')
factory.synchronize()

gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)
gmsh.model.mesh.refine()
gmsh.model.mesh.refine()


gmsh.write("TE102_test001.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()


