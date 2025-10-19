import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("DielectricResonator")

factory = gmsh.model.occ

dielectricOD = 12e-3
dielectricID = 0
dielectricHeight = 6e-3

resonatorOD = 30e-3
resonatorHeight = 20e-3


dielectric = factory.addCylinder(0,0,-dielectricHeight/2,0,0,dielectricHeight, dielectricOD/2)
resonator = factory.addCylinder(0,0,-resonatorHeight/2,0,0,resonatorHeight,resonatorOD/2)
result_tags, out_dim_tags_map = factory.cut([(3,resonator)], [(3, dielectric)], removeTool = False)


factory.synchronize()

print(dielectric)
print(resonator)
print(result_tags)
print('here')

gmsh.model.addPhysicalGroup(3, [resonator], 1, "resonator")
gmsh.model.addPhysicalGroup(3, [dielectric], 2, "dielectric")



#gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)

filename = "dielectric_resonator_test001.msh"
gmsh.write(filename)

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()


