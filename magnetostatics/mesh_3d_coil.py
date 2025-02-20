import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("2D_Rectangle")

factory = gmsh.model.occ

box_size = 2.0

boundary = factory.addBox(-box_size/2, -box_size/2, -box_size/2, box_size,box_size,box_size)

coil_rect = factory.addRectangle(-0.1, 0.3, 0, 0.2, 0.2)
yoke_rect = factory.addRectangle(-0.1, 0.0, 0, 0.2, 0.29)

revolve_tags = factory.revolve([(2, coil_rect)], 0,0,0, 1, 0, 0, 2*np.pi)
for each in revolve_tags:
    if each[0] == 3:
        coil = each[1]
print(coil)

revolve_yoke_tags = factory.revolve([(2, yoke_rect)], 0,0,0, 1, 0, 0, 2*np.pi)
for each in revolve_yoke_tags:
    if each[0] == 3:
        yoke = each[1]

out = factory.cut([(3,boundary)], [(3, coil)], removeTool = False)
out = factory.cut([(3,boundary)], [(3, yoke)], removeTool = False)

factory.synchronize()
gmsh.model.addPhysicalGroup(3, [boundary], 1)
gmsh.model.addPhysicalGroup(3, [coil], 2)
gmsh.model.addPhysicalGroup(3, [yoke], 3)


factory.synchronize()


#gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)
gmsh.model.mesh.refine()

#gmsh.model.addPhysicalGroup(3, [boundary], 1)
#gmsh.model.addPhysicalGroup(3, [coil], 2)

gmsh.write("coil_3d_test001.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()
