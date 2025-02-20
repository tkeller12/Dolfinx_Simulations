import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("2D_Rectangle")

factory = gmsh.model.occ

box_size = 2.0


boundary = factory.addBox(0, -box_size/2, -box_size/2, box_size/2.,box_size,box_size)

yoke = factory.addBox(0.50, -0.8, -0.3, 0.2,1.6,0.6)
yoke_top = factory.addBox(0, 0.6, -0.3, 0.7,0.2,0.6)
yoke_bottom = factory.addBox(0, -0.6, -0.3, 0.7,-0.2,0.6)

coil_rect = factory.addRectangle(0.15, 0.35, 0, 0.3, 0.2)
pole_rect = factory.addRectangle(0.1, 0.0, 0, 0.4, 0.3)

revolve_tags = factory.revolve([(2, coil_rect)], 0,0,0, 1, 0, 0, 2*np.pi)
for each in revolve_tags:
    if each[0] == 3:
        coil = each[1]
print(coil)

revolve_pole_tags = factory.revolve([(2, pole_rect)], 0,0,0, 1, 0, 0, 2*np.pi)
for each in revolve_pole_tags:
    if each[0] == 3:
        pole = each[1]

out = factory.fuse([(3, pole)], [(3, yoke), (3, yoke_top), (3, yoke_bottom)])
print(out)
for each in out:
    if each[0] == 3:
        yoke = each[1]
yoke = out[0][0][1]
print(yoke)


out = factory.cut([(3,boundary)], [(3, coil)], removeTool = False)
#out = factory.cut([(3,boundary)], [(3, pole)], removeTool = False)
out = factory.cut([(3,boundary)], [(3, yoke)], removeTool = False)

factory.synchronize()
gmsh.model.addPhysicalGroup(3, [boundary], 1)
gmsh.model.addPhysicalGroup(3, [coil], 2)
#gmsh.model.addPhysicalGroup(3, [pole], 3)
gmsh.model.addPhysicalGroup(3, [yoke], 3)


factory.synchronize()


#gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)
gmsh.model.mesh.refine()
gmsh.model.mesh.refine()

#gmsh.model.addPhysicalGroup(3, [boundary], 1)
#gmsh.model.addPhysicalGroup(3, [coil], 2)

filename = "electromagnet_3d_test001.msh"
gmsh.write(filename)

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()
