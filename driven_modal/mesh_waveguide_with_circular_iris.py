import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add('Waveguide Iris')

factory = gmsh.model.occ

a = 1
b = 0.5
d = 1.5

iris_radius = 0.2
iris_length = 0.02

waveguide_length = 0.5

port_y = -d/2 - iris_length - waveguide_length
print('PORT Y:', port_y)

resonator = factory.addBox(-b/2, -d/2, -a/2, b, d, a)

iris = factory.addCylinder(0.0, -d/2, 0.0, 0.0, -iris_length, 0.0, iris_radius)

waveguide = factory.addBox(-b/2, -d/2 - iris_length , -a/2, b, -waveguide_length, a)


out = factory.fuse([(3, resonator)], [(3, iris),(3, waveguide)])
#print(out)
result = out[0][0][1]
#print(result)

surfaces = factory.getEntities(2)

correct_surface = 13
print('surfaces', surfaces)

tol = 1e-5

factory.synchronize()
for s in surfaces:
#    dist = factory.getDistance(each[0],each[1], 2, correct_surface)
    print(s)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, s[1])
    if (abs(ymin - (port_y)) < tol) and abs(ymax - (port_y)) < tol:
        print('LOCATED PORT:', s[1])
        waveguide_port = s[1]


gmsh.model.addPhysicalGroup(3, [result], tag = 1, name = 'resonator')
gmsh.model.addPhysicalGroup(2, [waveguide_port], tag = 2, name = 'waveguide port')
factory.synchronize()

gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)
gmsh.model.mesh.refine()
#gmsh.model.mesh.refine()


gmsh.write("TE102_test001.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()


