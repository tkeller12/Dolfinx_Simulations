import numpy as np

import gmsh

gmsh.initialize()

gmsh.model.add("2D_Rectangle")

factory = gmsh.model.occ

filename = 'halbach_2d_001'

DEFAULT_MESH_SIZE = 0.1
FINE_MESH_SIZE = 0.001
magnet_mesh_size = 0.005
box_size = 1.0 # 2x box size is full boundary edge
r = 0.5
magnet_size = 0.070

#total_magnets = 32
total_magnets = 8

#if (total_magnets % 4) !=0:
#    raise ValueError('total magnets must be divisible by 4')

def addRectangle(x, y, z, dx, dy, theta = 0., theta_2 = 0., meshSize = DEFAULT_MESH_SIZE):
    if type(meshSize) is list:
        p1 = factory.addPoint(x, y, z, meshSize[0])
        p2 = factory.addPoint(x+dx, y, z, meshSize[1])
        p3 = factory.addPoint(x+dx, y+dy, z, meshSize[2])
        p4 = factory.addPoint(x, y+dy, z, meshSize[3])
    else:
        p1 = factory.addPoint(x, y, z, meshSize)
        p2 = factory.addPoint(x+dx, y, z, meshSize)
        p3 = factory.addPoint(x+dx, y+dy, z, meshSize)
        p4 = factory.addPoint(x, y+dy, z, meshSize)

    lines = []

    lines.append(factory.addLine(p1,p2))
    lines.append(factory.addLine(p2,p3))
    lines.append(factory.addLine(p3,p4))
    lines.append(factory.addLine(p4,p1))


    loop = factory.addCurveLoop(lines)
    rect = factory.addPlaneSurface([loop])

    factory.rotate([(2,rect)],0, 0, 0, 0, 0, 1, theta * np.pi / 180.)

    center_of_mass = gmsh.model.occ.getCenterOfMass(2, rect)
    center_x = center_of_mass[0]
    center_y = center_of_mass[1]
    factory.rotate([(2,rect)],center_x, center_y, 0, 0, 0, 1, theta_2 * np.pi / 180.)


    return rect




boundary = addRectangle(-box_size,-box_size,0,2*box_size, 2*box_size, meshSize = [DEFAULT_MESH_SIZE, DEFAULT_MESH_SIZE, DEFAULT_MESH_SIZE, DEFAULT_MESH_SIZE])

#center_point = factory.addPoint(0,0,0, FINE_MESH_SIZE) # doesn't work

magnets = total_magnets
index = range(magnets)
#print(index)
#print(index[-1])

magnet_tags = []
M_angle = []
for each in index:
    print(each)
    theta = each*(360/(magnets))
    M_angle.append(theta*2.0)
    print(theta)
    rect = addRectangle(r-magnet_size/2, -magnet_size/2, 0, magnet_size, magnet_size, theta = theta, theta_2 = theta, meshSize = magnet_mesh_size)

    factory.cut([(2,boundary)], [(2, rect)], removeTool = False)
    magnet_tags.append(rect)
print(M_angle)
print(magnet_tags)
M_x = [0]
M_y = [0]
for ix, theta in enumerate(M_angle):
    print(magnet_tags[ix], theta)
#    print(np.sin(M_value * np.pi / 180.), np.cos(M_value * np.pi / 180.))
    print(np.sin(theta * np.pi / 180.), np.cos(theta * np.pi / 180.))
    M_x.append(np.sin(theta * np.pi / 180.))
    M_y.append(-1.0*np.cos(theta * np.pi / 180.))

#factory.addPoint(0,0,0, FINE_MESH_SIZE) # doesn't Work
factory.synchronize()


#gmsh.model.mesh.recombine()

gmsh.model.addPhysicalGroup(2, [boundary], boundary)
permeability_list = [1]
for ix, tag in enumerate(magnet_tags):
    gmsh.model.addPhysicalGroup(2, [tag], tag)
    permeability_list.append(1.05) # 1.05 for neodymium magnet

factory.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize()
all_tags = [boundary] + magnet_tags

### Create Array for saving ###
# physical group, permeability, magnetization x, magnetization y
save_array = np.vstack((all_tags, permeability_list, M_x, M_y)).T

print(save_array)

np.savetxt(filename + '.csv', save_array, delimiter = ',')
gmsh.write(filename + '.msh')

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()



