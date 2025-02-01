import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("2D_Rectangle")

factory = gmsh.model.occ

def addMultiPointLine(startTag, endTag, N = 10):
    if N <= 2:
        line = factory.addLine(startTag, endTag)
#        factory.setTransfiniteCurve(line, 10)
        gmsh.model.geo.mesh.setTransfiniteCurve(line, 20)
        return [line]

    else:
        factory.synchronize()
        x0, y0, z0 = gmsh.model.getValue(0, startTag, [])
        x1, y1, z1 = gmsh.model.getValue(0, endTag, [])

        dx = (x1 - x0) / (N - 1)
        dy = (y1 - y0) / (N - 1)
        dz = (z1 - z0) / (N - 1)

        points_list = [startTag]
        for ix in range(N-2):
            point = factory.addPoint(x0+(ix+1)*dx,y0+(ix+1)*dy,z0+(ix+1)*dz)
            points_list.append(point)
        points_list.append(endTag)

        lines_list = []
        for ix in range(N-1):
            line = factory.addLine(points_list[ix], points_list[ix+1])
            lines_list.append(line)

        return lines_list

#        coord = gmsh.model.mesh.getNode(startTag)
#        dx = (x2-x1) / (N-1)
#
#    return line

def addMultiPointRectangle(x, y, z, dx, dy, N_x = 10, N_y = 10):
    p0 = factory.addPoint(x,y,z)
    p1 = factory.addPoint(x+dx,y,z)
    p2 = factory.addPoint(x+dx,y+dy,z)
    p3 = factory.addPoint(x,y+dy,z)

    lines = addMultiPointLine(p0, p1, N = N_x)
    lines += addMultiPointLine(p1, p2, N = N_y)
    lines += addMultiPointLine(p2, p3, N = N_x)
    lines += addMultiPointLine(p3, p0, N = N_y)

    rect = factory.addCurveLoop(lines)

    return rect

curve = addMultiPointRectangle(0, 0, 0, 1, 2, N_x = 20, N_y = 20)
plane = factory.addPlaneSurface([curve])

print(plane)
circle = factory.addCircle(0,0,0, 1)
print(circle)
circle_loop = factory.addCurveLoop([circle])
circle_plane = factory.addPlaneSurface([circle_loop])


#new_plane = factory.fuse([(2,plane)],[(2,circle_plane)])
#new_plane = factory.fuse([(2,circle_plane)],[(2,plane)])



#extrude = factory.extrude([(2,plane)], 0, 0, 1, [5])
extrude = factory.extrude([(2,plane)], 0, 0, 1)
print(extrude)
#circle_extrude = factory.extrude([(2,circle_plane)], 0, 0, 1, [5])
circle_extrude = factory.extrude([(2,circle_plane)], 0, 0, 1)
print(circle_extrude)

#fused = factory.fuse([(3,extrude)],[(3,circle_extrude)])
#fused = factory.fuse([(3,1)],[(3,2)])
#print(fused)
fragment = factory.fragment([(3,1)],[(3,2)])
print(fragment)

factory.synchronize()


#gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)
#gmsh.model.mesh.generate(2)

#
#x1 = 0
#x2 = 1
#
#pts = 11

#dx = (x2-x1) / (pts-1)
#
#points_list = []
#for ix in range(pts):
#    point = factory.addPoint(ix * dx, 0, 0)
#    print(point)
    
#    coord = gmsh.model.getValue(0, point, [])
#    xp,yp,zp = gmsh.model.getValue(0, point, [])
#    print(xp,yp,zp)
#    print(coord)
#    points_list.append(point)

#for ix in range(pts-1):
#    line = factory.addLine(points_list[ix],points_list[ix+1])
#
factory.synchronize()

#gmsh.write("line_test.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()
