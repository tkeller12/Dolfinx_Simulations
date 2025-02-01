import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("2D_Rectangle")

factory = gmsh.model.occ

sample_loop_radius = 1.5e-3
return_loop_radius = 3.0e-3
gap_width = 0.2e-3
gap_length = 1e-3
height = 5.0e-3

void_thickness = 2e-3
void_radius = sample_loop_radius + gap_length + 2*return_loop_radius + void_thickness
void_height = 5e-3

z = -height/2

def addPolygon(x, y, z, radius=1.0, N = 20):
    """
    Generate a N-sided polygon using GMSH.
    
    Parameters:
    N      : Number of sides (vertices) of the polygon.
    radius : The radius of the polygon's circumcircle.
    """
    # Angle between adjacent vertices
    angle_step = 2 * np.pi / N

    # Create vertices
    points = []
    for i in range(N):
        angle = i * angle_step
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        # Add point to the model
        point = factory.addPoint(x+dx, y+dy, z)
        points.append(point)

    # Create edges (lines between consecutive points)

    lines = []
    for i in range(N):
        line = factory.addLine(points[i], points[(i + 1) % N])
        lines.append(line)

    # Create the loop to form the polygon
#    loop = factory.addCurveLoop([i+1 for i in range(N)])
    loop = factory.addCurveLoop(lines)

    # Create the surface enclosed by the polygon
#    poly = factory.addPlaneSurface([loop])
    return loop#poly

def addMultiPointLine(startTag, endTag, N = 100):
    if N <= 2:
        line = factory.addLine(startTag, endTag)

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


def addMultiPointRectangle(x, y, z, dx, dy, N_x = 200, N_y = 200):
    p0 = factory.addPoint(x,y,z, 1e-3)
    p1 = factory.addPoint(x+dx,y,z, 1e-3)
    p2 = factory.addPoint(x+dx,y+dy,z, 1e-3)
    p3 = factory.addPoint(x,y+dy,z, 1e-3)

    lines = addMultiPointLine(p0, p1, N = N_x)
    lines += addMultiPointLine(p1, p2, N = N_y)
    lines += addMultiPointLine(p2, p3, N = N_x)
    lines += addMultiPointLine(p3, p0, N = N_y)

    rect = factory.addCurveLoop(lines)

    rect = factory.addPlaneSurface([rect])
    return rect

def addGap(r0, r1, gap_width, gap_length, theta = 0):
    x = 0
    y = -gap_width/2
    dx = r0+r1+gap_length
    dy = gap_width
#    rect = factory.addRectangle(x, y, z, dx, dy)
    rect = addMultiPointRectangle(x, y, z, dx, dy)
    print('rect',rect)
    print('theta',theta)
    factory.rotate([(2,rect)], 0,0,0,0,0,1,theta*np.pi/180.)
    return [(2,rect)]

def addLoop(x, y, z, r, theta = 0, sides = 20):
    if (sides < 3) or (sides is None):
        c = factory.addCircle(x,y,z, r)
        cc = factory.addCurveLoop([c])
        loop = factory.addPlaneSurface([cc])
    else:
        cc = addPolygon(x,y,z, radius = r, N = sides)
        loop = factory.addPlaneSurface([cc])
        print(loop)

    factory.rotate([(2,loop)], 0,0,0,0,0,1,theta*np.pi/180.)

    return loop

def loop_gap_resonator(r0, r1, gap_width, gap_length, gaps = 2):



    # Add sample Loop
    sample_loop = addLoop(0, 0, z, theta = 0, r = r0)

    fused_dim_tag = [(2,sample_loop)]

    # Add Gaps
    print(fused_dim_tag)
    for ix in range(gaps):
        print(ix)
        theta = ix * (360./gaps)
        gap = addGap(r0, r1, gap_width, gap_length, theta)
        out = factory.fuse(fused_dim_tag,gap)
        fused_dim_tag = out[0]

    # Add Loops
    for ix in range(gaps):
        print(ix)
        theta = ix * (360./gaps)
        return_loop = addLoop(r0+r1+gap_length, 0, z, r1, theta)
        out = factory.fuse(fused_dim_tag,[(2, return_loop)])
        fused_dim_tag = out[0]

    return fused_dim_tag


lgr = loop_gap_resonator(sample_loop_radius, return_loop_radius, gap_width, gap_length, gaps = 4)

lgr_3d = factory.extrude(lgr, 0, 0, height, [10]) # ((dimTags) dx, dy, dz) need extrusion for mesh

#void_top = 

factory.synchronize()


gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)
#gmsh.model.mesh.refine()

element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(3)

# Print the element numbers
ix = 0
for tag in element_tags:
#    print(tag)
    ix += len(tag)
    print(type(tag))
    print("Element number:", tag)
print('-'*50)
print('Total Elements:', ix)
print('-'*50)



gmsh.write("lgr_test001.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()
