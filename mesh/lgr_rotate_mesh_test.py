import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("2D_Rectangle")

factory = gmsh.model.occ

r0 = 1.0
r1 = 2.0
gap_width = 0.5
gap_length = 1.0

def addGap(r0, r1, gap_width, gap_length, theta = 0):
    x = 0
    y = -gap_width/2
    z = 0
    dx = r0+r1+gap_length
    dy = gap_width
    rect = factory.addRectangle(x, y, z, dx, dy)
    print('rect',rect)
    print('theta',theta)
    factory.rotate([(2,rect)], 0,0,0,0,0,1,theta*np.pi/180.)
    return [(2,rect)]

def addLoop(x, y, r, theta = 0):
    c = factory.addCircle(x,y,0, r0)
    cc = factory.addCurveLoop([c])
    loop = factory.addPlaneSurface([cc])
    factory.rotate([(2,loop)], 0,0,0,0,0,1,theta*np.pi/180.)

    return loop

def loop_gap_resonator(r0, r1, gap_width, gap_length, gaps = 2):

    # Add sample Loop
    sample_loop = addLoop(0, 0, r0)

    # Add Gaps
    fused_dim_tag = [(2,sample_loop)]
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
        return_loop = addLoop(r0+r1+gap_length, 0, r1, theta)
        out = factory.fuse(fused_dim_tag,[(2, return_loop)])
        fused_dim_tag = out[0]



lgr = loop_gap_resonator(r0, r1, gap_width, gap_length, gaps = 8)



factory.synchronize()

gmsh.model.mesh.generate(2)

gmsh.write("test001.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()
