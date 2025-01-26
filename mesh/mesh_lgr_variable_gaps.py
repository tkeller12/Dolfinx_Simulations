import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("2D_Rectangle")

factory = gmsh.model.occ

sample_loop_radius = 0.5
return_loop_radius = 1.0
gap_width = 0.1
gap_length = 1.0
height = 5.0

void_thickness = 0.1
void_radius = sample_loop_radius + gap_length + 2*return_loop_radius + void_thickness
void_height = 1.0

z = -height/2

def addGap(r0, r1, gap_width, gap_length, theta = 0):
    x = 0
    y = -gap_width/2
    dx = r0+r1+gap_length
    dy = gap_width
    rect = factory.addRectangle(x, y, z, dx, dy)
    print('rect',rect)
    print('theta',theta)
    factory.rotate([(2,rect)], 0,0,0,0,0,1,theta*np.pi/180.)
    return [(2,rect)]

def addLoop(x, y, r, theta = 0):
    c = factory.addCircle(x,y,z, r)
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

    return fused_dim_tag


lgr = loop_gap_resonator(sample_loop_radius, return_loop_radius, gap_width, gap_length, gaps = 4)

lgr_3d = factory.extrude(lgr, 0, 0, height, [20]) # ((dimTags) dx, dy, dz) need extrusion for mesh

#void_top = 

factory.synchronize()


gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)
#gmsh.model.mesh.refine()


gmsh.write("lgr_test001.msh")

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()
