import gmsh



# Initialize Gmsh

gmsh.initialize()
factory = gmsh.model.occ



# Create two points for the line

pt1 = factory.addPoint(0, 0, 0, 1)

pt2 = factory.addPoint(1, 0, 0, 1)



# Create the line using the points

line = factory.addLine(pt1, pt2, 1)



# Set the number of points along the line using transfinite curve

gmsh.model.mesh.setTransfiniteCurve(line, 20)  # 20 points on the line



# Generate the mesh

gmsh.mesh.generate(2)
gmsh.fltk.run()



# Finalize Gmsh

gmsh.finalize()
