import gmsh

gmsh.initialize()

factory = gmsh.model.occ
gmsh.model.add('cylinder')
height = 1.0
radius = 0.5
factory.addCylinder(0,0,0, 0,0,height,radius)  # Volume for mesh
factory.synchronize()

#gmsh.option.setNumber("Mesh.ElementOrder", 3)
#gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 1 for hexahedral

gmsh.option.setNumber('Mesh.RecombineAll', 1)
gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
gmsh.option.setNumber('Mesh.Recombine3DLevel', 2)
gmsh.option.setNumber('Mesh.ElementOrder', 2)
gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
gmsh.option.setNumber('Mesh.MedFileMinorVersion', 0)
gmsh.option.setNumber('Mesh.SaveAll', 0)
gmsh.option.setNumber('Mesh.SaveGroupsOfNodes', 1)
gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
    
factory.synchronize()
gmsh.model.mesh.setOrder(3)
gmsh.model.mesh.generate(3)  # Generate 3D mesh
#gmsh.model.mesh.recombineAll()  # Generate 3D mesh

# Save the generated mesh
#gmsh.write("cylinder_hexahedral.msh")

# Finalize the Gmsh session
gmsh.fltk.run()
gmsh.finalize()
