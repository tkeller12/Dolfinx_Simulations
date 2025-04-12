import gmsh

gmsh.initialize()

factory = gmsh.model.occ
gmsh.model.add('microstrip')
height = 20e-3
width = 30e-3
length = 50e-3
stripline_length = 25e-3
stripline_width = 5e-3
copper_thickness = 35e-6
substrate_thickness = 1.6e-3
box = factory.addBox(-width/2,-length/2,-height/2, width, length, height)  # Volume for mesh
copper = factory.addBox(-stripline_width/2,-stripline_length/2,-height/2 + substrate_thickness, stripline_width, stripline_length, copper_thickness)  # Volume for mesh

factory.cut([(3, box)], [(3, copper)])
factory.synchronize()

    
factory.synchronize()

gmsh.model.addPhysicalGroup(3, [1], name = 'Resonator')#, 1) # need to add physcial groups


#gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option



gmsh.model.mesh.generate(3)  # Generate 3D mesh
#gmsh.model.mesh.refine()  # generate 3d mesh
#gmsh.model.mesh.refine()  # generate 3d mesh

# Save the generated mesh
gmsh.write("mesh/microstrip_mesh.msh")

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

# Finalize the Gmsh session
gmsh.fltk.run()
gmsh.finalize()
