import gmsh

gmsh.initialize()

factory = gmsh.model.occ
gmsh.model.add('microstrip')
height = 0.5
width = 1.0
length = 1.0
stripline_length = 0.5
stripline_width = 0.05
copper_thickness = 0.001
substrate_thickness = 0.05
box = factory.addBox(-width/2,-length/2,-height/2, width, length, height)  # Volume for mesh
copper = factory.addBox(-stripline_width/2,-stripline_length/2,-height/2 + substrate_thickness, stripline_width, stripline_length, copper_thickness)  # Volume for mesh

factory.cut([(3, box)], [(3, copper)])
factory.synchronize()

    
factory.synchronize()

gmsh.model.addPhysicalGroup(3, [1], name = 'Resonator')#, 1) # need to add physcial groups




gmsh.model.mesh.generate(3)  # Generate 3D mesh
gmsh.model.mesh.refine()  # Generate 3D mesh

# Save the generated mesh
gmsh.write("microstrip_mesh.msh")

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
