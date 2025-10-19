import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("DielectricResonator")

factory = gmsh.model.occ

dielectricOD = 12e-3
dielectricID = 0
dielectricHeight = 6e-3
eps_r = 9.3
dielectric_tag = 2

resonatorOD = 30e-3
resonatorHeight = 20e-3
vacuum_tag = 1

lc = 0.01
base_size = 0.001


dielectric = factory.addCylinder(0,0,-dielectricHeight/2,0,0,dielectricHeight, dielectricOD/2)
resonator = factory.addCylinder(0,0,-resonatorHeight/2,0,0,resonatorHeight,resonatorOD/2)
result_tags, out_dim_tags_map = factory.cut([(3,resonator)], [(3, dielectric)], removeTool = False)

center_tag = factory.addPoint(0, 0, 0, meshSize=lc / (eps_r**0.5))

factory.synchronize()

print(dielectric)
print(resonator)
print(result_tags)
print('here')

gmsh.model.addPhysicalGroup(3, [resonator], vacuum_tag, "resonator")
gmsh.model.addPhysicalGroup(3, [dielectric], dielectric_tag, "dielectric")

# Vacuum
#gmsh.model.mesh.setSize(gmsh.model.getEntities(3), base_size)

# Dielectric — scale by 1/sqrt(eps_r)
dielectric_entities = gmsh.model.getEntitiesForPhysicalGroup(3, dielectric_tag)
#dielectric_entities = gmsh.model.getEntitiesForPhysicalGroup(2, dielectric_tag)



dielectric_volumes = gmsh.model.getEntitiesForPhysicalGroup(3, dielectric_tag)
dielectric_surfaces = []
for vol in dielectric_volumes:
    # get the 2D surfaces bounding this volume
    surfaces = gmsh.model.getBoundary([(3,vol)], oriented=False, recursive=False)
    for s in surfaces:
        if s[0] == 2:  # dimension 2 = surface
            dielectric_surfaces.append(s)

lc_diel = base_size / (eps_r**0.5)
gmsh.model.mesh.setSize(dielectric_surfaces, 0.001)

print('dielectric_entities:',dielectric_entities)
#gmsh.model.mesh.setSize([(3, tag) for tag in dielectric_entities], 0.001)
#gmsh.model.mesh.setSize([(3, tag) for tag in dielectric_entities], base_size / (eps_r**0.5))

#factory.synchronize()

#dielectric_entities = gmsh.model.getEntitiesForPhysicalGroup(3, dielectric_tag)
#print('dielectric_entities:',dielectric_entities)
#gmsh.model.mesh.setSize(([3,dielectric_entities[0]]), base_size / (eps_r**0.5))



# Set mesh order (2 = quadratic tetrahedra)
#gmsh.option.setNumber("Mesh.ElementOrder", 2)

# Set maximum element size
#gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

# Mesh smoothing (number of iterations)
#gmsh.option.setNumber("Mesh.Smoothing", 10)

# Base mesh size
#base_size = 0.002
#eps_r = 1

# Field for vacuum
#field_vac = gmsh.model.mesh.field.add("Constant")
#gmsh.model.mesh.field.setNumber(field_vac, "VIn", base_size)

# Field for dielectric
#dielectric_entities = gmsh.model.getEntitiesForPhysicalGroup(3, dielectric_tag)
#field_diel = gmsh.model.mesh.field.add("MathEval")
# Use sqrt(eps_r) refinement
#gmsh.model.mesh.field.setString(field_diel, "F", f"{base_size}/{eps_r**0.5}")

# Apply the dielectric field only inside the dielectric volume
# We do this via a "Distance" + "Threshold" field if you want smooth grading,
# or just set background field to the dielectric field for simplicity
#gmsh.model.mesh.field.setAsBackgroundMesh(field_diel)




#gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
#gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
gmsh.model.mesh.generate(3)

filename = "dielectric_resonator_test001.msh"
gmsh.write(filename)

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()


