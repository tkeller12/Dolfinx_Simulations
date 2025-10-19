import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()
gmsh.model.add("DielectricResonator")

factory = gmsh.model.occ

# --- Geometry parameters ---
#dielectricOD = 12.7e-3
#dielectricHeight = 6.35e-3

dielectricOD = 10e-3
dielectricHeight = 13e-3
dielectricID = 5.1e-3
eps_r = 9.3
dielectric_tag = 2

resonatorOD = 20e-3
resonatorHeight = 20e-3
vacuum_tag = 1

base_size = 0.003
#base_size = 0.008

# --- Create geometry ---
dielectric = factory.addCylinder(0,0,-dielectricHeight/2,0,0,dielectricHeight, dielectricOD/2)
if not np.isclose(dielectricID ,0):
    dielectricVoid = factory.addCylinder(0,0,-dielectricHeight/2,0,0,dielectricHeight, dielectricID/2)
    result_tags, out_dim_tags_map = factory.cut([(3,dielectric)], [(3, dielectricVoid)], removeTool = True)
    print(result_tags)



resonator = factory.addCylinder(0,0,-resonatorHeight/2,0,0,resonatorHeight,resonatorOD/2)
result_tags, out_dim_tags_map = factory.cut([(3,resonator)], [(3, dielectric)], removeTool = False)

factory.synchronize()

# --- Physical groups ---
gmsh.model.addPhysicalGroup(3, [resonator], vacuum_tag, "resonator")
gmsh.model.addPhysicalGroup(3, [dielectric], dielectric_tag, "dielectric")

# --- Mesh refinement fields ---
lc_diel = base_size / (eps_r**0.5)
lc_vac = base_size

# 1. Base size field (coarse everywhere)
#field_base = gmsh.model.mesh.field.add("Constant")
#gmsh.model.mesh.field.setNumber(field_base, "VIn", base_size)

# 2. Box field for vacuum (coarse)
field_vac = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field_vac, "VIn", lc_vac)
gmsh.model.mesh.field.setNumber(field_vac, "XMin", -resonatorOD/2)
gmsh.model.mesh.field.setNumber(field_vac, "XMax", resonatorOD/2)
gmsh.model.mesh.field.setNumber(field_vac, "YMin", -resonatorOD/2)
gmsh.model.mesh.field.setNumber(field_vac, "YMax", resonatorOD/2)
gmsh.model.mesh.field.setNumber(field_vac, "ZMin", -resonatorHeight/2)
gmsh.model.mesh.field.setNumber(field_vac, "ZMax", resonatorHeight/2)

# 2. Box field to refine dielectric interior
field_box = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field_box, "VIn", lc_diel)
gmsh.model.mesh.field.setNumber(field_box, "XMin", -dielectricOD/2)
gmsh.model.mesh.field.setNumber(field_box, "XMax", dielectricOD/2)
gmsh.model.mesh.field.setNumber(field_box, "YMin", -dielectricOD/2)
gmsh.model.mesh.field.setNumber(field_box, "YMax", dielectricOD/2)
gmsh.model.mesh.field.setNumber(field_box, "ZMin", -dielectricHeight/2)
gmsh.model.mesh.field.setNumber(field_box, "ZMax", dielectricHeight/2)


# 3. Combine fields: take minimum size (fine in dielectric, coarse outside)
field_min = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field_vac, field_box])

# 4. Set as background mesh
gmsh.model.mesh.field.setAsBackgroundMesh(field_min)

#gmsh.option.setNumber("Mesh.CharacteristicLengthMax", base_size) # over-rides box


# --- Generate 3D mesh ---
gmsh.model.mesh.generate(3)
gmsh.write("dielectric_resonator_refined.msh")
gmsh.fltk.run()
gmsh.finalize()
