import gmsh
import numpy as np

# Initialize GMSH
gmsh.initialize()

gmsh.model.add("DielectricResonator")

factory = gmsh.model.occ

dielectricOD = 12e-3
dielectricID = 0
dielectricHeight = 6e-3
eps_r = 20
dielectric_tag = 2

resonatorOD = 30e-3
resonatorHeight = 20e-3
vacuum_tag = 1

base_size = 0.01


dielectric = factory.addCylinder(0,0,-dielectricHeight/2,0,0,dielectricHeight, dielectricOD/2)
resonator = factory.addCylinder(0,0,-resonatorHeight/2,0,0,resonatorHeight,resonatorOD/2)
result_tags, out_dim_tags_map = factory.cut([(3,resonator)], [(3, dielectric)], removeTool = False)

factory.synchronize()

# Assign physical groups
gmsh.model.addPhysicalGroup(3, [resonator], vacuum_tag, "resonator")
gmsh.model.addPhysicalGroup(3, [dielectric], dielectric_tag, "dielectric")

#factory.synchronize() #?

#### Mesh Code ####
# --- Background mesh field ---
# 1. Constant field for vacuum
field_vac = gmsh.model.mesh.field.add("Constant")
gmsh.model.mesh.field.setNumber(field_vac, "VIn", base_size)

# 2. MathEval field for dielectric refinement
lc_diel = base_size / (eps_r**0.5)
field_diel = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(field_diel, "F", f"{lc_diel}")

# 3. Combine fields using Min (smallest size dominates)
field_min = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field_vac, field_diel])

# 4. Set as background mesh
gmsh.model.mesh.field.setAsBackgroundMesh(field_min)


#field_dist = gmsh.model.mesh.field.add("Distance")
#gmsh.model.mesh.field.setNumbers(field_dist, "VolumesList", [dielectric_tag])
#
#field_thresh = gmsh.model.mesh.field.add("Threshold")
#gmsh.model.mesh.field.setNumber(field_thresh, "InField", field_dist)
#gmsh.model.mesh.field.setNumber(field_thresh, "SizeMin", lc_diel)
#gmsh.model.mesh.field.setNumber(field_thresh, "SizeMax", base_size)
#field_min = gmsh.model.mesh.field.add("Min")
#gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field_thresh])
#gmsh.model.mesh.field.setAsBackgroundMesh(field_min)

gmsh.model.mesh.generate(3)

filename = "dielectric_resonator_test001.msh"
gmsh.write(filename)

# Run the GMSH GUI to visualize the mesh (comment out if you don't want to use the GUI)
gmsh.fltk.run()

# Finalize the GMSH API
gmsh.finalize()


