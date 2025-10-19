import gmsh

gmsh.initialize()
gmsh.model.add("DielectricResonator")
factory = gmsh.model.occ

# Geometry
dielectricOD = 12e-3
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

# Physical groups
gmsh.model.addPhysicalGroup(3, [resonator], vacuum_tag, "resonator")
gmsh.model.addPhysicalGroup(3, [dielectric], dielectric_tag, "dielectric")

# --- Mesh fields ---

lc_diel = base_size / (eps_r**0.5)

# 1. Distance field from dielectric
field_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(field_dist, "VolumesList", [dielectric])

# 2. Threshold field: small mesh near dielectric
field_thresh = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field_thresh, "InField", field_dist)
gmsh.model.mesh.field.setNumber(field_thresh, "SizeMin", lc_diel)
gmsh.model.mesh.field.setNumber(field_thresh, "SizeMax", base_size)
gmsh.model.mesh.field.setNumber(field_thresh, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(field_thresh, "DistMax", 0.02)  # controls grading

# 3. Set as background mesh
gmsh.model.mesh.field.setAsBackgroundMesh(field_thresh)

# Generate mesh
gmsh.model.mesh.generate(3)
gmsh.write("dielectric_resonator_refined.msh")
gmsh.fltk.run()
gmsh.finalize()
