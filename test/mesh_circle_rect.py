import gmsh

# Initialize GMSH
gmsh.initialize()

# Create a new model
gmsh.model.add("circle_intersection_rectangle")

# Define parameters for the rectangle and circle
#rect_width = 10.0  # Width of the rectangle
#rect_height = 5.0  # Height of the rectangle
#circle_radius = 3.0  # Radius of the circle
#circle_center_x = 5.0  # X-coordinate of circle center
#circle_center_y = 2.5  # Y-coordinate of circle center
#
gmsh.model.occ.addCircle(0,0,0,1)

# Define the rectangle
#rect_points = [
#    (0.0, 0.0),  # Point 1 (bottom-left corner)
#    (rect_width, 0.0),  # Point 2 (bottom-right corner)
#    (rect_width, rect_height),  # Point 3 (top-right corner)
#    (0.0, rect_height),  # Point 4 (top-left corner)
#]

# Add rectangle points and lines
#for i, (x, y) in enumerate(rect_points):
#    gmsh.model.geo.addPoint(x, y, 0, 1.0, i+1)
#
#gmsh.model.geo.addLine(1, 2)
#gmsh.model.geo.addLine(2, 3)
#gmsh.model.geo.addLine(3, 4)
#gmsh.model.geo.addLine(4, 1)
#
## Define the circle
#circle_center = (circle_center_x, circle_center_y)
#gmsh.model.geo.addPoint(circle_center[0], circle_center[1], 0, 1.0, len(rect_points) + 1)
#
## Circle is created as an arc
#circle_start_angle = 0.0
#circle_end_angle = 2.0 * 3.141592653589793  # Full circle (in radians)
#circle_tag = gmsh.model.geo.addCircleArc(len(rect_points) + 1, len(rect_points) + 2, len(rect_points) + 3)
#
## Create a physical group for the geometry
gmsh.model.occ.synchronize()
#
## Define a surface for the intersection (rectangle and circle)
#gmsh.model.geo.addCurveLoop([1, 2, 3, 4, circle_tag])
#gmsh.model.geo.addPlaneSurface([1])
#
## Mesh the geometry
gmsh.model.mesh.generate(2)
#
## Save the mesh to a file
gmsh.write("circle1.msh")
#
## Finalize the GMSH session
gmsh.finalize()
