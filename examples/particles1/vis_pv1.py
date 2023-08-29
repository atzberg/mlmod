# state file generated using paraview version 5.9.1

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

import glob

#output_dir = './output/run_proto_test3/batch_00_test001/vtk';
output_dir = './output/run_sim_particles1/batch_00_test001/vtk';
print("output_dir = " + output_dir);

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
flag_view='view1';
if flag_view=='view1':
  # Create a new 'Render View'
  renderView1 = CreateView('RenderView')
  renderView1.ViewSize = [1241, 479]
  renderView1.AxesGrid = 'GridAxes3DActor'
  renderView1.StereoType = 'Crystal Eyes'
  renderView1.CameraFocalDisk = 1.0
  renderView1.BackEnd = 'OSPRay raycaster'
  renderView1.OSPRayMaterialLibrary = materialLibrary1

elif flag_view=='view2':
  renderView1 = CreateView('RenderView')
  renderView1.ViewSize = [1166, 640]
  renderView1.AxesGrid = 'GridAxes3DActor'
  renderView1.StereoType = 'Crystal Eyes'
  renderView1.CameraPosition = [0.0, 0.0, 67.99559695559125]
  renderView1.CameraFocalDisk = 1.0
  renderView1.CameraParallelScale = 31.176914536239792
  renderView1.BackEnd = 'OSPRay raycaster'
  renderView1.OSPRayMaterialLibrary = materialLibrary1


SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1166, 640)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML PolyData Reader'
particles_file_list = glob.glob(output_dir + '/' + 'Particles*.vtp');
particles_file_list.sort();
num_particles_file_list = len(particles_file_list);
print("num_particles_file_list = " + str(num_particles_file_list));

particlesData = XMLPolyDataReader(registrationName='particles', FileName=particles_file_list)
particlesData.PointArrayStatus = ['id', 'type', 'vx', 'fx']
particlesData.TimeArray = 'None'

# create a new 'XML Unstructured Grid Reader'
bounding_box_file_list = glob.glob(output_dir + '/' + 'Particles*boundingBox.vtu');
bounding_box_file_list.sort();
num_bounding_box_file_list = len(bounding_box_file_list);
print("num_bounding_box_file_list = " + str(num_bounding_box_file_list));
boundingBoxData = XMLUnstructuredGridReader(registrationName='bounding_box', FileName=bounding_box_file_list)
boundingBoxData.TimeArray = 'None'

# create a new 'Glyph'
particle_sphere = Glyph(registrationName='particle_sphere', Input=particlesData,
    GlyphType='Sphere')
particle_sphere.OrientationArray = ['POINTS', 'No orientation array']
particle_sphere.ScaleArray = ['POINTS', 'No scale array']
particle_sphere.ScaleFactor = 1.0199999809265137
particle_sphere.GlyphTransform = 'Transform2'
particle_sphere.GlyphMode = 'All Points'

# create a new 'Glyph'
particle_fx = Glyph(registrationName='particle_fx', Input=particlesData,
    GlyphType='Arrow')
particle_fx.OrientationArray = ['POINTS', 'No orientation array']
particle_fx.ScaleArray = ['POINTS', 'fx']
particle_fx.ScaleFactor = 0.28998167514801027
particle_fx.GlyphTransform = 'Transform2'
particle_fx.GlyphMode = 'All Points'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from boundingBoxData
boundingBoxDisplay = Show(boundingBoxData, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
boundingBoxDisplay.Representation = 'Wireframe'
boundingBoxDisplay.AmbientColor = [0.0, 0.6666666666666666, 1.0]
boundingBoxDisplay.ColorArrayName = [None, '']
boundingBoxDisplay.DiffuseColor = [0.0, 0.6666666666666666, 1.0]
boundingBoxDisplay.SelectTCoordArray = 'None'
boundingBoxDisplay.SelectNormalArray = 'None'
boundingBoxDisplay.SelectTangentArray = 'None'
boundingBoxDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
boundingBoxDisplay.SelectOrientationVectors = 'None'
boundingBoxDisplay.ScaleFactor = 3.6
boundingBoxDisplay.SelectScaleArray = 'None'
boundingBoxDisplay.GlyphType = 'Arrow'
boundingBoxDisplay.GlyphTableIndexArray = 'None'
boundingBoxDisplay.GaussianRadius = 0.18
boundingBoxDisplay.SetScaleArray = [None, '']
boundingBoxDisplay.ScaleTransferFunction = 'PiecewiseFunction'
boundingBoxDisplay.OpacityArray = [None, '']
boundingBoxDisplay.OpacityTransferFunction = 'PiecewiseFunction'
boundingBoxDisplay.DataAxesGrid = 'GridAxesRepresentation'
boundingBoxDisplay.PolarAxes = 'PolarAxesRepresentation'
boundingBoxDisplay.ScalarOpacityUnitDistance = 62.353829072479584
boundingBoxDisplay.OpacityArrayName = [None, '']

# show data from particlesData
particlesDisplay = Show(particlesData, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
particlesDisplay.Representation = 'Surface'
particlesDisplay.ColorArrayName = [None, '']
particlesDisplay.SelectTCoordArray = 'None'
particlesDisplay.SelectNormalArray = 'None'
particlesDisplay.SelectTangentArray = 'None'
particlesDisplay.OSPRayScaleArray = 'fx'
particlesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
particlesDisplay.SelectOrientationVectors = 'None'
particlesDisplay.ScaleFactor = 1.0199999809265137
particlesDisplay.SelectScaleArray = 'None'
particlesDisplay.GlyphType = 'Arrow'
particlesDisplay.GlyphTableIndexArray = 'None'
particlesDisplay.GaussianRadius = 0.050999999046325684
particlesDisplay.SetScaleArray = ['POINTS', 'fx']
particlesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
particlesDisplay.OpacityArray = ['POINTS', 'fx']
particlesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
particlesDisplay.DataAxesGrid = 'GridAxesRepresentation'
particlesDisplay.PolarAxes = 'PolarAxesRepresentation'

# show data from particle_fx
particle_fxDisplay = Show(particle_fx, renderView1, 'GeometryRepresentation')

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
particlesDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 10.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
particlesDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 10.0, 1.0, 0.5, 0.0]

# show data from particle_sphere
particle_sphereDisplay = Show(particle_sphere, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
particle_sphereDisplay.Representation = 'Surface'
particle_sphereDisplay.ColorArrayName = [None, '']
particle_sphereDisplay.SelectTCoordArray = 'None'
particle_sphereDisplay.SelectNormalArray = 'Normals'
particle_sphereDisplay.SelectTangentArray = 'None'
particle_sphereDisplay.OSPRayScaleArray = 'Normals'
particle_sphereDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
particle_sphereDisplay.SelectOrientationVectors = 'None'
particle_sphereDisplay.ScaleFactor = 1.1219999313354492
particle_sphereDisplay.SelectScaleArray = 'None'
particle_sphereDisplay.GlyphType = 'Arrow'
particle_sphereDisplay.GlyphTableIndexArray = 'None'
particle_sphereDisplay.GaussianRadius = 0.05609999656677246
particle_sphereDisplay.SetScaleArray = ['POINTS', 'Normals']
particle_sphereDisplay.ScaleTransferFunction = 'PiecewiseFunction'
particle_sphereDisplay.OpacityArray = ['POINTS', 'Normals']
particle_sphereDisplay.OpacityTransferFunction = 'PiecewiseFunction'
particle_sphereDisplay.DataAxesGrid = 'GridAxesRepresentation'
particle_sphereDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
particle_sphereDisplay.ScaleTransferFunction.Points = [-0.9749279022216797, 0.0, 0.5, 0.0, 0.9749279022216797, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
particle_sphereDisplay.OpacityTransferFunction.Points = [-0.9749279022216797, 0.0, 0.5, 0.0, 0.9749279022216797, 1.0, 0.5, 0.0]

# trace defaults for the display properties.
particle_fxDisplay.Representation = 'Surface'
particle_fxDisplay.ColorArrayName = [None, '']
particle_fxDisplay.SelectTCoordArray = 'None'
particle_fxDisplay.SelectNormalArray = 'None'
particle_fxDisplay.SelectTangentArray = 'None'
particle_fxDisplay.OSPRayScaleArray = 'fx'
particle_fxDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
particle_fxDisplay.SelectOrientationVectors = 'None'
particle_fxDisplay.ScaleFactor = 0.6379596829414368
particle_fxDisplay.SelectScaleArray = 'None'
particle_fxDisplay.GlyphType = 'Arrow'
particle_fxDisplay.GlyphTableIndexArray = 'None'
particle_fxDisplay.GaussianRadius = 0.03189798414707184
particle_fxDisplay.SetScaleArray = ['POINTS', 'fx']
particle_fxDisplay.ScaleTransferFunction = 'PiecewiseFunction'
particle_fxDisplay.OpacityArray = ['POINTS', 'fx']
particle_fxDisplay.OpacityTransferFunction = 'PiecewiseFunction'
particle_fxDisplay.DataAxesGrid = 'GridAxesRepresentation'
particle_fxDisplay.PolarAxes = 'PolarAxesRepresentation'
particle_fxDisplay.SelectInputVectors = [None, '']
particle_fxDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
particle_fxDisplay.ScaleTransferFunction.Points = [-10.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
particle_fxDisplay.OpacityTransferFunction.Points = [-10.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# restore active source
SetActiveSource(None)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
