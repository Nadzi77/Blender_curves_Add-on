bl_info = {
    "name": "CurvesAddon",
    "blender": (2, 80, 0),
    "category": "Curve",
    "author" : "Nadzady",
    "description" : "",
    "version" : (0, 1, 1),
    "location" : "Panel menu",
    "warning" : "",
}

import bpy
import gpu
import bgl
import time
import mathutils
import numpy as np
from math import sqrt
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader

eps = 1e-6
class MyProperties(bpy.types.PropertyGroup):
    """ Properties used in user interface to set parameters of spline. """
    KochBarTension: bpy.props.FloatProperty(name = "", description = "Tension", default = 0, min = -1+eps, max = 1-eps)
    KochBarBias: bpy.props.FloatProperty(name = "", description = "Bias", default = 0, min = -1+eps, max = 1-eps)
    KochBarContinuity: bpy.props.FloatProperty(name = "", description = "Continuity", default = 0, min = -1+eps, max = 1-eps)

    checkbox: bpy.props.BoolProperty(
        name="",
        description="A bool property",
        default = False
        )
    end_conditions_kochbar: bpy.props.EnumProperty(
        items=(('RELAXED', 'Relaxed', "Relaxed/Natural end condition"),
               ('CLAMPED', 'Clamped', "Clamped end condition"),
               ('NOTAKNOT', 'Not-a-knot', "Not-a-knot end condition"),
               ('BESSEL', 'Bessel', "Bessel end condition")),
        name="",
        description="Choose end condition",
        default='RELAXED',
        options={'ANIMATABLE'}
        )

    end_conditions_hermite: bpy.props.EnumProperty(
        items=(('RELAXED', 'Relaxed', "Relaxed/Natural end condition"),
               ('CLAMPED', 'Clamped', "Clamped end condition"),
               ('NOTAKNOT', 'Not-a-knot', "Not-a-knot end condition"),
               ('BESSEL', 'Bessel', "Bessel end condition"),
               ('FMM', 'FMM', "Forsythe-Malcolm-Moler end condition"),
               ('CYCLIC', 'Cyclic', "Cyclic end condition"),
               ('ACYCLIC', 'Acyclic', "Acyclic end condition")),
        name="",
        description="Choose end condition",
        default='RELAXED',
        options={'ANIMATABLE'}
        )

    param_type_hermite: bpy.props.EnumProperty(
        items=(('UNIFORM', 'Uniform', "Uniform parametrization"),
               ('CHORD', 'Chord Length', "Chord length parametrization"),
               ('CENTRIP', 'Centripetal', "Centripetal parametrization")),
        name="",
        description="Choose end condition",
        default='UNIFORM',
        options={'ANIMATABLE'}
        )
    spline_type: bpy.props.EnumProperty(
        items=(('HERM', 'Hermite Cubic', "Hermite Cubic Spline"),
               ('KOBA', 'Kochanek-Bartels', "Kochanek-Bartels Spline")),
        name="",
        description="Choose type of spline",
        default='HERM',
        options={'ANIMATABLE'}
        )


def drawCurve(vertices,color):
    """ Draw curve from given vertices with given color. """
    shader = gpu.shader.from_builtin('3D_SMOOTH_COLOR')
    help_vertices = []
    lenVer = len(vertices)
    for i in range(lenVer):
        help_vertices.append(vertices[i])
        if i > 0 and i < (lenVer-1):
            help_vertices.append(vertices[i])
    col = [color] * (2 * lenVer - 2)
    batch = (batch_for_shader(shader, 'LINES', {"pos": help_vertices, "color": col}))
    def draw():
        bgl.glLineWidth(4)
        shader.bind()
        batch.draw(shader)
        bgl.glLineWidth(1)
    draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')

    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def drawBlenderCurve(vertices, rightTangents, leftTangents, ratios=[]):
    """ Create Blender object curve from given vertices and tangent vertices """
    verts = vertices
    t = ratios
    curvedata = bpy.data.curves.new(name='curve', type='CURVE')
    curvedata.dimensions = '3D'

    objectdata = bpy.data.objects.new("Spline", curvedata)
    objectdata.location = (0,0,0) #object origin
    bpy.context.collection.objects.link(objectdata)

    spline = curvedata.splines.new('BEZIER')
    spline.bezier_points.add(len(verts)-1)
    for num in range(len(verts)):
        x, y, z = verts[num]
        if (num < len(verts)-1):
            spline.bezier_points[num].handle_right = (x+rightTangents[num].x*t[num]/3,y+rightTangents[num].y*t[num]/3,z+rightTangents[num].z*t[num]/3)
        else:
            spline.bezier_points[num].handle_right = (x+leftTangents[num-1].x*t[num-1]/3,y+leftTangents[num-1].y*t[num-1]/3,z+leftTangents[num-1].z*t[num-1]/3)
        spline.bezier_points[num].co = (x, y, z)
        if (num > 0):
            spline.bezier_points[num].handle_left = (x-leftTangents[num-1].x*t[num-1]/3,y-leftTangents[num-1].y*t[num-1]/3,z-leftTangents[num-1].z*t[num-1]/3)
        else:
            spline.bezier_points[num].handle_left = (x-rightTangents[num].x*t[num]/3,y-rightTangents[num].y*t[num]/3,z-rightTangents[num].z*t[num]/3)

    spline.order_u = len(spline.bezier_points)-1
    objectdata.data.resolution_u = 64
    spline.use_endpoint_u = True


def computeHermiteSegment(R0,R1,r0,r1,numberOfLineSegments):
    """ Compute Hermite segment from two vertices, two tangent vertices and order of approximation. """
    spline_vertices = []
    for i in range(numberOfLineSegments + 1):
        t = i/numberOfLineSegments
        H0 = 1-3*t*t+2*t*t*t
        H3 = 3*t*t-2*t*t*t
        H1 = t-2*t*t+t*t*t
        H2 = -t*t+t*t*t
        sum = H0*R0 + H3*R1 + H1*r0 + H2*r1
        spline_vertices.append(sum)
    return spline_vertices

def computeBezierSegment(V0,V1,V2,V3,numberOfLineSegments):
    """ Compute bezier segment from four vertices and order of approximation. """
    spline_vertices = []
    vertices = [V0,V1,V2,V3]
    for i in range(numberOfLineSegments + 1):
        t = i/numberOfLineSegments
        help = vertices.copy()
        for k in reversed(range(4)):
            if k > 0:
                for j in range(k):
                    help[j] = (t * help[j]) + ((1-t) * help[j+1])
                help.pop()
            else:
                sum = help[0]
        spline_vertices.append(sum)
    return spline_vertices

class NmenuAddon(bpy.types.Panel):
    """ Draw user interface in panel menu. """
    bl_idname = "OBJECT_PT_nMenuPanel"
    bl_label = "Create a new curve"
    bl_category = "New Curve"
    bl_space_type = "VIEW_3D"
    bl_region_type = 'UI'

    def draw(self, context):
        scene = context.scene
        mytool = scene.my_tool
        spline_type = bpy.context.scene.my_tool.spline_type
        layout = self.layout

        layout.separator()
        layout.operator("object.set_control_points")

        layout.separator()
        layout.label(text="Create Spline")
        box = layout.box()
        row = box.row()
        split = row.split(factor=0.45)
        col1 = split.column()
        col2 = split.column()
        col1.alignment = 'RIGHT'
        col1.label(text="Spline Type:")
        col2.prop(mytool, "spline_type")
        if (spline_type == 'KOBA'):
            col1.label(text="Tension")
            col2.prop(mytool, "KochBarTension")
            col1.label(text="Bias")
            col2.prop(mytool, "KochBarBias")
            col1.label(text="Continuity")
            col2.prop(mytool, "KochBarContinuity")
            col1.label(text="End Conditions:")
            col2.prop(mytool, "end_conditions_kochbar")
        elif (spline_type == 'HERM'):
            col1.label(text="Parametrization")
            col2.prop(mytool, "param_type_hermite")
            col1.label(text="End Conditions:")
            col2.prop(mytool, "end_conditions_hermite")
        layout.operator("object.draw_cubic_spline")

        
        #layout.label(text="Bezier Spline")
        #layout.operator("object.draw_bezier_spline")
        #layout.separator()
        #layout.label(text="Bezier Curve")
        #layout.operator("object.draw_bezier_curve")


class DrawCustomSpline(bpy.types.Operator):
    """ Draw custom cubic spline according to given parameters values. """
    bl_idname = "object.draw_cubic_spline"
    bl_label = "Draw Cubic Spline"
    bl_options = {'REGISTER', 'UNDO'}
    numberOfLineSegments = 500

    def execute(self, context):
        numberOfLineSegments = self.numberOfLineSegments
        spline_type = bpy.context.scene.my_tool.spline_type

        #curve = bpy.context.active_object
        #bez_points = curve.data.splines[0].bezier_points
        #lenVer = len(bez_points)
        #vertices = []
        #for bez_point in bez_points:
        #    vertices.append(bez_point.co)
        #controlPoints = vertices

        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['ControlPoints'].select_set(True)
        obj = bpy.context.active_object
        verts = obj.data.vertices
        lenVer = len(verts)
        controlPoints = []
        for i in range(lenVer):
            controlPoints.append(verts[i].co)

        if (spline_type == 'HERM'):
            calculateHermiteSpline(controlPoints)
        elif (spline_type == 'KOBA'):
            calculateKochBarSpline(controlPoints)

        return {'FINISHED'}

def calculateHermiteSpline(vertices):
    """ Create Hermite spline from control points (vertices) and order of approximation. """
    #spline_vertices = []
    end_con_type = bpy.context.scene.my_tool.end_conditions_hermite
    param_type = bpy.context.scene.my_tool.param_type_hermite
    lenVer = len(vertices)
    t = []
    sum = 0
    a = 0
    dist = []
    if (param_type == 'UNIFORM'):
        for j in range(lenVer-1):
            t.append(1)
    elif (param_type == 'CHORD'):
        for j in range(lenVer-1):
            dist.append(sqrt((vertices[j+1].x - vertices[j].x)**2 + (vertices[j+1].y - vertices[j].y)**2 + (vertices[j+1].z - vertices[j].z)**2))
            sum += dist[j]
        for j in range(lenVer-1):
            t.append((lenVer-1)*(dist[j]/sum))
            a += t[j]
    elif (param_type == 'CENTRIP'):
        for j in range(lenVer-1):
            dist.append(sqrt(sqrt((vertices[j+1].x - vertices[j].x)**2 + (vertices[j+1].y - vertices[j].y)**2 + (vertices[j+1].z - vertices[j].z)**2)))
            sum += dist[j]
        for j in range(lenVer-1):
            #t.append(sqrt(dist/(lenVer-1)))
            t.append((lenVer-1)*(dist[j]/sum))
    print(t)

    Q = np.empty((lenVer),mathutils.Vector)
    M = np.zeros((lenVer,lenVer), dtype=float)
    for j in range(1,lenVer-1):
        M[j, j - 1] = t[j]
        M[j, j] = 2*(t[j]+t[j-1])
        M[j, j + 1] = t[j-1]
        Q[j] = (2 / (t[j]*t[j-1])) \
              * (t[j-1] * t[j-1] * (vertices[j+1]-vertices[j]) + t[j] * t[j] * (vertices[j]-vertices[j-1]))
    if (end_con_type == 'CLAMPED'): # add first and last row according to end conditions
        M[0][0] = 1*t[0]
        M[lenVer-1][lenVer-1] = 1*t[lenVer-2]
        Q[0] = vertices[1]-vertices[0]
        Q[lenVer-1] = vertices[lenVer-1]-vertices[lenVer-2]
    elif (end_con_type == 'RELAXED'):
        M[0][0] = 2*t[0]
        M[0][1] = 1*t[0]
        M[lenVer-1][lenVer-2] = 1*t[lenVer-2]
        M[lenVer-1][lenVer-1] = 2*t[lenVer-2]
        Q[0] = (3*(vertices[1]-vertices[0]))
        Q[lenVer-1] = (3*(vertices[lenVer-1]-vertices[lenVer-2]))
    elif (end_con_type == 'NOTAKNOT'):
        M[0][0] = -1*t[0]
        M[0][2] = 1*t[1]
        M[lenVer-1][lenVer-3] = 1*t[lenVer-3]
        M[lenVer-1][lenVer-1] = -1*t[lenVer-2]
        Q[0] = (2*(vertices[0]-2*vertices[1]+vertices[2]))
        Q[lenVer-1] = (-2*(vertices[lenVer-3]-2*vertices[lenVer-2]+vertices[lenVer-1]))
    elif (end_con_type == 'BESSEL'):
        M[0][0] = 1*t[0]
        M[lenVer-1][lenVer-1] = 1
        Q[0] = vertices[1]-vertices[0]+vertices[1]-((1/2)*vertices[2]+(1/2)*vertices[0])
        Q[lenVer-1] = ((1/2)*vertices[lenVer-3]+(1/2)*vertices[lenVer-1])-vertices[lenVer-2]+vertices[lenVer-1]-vertices[lenVer-2]
    elif (end_con_type == 'FMM'):
        M[0][0] = 6
        M[0][1] = 6
        M[lenVer-1][lenVer-2] = 6
        M[lenVer-1][lenVer-1] = 6
        Q[0] = vertices[3]-vertices[2]+2*(vertices[1]-vertices[2])+13*(vertices[1]-vertices[0])
        Q[lenVer-1] = vertices[lenVer-2]-vertices[lenVer-3]+2*(vertices[lenVer-2]-vertices[lenVer-1])+13*(vertices[lenVer-1]-vertices[lenVer-2])
    elif (end_con_type == 'CYCLIC'):
        M[0][0] = 1*t[0]
        M[0][lenVer-1] = (-1)*t[lenVer-2]
        M[lenVer-1][0] = 2*t[0]
        M[lenVer-1][1] = 1*t[0]
        M[lenVer-1][lenVer-2] = 1*t[lenVer-2]
        M[lenVer-1][lenVer-1] = 2*t[lenVer-2]
        Q[0] = mathutils.Vector((0, 0, 0))
        Q[lenVer-1] = (3*(vertices[1]-vertices[0]+vertices[lenVer-1]-vertices[lenVer-2]))
    elif (end_con_type == 'ACYCLIC'):
        M[0][0] = 1*t[0]
        M[0][lenVer-1] = 1*t[lenVer-2]
        M[lenVer-1][0] = 2*t[0]
        M[lenVer-1][1] = 1*t[0]
        M[lenVer-1][lenVer-2] = -1*t[lenVer-2]
        M[lenVer-1][lenVer-1] = -2*t[lenVer-2]
        Q[0] = mathutils.Vector((0, 0, 0))
        Q[lenVer-1] = (3*(vertices[1]-vertices[0]+vertices[lenVer-2]-vertices[lenVer-1]))
    invM = np.linalg.inv(M)
    r = invM.dot(Q) # calculate tangent vector points

    leftTangents = np.delete(r, 0)
    drawBlenderCurve(vertices,r,leftTangents,t)

def calculateKochBarSpline(vertices):
    """ Create Kochanek-Bartels spline from control points (vertices) and order of approximation. """
    lenVer = len(vertices)
    T = bpy.context.scene.my_tool.KochBarTension
    b = bpy.context.scene.my_tool.KochBarBias
    c = bpy.context.scene.my_tool.KochBarContinuity
    s = (1-T)/2

    K = np.array([[0,1,0,0],
                 [-(1/2)*(1-T)*(1+b)*(1-c),(1-T)*(b-c),(1/2)*(1-T)*(1-b)*(1+c),0],
                 [(1-T)*(1+b)*(1-c),-3+(1/2)*(1-T)*((1+b)*(1+c)-4*(b-c)),3-(1-T)*(1+c*(2-b)),-(1/2)*(1-T)*(1-b)*(1-c)],
                 [-(1/2)*(1-T)*(1+b)*(1-c),2+(1/2)*(1-T)*(2*(b-c)-(1+b)*(1+c)),-2+(1/2)*(1-T)*(2*(b+c)+(1-b)*(1+c)),(1/2)*(1-T)*(1-b)*(1-c)]])

    if bpy.context.scene.my_tool.end_conditions_kochbar == 'CLAMPED':
        vertices.append(vertices[lenVer-1])
        vertices.insert(0,vertices[0])
    if bpy.context.scene.my_tool.end_conditions_kochbar == 'RELAXED':
        va = 1/(2*K[2,0]) * (-2*K[2,1]*vertices[0]-2*K[2,2]*vertices[1]-2*K[2,3]*vertices[2])
        vb = 1/(2*K[2,3]+6*K[3,3]) * (-(2*K[2,0]+6*K[3,0])*vertices[lenVer-3]-(2*K[2,1]+6*K[3,1])*vertices[lenVer-2]-(2*K[2,2]+6*K[3,2])*vertices[lenVer-1])
        vertices.append(mathutils.Vector((vb[0], vb[1], vb[2])))
        vertices.insert(0,mathutils.Vector((va[0], va[1], va[2])))
    if bpy.context.scene.my_tool.end_conditions_kochbar == 'NOTAKNOT':
        va = (1/K[3,0]) * ((K[3,0]-K[3,1])*vertices[0]+(K[3,1]-K[3,2])*vertices[1]+(K[3,2]-K[3,3])*vertices[2]+K[3,3]*vertices[3])
        vb = (1/K[3,3]) * (K[3,0]*vertices[lenVer-4]+(K[3,1]-K[3,0])*vertices[lenVer-3]+(K[3,2]-K[3,1])*vertices[lenVer-2]+(K[3,3]-K[3,2])*vertices[lenVer-1])
        vertices.append(mathutils.Vector((vb[0], vb[1], vb[2])))
        vertices.insert(0,mathutils.Vector((va[0], va[1], va[2])))
    if bpy.context.scene.my_tool.end_conditions_kochbar == 'BESSEL':
        va = 1/(K[1,0]) * ((-(3/2)-K[1,1])*vertices[0]+(2-K[1,2])*vertices[1]-(1/2)*vertices[2])
        hea = (K[1,0]+2*K[2,0]+3*K[3,0])
        heb = (K[1,1]+2*K[2,1]+3*K[3,1])
        hec = (K[1,2]+2*K[2,2]+3*K[3,2])
        vb = 1/(2*K[2,3]+3*K[3,3]) * (((1/2)-hea)*vertices[lenVer-3]+(-2-heb)*vertices[lenVer-2]+((3/2)-hec)*vertices[lenVer-1])
        vertices.append(mathutils.Vector((vb[0], vb[1], vb[2])))
        vertices.insert(0,mathutils.Vector((va[0], va[1], va[2])))

    lenVer = len(vertices)
    V = [0]*4
    C = [0]*4

    rightTangents = []
    leftTangents = []
    for j in range(lenVer-3):
        rightTangents.append((1/2)*(1-T)*((1+b)*(1+c)*(vertices[j+1]-vertices[j])+(1-b)*(1-c)*(vertices[j+2]-vertices[j+1])))
        leftTangents.append((1/2)*(1-T)*((1+b)*(1-c)*(vertices[j+2]-vertices[j+1])+(1-b)*(1+c)*(vertices[j+3]-vertices[j+2])))
    innerVerts = vertices.copy()
    innerVerts.pop(0) # Delete first "fantom" vertex
    innerVerts.pop()  # Delete last "fantom" vertex
    ones = [1] * (len(innerVerts)-1)
    drawBlenderCurve(innerVerts,rightTangents,leftTangents,ones)

#-----------------------------------------------------------------------------------------------------------------


class ModalOperator(bpy.types.Operator):
    """ Set control points in scene. Use only with view from above ("7").
    Set points with Left Mouse Button, cancel with Right Mouse Button or Esc Button. """
    bl_idname = "object.set_control_points"
    bl_label = "Set Control Points"

    ob = None
    me = None
    verts = []
    edges = []
    first = True

    def addVerts(self,verts):
        curve = bpy.context.active_object
        bez_points = curve.data
        bpy.data.meshes.remove(bez_points)
        self.first = False;
        me = bpy.data.meshes.new("NewMesh")
        ob = bpy.data.objects.new("ControlPoints", me)
        me.from_pydata(verts, [], [])
        me.update()
        bpy.context.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob

    def modal(self, context, event):
        verts = self.verts
        edges = self.edges
        me = self.me

        if event.type == 'LEFTMOUSE':
            if event.value == 'RELEASE':
                if (self.first):
                    verts = []
                    found = 'NewMesh' in bpy.data.meshes
                    print(found)
                    if found:
                        bpy.data.meshes.remove(bpy.data.meshes["NewMesh"])

                vectorClicked = (view3d_utils.region_2d_to_location_3d(bpy.context.region, bpy.context.space_data.region_3d,
                                (event.mouse_region_x, event.mouse_region_y), mathutils.Vector((0, 0, 0))))
                verts.append((vectorClicked[0], vectorClicked[1], 0))
                for i in range(len(self.verts) - 1):
                    edges.append((i, i+1))

                self.addVerts(verts)
                self.verts = verts

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        me = bpy.data.meshes.new("AddonMeshA")
        ob = bpy.data.objects.new("ControlPoints", me)
        self.ob = ob
        self.me = me

        bpy.context.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob

        if context.object:
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish")
            return {'CANCELLED'}

class DrawBezierCurve(bpy.types.Operator):
    """ Draw Bezier curve. """
    bl_idname = "object.draw_bezier_curve"
    bl_label = "Bezier Curve"
    bl_options  = {'REGISTER', 'UNDO'}
    numberOfLineSegments = 20

    def execute(self, context):
        curve = bpy.context.active_object
        numberOfLineSegments = self.numberOfLineSegments

        vertices = []
        spline_vertices = []

        bez_points = curve.data.splines[0].bezier_points
        lenVer = len(bez_points)
        for bez_point in bez_points:
            vertices.append(bez_point.co)
        start = time.time()
        spl2 = []
        for i in range(1000):
            spl2.append(computeHermiteSegment(vertices[1],vertices[2],vertices[1]-vertices[0],vertices[3]-vertices[2],50))  ################ TEST HERMITE
        end = time.time()
        print("Hermite")
        print(end - start)

        start = time.time()
        spl3 = []
        for i in range(1000):
            spl3.append(computeBezierSegment(vertices[1],vertices[1]+(vertices[1]-vertices[0])/3,vertices[2]+(vertices[2]-vertices[3])/3,vertices[2],50))  ################ TEST BEZIER
        end = time.time()
        print("\nBezier")
        print(end - start)
        print("\n")

        for i in range(numberOfLineSegments+1):
            t = i/numberOfLineSegments

            help = vertices.copy()
            for k in reversed(range(lenVer)):
                if k > 0:
                    for j in range(k):
                        help[j] = ((1-t) * help[j]) + (t * help[j+1])
                    help.pop()
                else:
                    sum = help[0]
            spline_vertices.append(sum)
        drawCurve(vertices,(0.0, 0.7, 0.0, 0.1))
        drawCurve(spline_vertices,(0.1, 0.1, 0.6, 1.0))
        #drawCurve(spl2,(0.9, 0.9, 0.9, 1.0)) ################ TEST HERMITE
        #drawCurve(spl3,(0.9, 0.0, 0.9, 1.0)) ################ TEST BEZIER

        return {'FINISHED'}

classes = (MyProperties,DrawCustomSpline,NmenuAddon,
           ModalOperator,DrawBezierCurve)

def register():
    #bpy.types.VIEW3D_MT_object.append(menu_func)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=MyProperties)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.my_tool  # remove PG_MyProperties


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()
