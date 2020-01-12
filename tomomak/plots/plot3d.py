import numpy as np
import mayavi.mlab
import random


z_lim = 0
intensity_threshold = 1

def detector_colormesh3d(data, axis1, axis2, axis3, title, *args, **kwargs):

    #           Z
    #           ^
    #           | pt1_ _ _ _ _ _ _ _ _ _ _pt2
    #           |  /|    Y                /|
    #           | / |   ^                / |
    #         pt3/_ | _/_ _ _ _ _ _ _pt4/  |
    #           |   | /                |   |
    #           |   |/                 |   |
    #           |  pt5_ _ _ _ _ _ _ _ _|_ _|pt6
    #           |  /                   |  /
    #           | /                    | /
    #        pt7|/_ _ _ _ _ _ _ _ _ _ _|/pt8_______\X
    #                                              /
    
    verts = axis1.cell_edges3d(axis2,axis3)
    mayavi.mlab.figure(title)
    print(axis1.units)
    mayavi.mlab.axes(xlabel="sas")
    
    
    # generate data values for test
    #for i in range(len(data)):
    #    for j in range(len(data[0])):
    #        for k in range(len(data[0][0])):
    #            data[i][j][k] = random.random()
    #i=j=k=0
    counter = 0
    x_lim = 1 # len(verts)
    y_lim = 1 # len(verts[0])
    z_lim = 1 # len(verts[0][0])

    for i in range(x_lim):
        for j in range(y_lim):
            for k in range(z_lim):
                if (i > 0 and j > 0 and k > 0) and (i < x_lim-1 and j < y_lim-1 and k < z_lim-1):
                    continue
                print(counter)
                counter+=1
                x1, y1, z1 = verts[i][j][k][6]  # | => pt1 (0, 1, 1)  
                x2, y2, z2 = verts[i][j][k][4]  # | => pt2 (1, 1, 1)  
                x3, y3, z3 = verts[i][j][k][7]  # | => pt3 (0, 0, 1)  
                x4, y4, z4 = verts[i][j][k][3]  # | => pt4 (1, 0, 1)  
                x5, y5, z5 = verts[i][j][k][5]  # | => pt5 (0, 1, 0)  
                x6, y6, z6 = verts[i][j][k][2]  # | => pt6 (1, 1, 0)  
                x7, y7, z7 = verts[i][j][k][0]  # | => pt7 (0, 0, 0)  
                x8, y8, z8 = verts[i][j][k][1]  # | => pt8 (1, 0, 0)  

                m_color = (data[i][j][k],data[i][j][k],data[i][j][k])
                if(k==z_lim-1):
                    mayavi.mlab.mesh([[x1, x2],
                                    [x3, x4]],  # | => x coordinate

                                    [[y1, y2],
                                    [y3, y4]],  # | => y coordinate

                                    [[z1, z2],
                                    [z3, z4]],  # | => z coordinate
                                    color=m_color)

                # Where each point will be connected with this neighbors :
                # (link = -)
                #
                # x1 - x2     y1 - y2     z1 - z2 | =>  pt1 - pt2
                # -    -  and  -   -  and -    -  | =>   -     -
                # x3 - x4     y3 - y4     z3 - z4 | =>  pt3 - pt4

                if(k==0):
                    mayavi.mlab.mesh([[x5, x6], [x7, x8]],
                                    [[y5, y6], [y7, y8]],
                                    [[z5, z6], [z7, z8]],
                                    color=m_color)
                if(i==0):
                    mayavi.mlab.mesh([[x1, x3], [x5, x7]],
                                    [[y1, y3], [y5, y7]],
                                    [[z1, z3], [z5, z7]],
                                    color=m_color)
                if(j==y_lim-1):
                    mayavi.mlab.mesh([[x1, x2], [x5, x6]],
                                    [[y1, y2], [y5, y6]],
                                    [[z1, z2], [z5, z6]],
                                    color=m_color)
                if(i==x_lim-1):
                    mayavi.mlab.mesh([[x2, x4], [x6, x8]],
                                    [[y2, y4], [y6, y8]],
                                    [[z2, z4], [z6, z8]],
                                    color=m_color)

                if(j==0):
                    mayavi.mlab.mesh([[x3, x4], [x7, x8]],
                                    [[y3, y4], [y7, y8]],
                                    [[z3, z4], [z7, z8]],
                                    color=m_color)

    mayavi.mlab.show()
    



# slider control example
# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# Copyright (c) 2008, Enthought, Inc.
# License: BSD Style.
"""




dphi = pi/1000.
phi = arange(0.0, 2*pi + 0.5*dphi, dphi, 'd')

def curve(n_mer, n_long):
    mu = phi*n_mer
    x = cos(mu) * (1 + cos(n_long * mu/n_mer)*0.5)
    y = sin(mu) * (1 + cos(n_long * mu/n_mer)*0.5)
    z = 0.5 * sin(n_long*mu/n_mer)
    t = sin(mu)
    return x, y, z, t


class MyModel(HasTraits):
    n_meridional    = Range(0, 30, 6, )#mode='spinner')
    n_longitudinal  = Range(0, 30, 11, )#mode='spinner')

    scene = Instance(MlabSceneModel, ())

    plot = Instance(PipelineBase)


    # When the scene is activated, or when the parameters are changed, we
    # update the plot.
    @on_trait_change('n_meridional,n_longitudinal,scene.activated')
    def update_plot(self):
        x, y, z, t = curve(self.n_meridional, self.n_longitudinal)
        if self.plot is None:
            self.plot = self.scene.mlab.plot3d(x, y, z, t,
                                tube_radius=0.025, colormap='Spectral')
        else:
            self.plot.mlab_source.trait_set(x=x, y=y, z=z, scalars=t)


    # The layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                Group(
                        '_', 'n_meridional', 'n_longitudinal',
                     ),
                resizable=True,
                )

my_model = MyModel()
my_model.configure_traits()"""