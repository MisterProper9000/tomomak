import numpy as np
import mayavi.mlab

from traits.api import HasTraits, Int, Range, Instance, \
        on_trait_change
from traitsui.api import View, Item

from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
                MlabSceneModel


class Plot3d(HasTraits):
    z_lim_range = Int()
    R = Range(low=1,high='z_lim_range',value=1)

    scene = Instance(MlabSceneModel, ())

    #plot = Instance(PipelineBase)

    

    def init(self, _data, axis1, axis2, axis3, _title, *args, **kwargs):
        self.title = _title
        self.data = _data
        self.verts = axis1.cell_edges3d(axis2,axis3)
        #mayavi.mlab.title(self.title)
       

    # When the scene is activated, or when the parameters are changed, we
    # update the plot.
    #
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
    
    @on_trait_change('R,scene.activated')
    def update_plot(self):
        mayavi.mlab.clf()
        counter=0
       
        x_lim = len(self.verts)
        y_lim = len(self.verts[0])
        z_lim = self.R

        for i in range(x_lim):
            for j in range(y_lim):
                for k in range(z_lim):
                    if (i > 0 and j > 0 and k > 0) and (i < x_lim-1 and j < y_lim-1 and k < z_lim-1):
                        continue
                    print(counter)
                    counter+=1
                    x1, y1, z1 = self.verts[i][j][k][6]  # | => pt1 (0, 1, 1)  
                    x2, y2, z2 = self.verts[i][j][k][4]  # | => pt2 (1, 1, 1)  
                    x3, y3, z3 = self.verts[i][j][k][7]  # | => pt3 (0, 0, 1)  
                    x4, y4, z4 = self.verts[i][j][k][3]  # | => pt4 (1, 0, 1)  
                    x5, y5, z5 = self.verts[i][j][k][5]  # | => pt5 (0, 1, 0)  
                    x6, y6, z6 = self.verts[i][j][k][2]  # | => pt6 (1, 1, 0)  
                    x7, y7, z7 = self.verts[i][j][k][0]  # | => pt7 (0, 0, 0)  
                    x8, y8, z8 = self.verts[i][j][k][1]  # | => pt8 (1, 0, 0)  

                    m_color = (self.data[i][j][k],self.data[i][j][k],self.data[i][j][k])
                    if(k==z_lim-1):
                        mayavi.mlab.mesh([[x1, x2],
                                        [x3, x4]],  # | => x coordinate

                                        [[y1, y2],
                                        [y3, y4]],  # | => y coordinate

                                        [[z1, z2],
                                        [z3, z4]],  # | => z coordinate
                                        color=m_color, 
                                        representation = 'fancymesh') # wireframe or surface

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
                                        color=m_color,
                                        representation = 'fancymesh')
                    if(i==0):
                        mayavi.mlab.mesh([[x1, x3], [x5, x7]],
                                        [[y1, y3], [y5, y7]],
                                        [[z1, z3], [z5, z7]],
                                        color=m_color,
                                        representation = 'fancymesh')
                    if(j==y_lim-1):
                        mayavi.mlab.mesh([[x1, x2], [x5, x6]],
                                        [[y1, y2], [y5, y6]],
                                        [[z1, z2], [z5, z6]],
                                        color=m_color,
                                        representation = 'fancymesh')
                    if(i==x_lim-1):
                        mayavi.mlab.mesh([[x2, x4], [x6, x8]],
                                        [[y2, y4], [y6, y8]],
                                        [[z2, z4], [z6, z8]],
                                        color=m_color,
                                        representation = 'fancymesh')

                    if(j==0):
                        mayavi.mlab.mesh([[x3, x4], [x7, x8]],
                                        [[y3, y4], [y7, y8]],
                                        [[z3, z4], [z7, z8]],
                                        color=m_color,
                                        representation = 'fancymesh')


    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                Item('R'),
                resizable=True,
                )

def detector_colormesh3d(data, axis1, axis2, axis3, title, *args, **kwargs):
    print(len(axis1.cell_edges3d(axis2,axis3)[0][0]))
    my_model = Plot3d(z_lim_range=len(axis1.cell_edges3d(axis2,axis3)[0][0]))
    my_model.init(data,axis1, axis2, axis3, title, *args, **kwargs)
    my_model.configure_traits()