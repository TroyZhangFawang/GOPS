from gops.utils.planner_benchmark.visualize.common import *
from gops.utils.planner_benchmark.visualize.line import *
from gops.utils.planner_benchmark.visualize.point import *
from gops.utils.planner_benchmark.visualize.surface import *
from gops.utils.planner_benchmark.visualize.surface3d import *
from gops.utils.planner_benchmark.visualize.elements import *

import matplotlib.pyplot as plt

def __getattr__(name):
    return getattr(plt, name)
    # if name == 'show':
    #     return plt.show
    # elif name == 'savefig':
    #     return plt.savefig
    # elif name == 'close':
    #     return plt.close
    # else:
    #     import warnings
    #     warnings.warn("{} is actually from matplotlib.pyplot. Please try to import plt directly.".format(name))
    #     return getattr(plt, name)

