

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
from pyretis.forcefield import PotentialFunction
import math


@np.vectorize
def himmelblau_potential(x,y):
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b

class Himmelblau(PotentialFunction):
    def __init__(self):
        super().__init__(dim=2, desc='The Himmelblau function')

    def potential(self, system):
        xpos = system.particles.pos[:, 0]
        ypos = system.particles.pos[:, 1]
        pot = himmelblau_potential(xpos, ypos)
        return pot


def main():
    from ParticleSwarmOptimization_EvolutionaryStrategy.function.settingFunction import setting_function
    MINX, MAXX = setting_function()
    xgrid, ygrid = np.meshgrid(np.linspace(MINX, MAXX, 100),
                               np.linspace(MINX, MAXX, 100))
    zgrid = himmelblau_potential(xgrid, ygrid)

    fig = plt.figure()
    plt.title("Himmelblau Function")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, projection='3d')
    ax1.contourf(xgrid, ygrid, zgrid)
    ax2.plot_surface(xgrid, ygrid, zgrid, cmap=plt.get_cmap('viridis'))
    # plt.savefig('data\HimmelblauFunction.png')
    plt.show()
    # plt.close()

if __name__ == '__main__':
    main()