
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
from pyretis.forcefield import PotentialFunction
import math

@np.vectorize
def easom_potential(x,y):
    return -np.cos(x)*np.cos(y)*np.exp(-(x-math.pi)**2-(y-math.pi)**2);

class Easom(PotentialFunction):
    def __init__(self):
        super().__init__(dim=2, desc='The Easom function')

    def potential(self, system):
        xpos = system.particles.pos[:, 0]
        ypos = system.particles.pos[:, 1]
        pot = easom_potential(xpos, ypos)
        return pot


def main():
    from ParticleSwarmOptimization_EvolutionaryStrategy.function.settingFunction import setting_function
    MINX, MAXX = setting_function()
    xgrid, ygrid = np.meshgrid(np.linspace(MINX, MAXX, 100),
                               np.linspace(MINX, MAXX, 100))
    zgrid = easom_potential(xgrid, ygrid)

    fig = plt.figure()
    plt.title("Easom Function")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, projection='3d')
    ax1.contourf(xgrid, ygrid, zgrid)
    ax2.plot_surface(xgrid, ygrid, zgrid, cmap=plt.get_cmap('viridis'))
    # plt.savefig('data\EasomFunction.png')
    plt.show()
    # plt.close()

if __name__ == '__main__':
    main()