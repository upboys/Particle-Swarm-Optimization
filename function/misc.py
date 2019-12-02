
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
from pyretis.forcefield import PotentialFunction

@np.vectorize
def misc_potential(x, y):
    r2 = x**2 + y**2
    return(-(x**2 + 3*y**2)/(0.1 + r2) - (x**2 + 5*y**2)*np.exp(1-r2)/2)

class Misk(PotentialFunction):
    def __init__(self):
        super().__init__(dim=2, desc='The Misk function')

    def potential(self, system):
        xpos = system.particles.pos[:, 0]
        ypos = system.particles.pos[:, 1]
        pot = misc_potential(xpos, ypos)
        return pot


def main():
    from ParticleSwarmOptimization_EvolutionaryStrategy.function.settingFunction import setting_function
    MINX, MAXX = setting_function()
    xgrid, ygrid = np.meshgrid(np.linspace(MINX, MAXX, 100),
                               np.linspace(MINX, MAXX, 100))
    zgrid = misc_potential(xgrid, ygrid)

    fig = plt.figure()
    plt.title("Misc Function")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, projection='3d')
    ax1.contourf(xgrid, ygrid, zgrid)
    ax2.plot_surface(xgrid, ygrid, zgrid, cmap=plt.get_cmap('viridis'))
    # plt.savefig('data\MiscFunction.png')
    plt.show()
    # plt.close()

if __name__ == '__main__':
    main()