import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
from pyretis.forcefield import PotentialFunction


TWO_PI = np.pi * 2.0
EXP = np.exp(1)


@np.vectorize
def ackley_potential(x, y):  # pylint: disable=invalid-name
    return (-20.0 * np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) -
            np.exp(0.5 * (np.cos(TWO_PI * x) + np.cos(TWO_PI * y))) +
            EXP + 20)


class Ackley(PotentialFunction):
    def __init__(self):
        super().__init__(dim=2, desc='The Ackley function')

    def potential(self, system):
        xpos = system.particles.pos[:, 0]
        ypos = system.particles.pos[:, 1]
        pot = ackley_potential(xpos, ypos)
        return pot

def main():
    from ParticleSwarmOptimization_EvolutionaryStrategy.function.settingFunction import setting_function
    MINX, MAXX = setting_function()
    xgrid, ygrid = np.meshgrid(np.linspace(MINX, MAXX, 100),
                               np.linspace(MINX, MAXX, 100))
    zgrid = ackley_potential(xgrid, ygrid)

    fig = plt.figure()
    plt.title("Acley Function")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, projection='3d')
    ax1.contourf(xgrid, ygrid, zgrid)
    ax2.plot_surface(xgrid, ygrid, zgrid, cmap=plt.get_cmap('viridis'))
    # plt.savefig('data\AckleyFunction.png')
    plt.show()
    # plt.close()

if __name__ == '__main__':
    main()