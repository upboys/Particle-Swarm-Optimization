import numpy as np

from ParticleSwarmOptimization_EvolutionaryStrategy.function.settingFunction import setting_function, TXT, set_up, \
    STEPS, function_name
from ParticleSwarmOptimization_EvolutionaryStrategy.function.settingFunction import  evaluate_potential_grid
from matplotlib import pyplot as plt
from matplotlib import animation, cm

MINX, MAXX=setting_function()
def update_animation(frame, system, simulation, scatter):
    patches = []
    if not simulation.is_finished() and frame > 0:
        results = simulation.step()
        best = results['gbest']
        if frame % 10 == 0:
            print(TXT.format(frame, best[1][0], best[1][1], best[0]))
    scatter.set_offsets(system.particles.pos)
    patches.append(scatter)
    return patches


def main_animation(simulation, system,name):
    # simulation, system = set_up()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    ax1.set_xlim((MINX, MAXX))
    ax1.set_ylim((MINX, MAXX))
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
	
    X, Y = np.meshgrid(np.linspace(MINX, MAXX, 100),
                       np.linspace(MINX, MAXX, 100))
    X, Y, pot = evaluate_potential_grid(X, Y)
    ax1.contourf(X, Y, pot, cmap=cm.viridis, zorder=1)
    scatter = ax1.scatter(system.particles.pos[:, 0],
                          system.particles.pos[:, 1], marker='o', s=50,
                          edgecolor='#262626', facecolor='white')

    def init():
        return [scatter]
    anim = animation.FuncAnimation(fig, update_animation,
                                   frames=STEPS+1,
                                   fargs=[system, simulation, scatter],
                                   repeat=False, interval=30, blit=True,
                                   init_func=init)

    plt.savefig('data'+'\\'+function_name()+'\\'+name+'.png')
    # plt.show()
    plt.close()
    return anim


if __name__ == '__main__':
    simulation, system = set_up()
    main_animation(simulation, system,function_name()+"_PopulationInit")