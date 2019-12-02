import numpy as np
from pyretis.core import create_box, Particles, System
from pyretis.simulation import Simulation
from pyretis.forcefield import ForceField


from ParticleSwarmOptimization_EvolutionaryStrategy.util.psoengine import PSOEngine

NPART = 30
STEPS = 1000
TXT = 'Step: {:5d}: Best: (x, y) = ({:10.3e}, {:10.3e}), pot = {:10.3e}'


ackley=True
easom=False
eggholder=False
himmelblau=False
holderTable=False
misc=False
rastrigin=False
styblinski=False
def setting_function():
    if (ackley):
        MINX, MAXX = -5, 5
        return MINX, MAXX
    if (easom):
        MINX, MAXX = -10, 10
        return MINX, MAXX
    if (eggholder):
        MINX, MAXX = -512, 512
        return MINX, MAXX
    if (himmelblau):
        MINX, MAXX = -5, 5
        return MINX, MAXX
    if (holderTable):
        MINX, MAXX = -10, 10
        return MINX, MAXX
    if (misc):
        MINX, MAXX = -5, 5
        return MINX, MAXX
    if(rastrigin):
        MINX, MAXX = -5.12, 5.12
        return MINX, MAXX
    if (styblinski):
        MINX, MAXX = -5, 5
        return MINX, MAXX

def set_up():
    """Just create system and simulation."""
    MINX,MAXX=setting_function()
    box = create_box(low=[MINX, MINX], high=[MAXX, MAXX],
                     periodic=[True, True])
    print('Created a box:')
    print(box)

    print('Creating system with {} particles'.format(NPART))
    system = System(units='reduced', box=box)
    system.particles = Particles(dim=2)
    for _ in range(NPART):
        pos = np.random.uniform(low=MINX, high=MAXX, size=(1, 2))
        system.add_particle(pos)
    ffield = ForceField('Single '+str(function_name())+' function',
                        potential=[potential_function()])
    system.forcefield = ffield
    print('Force field is:\n{}'.format(system.forcefield))

    print('Creating simulation:')
    engine = PSOEngine(0.7, 1.5, 1.5)
    simulation = Simulation(steps=STEPS)
    task_integrate = {'func': engine.integration_step,
                      'args': [system],
                      'result': 'gbest', 'first': True}
    simulation.add_task(task_integrate)
    return simulation, system

def potential_function():
    if (ackley):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.ackley import Ackley
        return Ackley()
    if (easom):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.easom import Easom
        return Easom()
    if (eggholder):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.eggholder import Eggholder
        return Eggholder()
    if (himmelblau):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.himmelblau import Himmelblau
        return Himmelblau()
    if (holderTable):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.holderTable import HolderTable
        return HolderTable()
    if (misc):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.misc import Misk
        return Misk()
    if(rastrigin):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.rastrigin import Rastrigin
        return Rastrigin()
    if (styblinski):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.styblinski import Styblinski
        return Styblinski()

def function_name():
    if (ackley):
        return 'Ackley'
    if (easom):
        return 'Easom'
    if (eggholder):
        return 'Eggholder'
    if (himmelblau):
        return 'Himmelblau'
    if (holderTable):
        return 'HolderTable'
    if (misc):
        return 'Misc'
    if(rastrigin):
        return 'Rastrigin'
    if (styblinski):
        return 'Styblinski'

def evaluate_potential_grid(X, Y):
    """Evaluate the Ackley potential on a grid"""
    # MINX, MAXX = setting_function()
    # X, Y = np.meshgrid(np.linspace(MINX, MAXX, 100),
    #                    np.linspace(MINX, MAXX, 100))
    if (ackley):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.ackley import ackley_potential
        Z = ackley_potential(X, Y)
    if (easom):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.easom import easom_potential
        Z = easom_potential(X, Y)
    if (eggholder):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.eggholder import eggholder_potential
        Z = eggholder_potential(X, Y)
    if (himmelblau):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.himmelblau import himmelblau_potential
        Z = himmelblau_potential(X, Y)
    if (holderTable):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.holderTable import holderTable_potential
        Z = holderTable_potential(X, Y)
    if (misc):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.misc import misc_potential
        Z = misc_potential(X, Y)
    if(rastrigin):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.rastrigin import rastrigin_potential
        Z = rastrigin_potential(X, Y)
    if (styblinski):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.styblinski import styblinski_potential
        Z = styblinski_potential(X, Y)
    return X, Y, Z

def chartFunction():
    if (ackley):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.ackley import  main
        main()
    if (easom):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.easom import  main
        main()
    if (eggholder):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.eggholder import  main
        main()
    if (himmelblau):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.himmelblau import  main
        main()
    if (holderTable):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.holderTable import  main
        main()
    if (misc):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.misc import  main
        main()
    if(rastrigin):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.rastrigin import  main
        main()
    if (styblinski):
        from ParticleSwarmOptimization_EvolutionaryStrategy.function.styblinski import  main
        main()
