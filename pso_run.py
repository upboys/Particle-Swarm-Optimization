from ParticleSwarmOptimization_EvolutionaryStrategy.util.draw import main_animation
from ParticleSwarmOptimization_EvolutionaryStrategy.function.settingFunction import TXT, set_up, chartFunction, \
    function_name


def naive_variance(data):
    n = 0
    Sum = 0
    Sum_sqr = 0

    for x in data:
        n = n + 1
        Sum = Sum + x
        Sum_sqr = Sum_sqr + x * x

    variance = (Sum_sqr - (Sum * Sum) / n) / (n - 1)
    return variance

def average(date):
    Sum=0
    for x in date:
        Sum+=x
    return  Sum/len(date)

def mainPso():
    simulation, _ = set_up()
    res=None
    counter=1
    for result in simulation.run():
        step = result['cycle']['step']
        best = result['gbest']
        # if(counter<=150):
        #     main_animation(result, _,function_name()+str(counter))
        counter=counter+1
        res=result
        if step % 10 == 0:
            print(TXT.format(step, best[1][0], best[1][1], best[0]))

    # main_animation(res, _,function_name()+"_GlobalBest")
    return best[0],best[1]

arrFitness=[]
arrPosition=[]

if __name__ == '__main__':
    step = 30
    for i in range(step):
        fitness, position = mainPso()
        arrFitness.append(fitness)
        arrPosition.append(position)
        # chartFunction()
        # simulation, _ = set_up()
        # main_animation(simulation, _ ,function_name()+"_PopulationInit")

print("==================================================================")
for i in range(step):
    print('step'+str(i+1))
    print('position: '+str(arrPosition[i]))
    print('bestFitness: '+ str(arrFitness[i]))
print("==================================================================")
print(arrFitness)
print('Variance: '+str(naive_variance(arrFitness)))
print('Average: '+str(average(arrFitness)))




