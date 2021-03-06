
import array
import sys
import random
import json
import subprocess
import pandas

import numpy
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import hypervolume
import problem
import cma
import emo
import copy
from deap import creator
from deap import tools
from hv import HyperVolume
import functools

# Problem size
N=2



creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin, volViolation=0, valConstr=None, volOverBounds=0,
               isFeasible=True, A=None, z=None, sigma=None, parent_genome=None, parent_obj=None,parent_c=None,paretoRank = 0)


def zdt1(LOWBOUNDS,UPBOUNDS,ind):
    gnm = []
    conresult = []
    for i, num in enumerate(ind):
        if num < LOWBOUNDS[i]:
            gnm.append(LOWBOUNDS[i])
            conresult.append(num - LOWBOUNDS[i])
            ind.isFeasible = False
        elif num > UPBOUNDS[i]:
            gnm.append(UPBOUNDS[i])
            conresult.append(UPBOUNDS[i] - num)
            ind.isFeasible = False
        else:
            gnm.append(num)
            conresult.append(numpy.minimum(num - LOWBOUNDS[i], UPBOUNDS[i] - num))
        # gnm.append((UPBOUNDS[i] - LOWBOUNDS[i]) * (num / subD) + LOWBOUNDS[i])
    gnm = numpy.array(gnm)
    result = []
    result.append(gnm[0])
    dgnm = gnm / (len(gnm) - 1)
    g = 1 + 9 * numpy.sum(dgnm[1:])
    h = 1 - numpy.sqrt(gnm[0] / g)
    result.append(g * h)
    ind.valConstr = conresult
    return result

subD = 1
toolbox = base.Toolbox()

def loadBoundary():
    data = numpy.loadtxt('boundary.csv',delimiter=',',dtype=float)
    dlow = []
    dhigh = []
    for row in data:
        dlow.append(row[0])
        dhigh.append(row[1])
    return dlow,dhigh

def recHV(population,refs):
    truefront = emo.selFinal(population,200)
    if len(truefront) == 1 and not truefront[0].isFeasible:
        return truefront, 0
    else:
        dcfront = copy.deepcopy(truefront)
        tfPoint = []
        for i, ind in enumerate(dcfront):
            if not checkDuplication(ind, dcfront[:i]):
                tfPoint.append([ind.fitness.values[0],ind.fitness.values[1]])
        hy = HyperVolume(refs)
        HV = hy.compute(tfPoint)
        return truefront,HV

def checkDuplication(ind, front):
    for prep in front:
        count = 0
        for i in range(0, len(ind.fitness.values)):
            if str(prep.fitness.values[i])[:16] == str(ind.fitness.values[i])[:16]:
                count += 1
        if count == len(ind.fitness.values):
            return True
    return False


toolbox.register("evaluate", zdt1)
def main():

    # The cma module uses the numpy random number generator

    numpy.random.seed()
    LOWBOUNDS = numpy.zeros(N)
    UPBOUNDS = numpy.ones(N)
    iniUPBOUNDS  = [0.1,1e-5]
    iniUPBOUNDS=numpy.array(iniUPBOUNDS)
    MU, LAMBDA = 100, 1

    NGEN = 1000
    eval_log = numpy.empty((0,2) , float)
    verbose = True
    create_plot = True
    indlogs = list()
    trueParetoLog = []
    allTF = []
    HVlog = []
    ref = [1.0, 1.0]
    sigmas = []
    Ases = []
    detA = []
    sucrate = []
    first_pc=[]

    # The MO-CMA-ES algorithm takes a full population as argument
    population = [creator.Individual(x) for x in (numpy.random.uniform(0, iniUPBOUNDS, (MU, N)))]
    fitnesses = toolbox.map(functools.partial(toolbox.evaluate,LOWBOUNDS,UPBOUNDS), population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        for i, e in enumerate(ind):
            if 0 > ind.valConstr[i]:
                ind.volViolation += abs(ind.valConstr[i])
        if ind.volViolation > 0:
            ind.fitness.values = (ind.fitness.values[0] + ind.volViolation*1000, ind.fitness.values[1] + ind.volViolation*1000)
        eval_log = numpy.append(eval_log, numpy.array([fit]), axis=0)
        genom = [ind[i] for i in range(0,N)]
        if ind.isFeasible:
            indlogs.append([genom, fit, 1, ind.valConstr, 0])
        else:
            indlogs.append([genom, fit, 0, ind.valConstr, 0])

    strategy = cma.NaturalStrategyMultiObjective(population, sigma=1e-6, mu=MU, lambda_=LAMBDA)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    t0 ,h0 = recHV(population,ref)
    trueParetoLog.append(t0)
    for ind in t0:
        geno =[ind[i] for i in range(0,N)]
        if ind.isFeasible:
            allTF.append([geno, ind.fitness.values, 1, ind.valConstr, 0])
        else:
            allTF.append([geno, ind.fitness.values, 0,ind.valConstr, 0])
    HVlog.append(h0)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

   

    logbook = tools.Logbook()

    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    for gen in range(NGEN):
        # Generate a new population
        population1 = toolbox.generate()
        population = population1

        # Evaluate the individuals
        fitnesses = toolbox.map(functools.partial(toolbox.evaluate,LOWBOUNDS,UPBOUNDS), population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            for i, e in enumerate(ind):
                if 0 > ind.valConstr[i]:
                    ind.volViolation += abs(ind.valConstr[i])
            if ind.volViolation > 0:
                ind.fitness.values = (
                ind.fitness.values[0] + ind.volViolation * 1000, ind.fitness.values[1] + ind.volViolation * 1000)
                ind.isFeasible = False
            eval_log = numpy.append(eval_log, numpy.array([fit]), axis=0)
            genom = [ind[i] for i in range(0, N)]
            if ind.isFeasible:
                indlogs.append([genom, fit, 1,ind.valConstr, ind.parent_genome,ind.parent_obj,gen])
            else:
                indlogs.append([genom, fit, 0, ind.valConstr, ind.parent_genome,ind.parent_obj,gen])
        
        # Update the strategy with the evaluated individuals
        toolbox.update(population,UPBOUNDS=UPBOUNDS,LOWBOUNDS=LOWBOUNDS,evalfunc=zdt1)
        dcparent = copy.deepcopy(strategy.parents)
        dcparent = sorted(dcparent,key=lambda x:x[0])
        C = numpy.dot(dcparent[-1].A, dcparent[-1].A.T)
        a, b = scipy.linalg.eigh(C)
        a = a.tolist()
        Ases.append(a)
        d = numpy.hstack((b[:,0],b[:,1]))
        first_pc.append(d.tolist())
        genSigma = []
        genSigma.append(dcparent[-1].sigma)
        detA.append(scipy.linalg.det(dcparent[-1].A))
        sigmas.append(genSigma)
        sucrate.append([strategy.dominating_Success,strategy.success_outer,strategy.success,strategy.missed_both_alive_out,strategy.missed_both_alive_in, strategy.parentonly_alive_out,strategy.parentonly_alive_in,strategy.less_constraint])
        tGen, hGen = recHV(strategy.parents, ref)
        trueParetoLog.append(tGen)
        for ind in tGen:
            geno = [ind[i] for i in range(0, N)]
            if ind.isFeasible:
                allTF.append([geno, ind.fitness.values, 1, ind.valConstr,ind.parent_genome,ind.parent_obj, gen + 1])
            else:
                allTF.append([geno, ind.fitness.values, 0, ind.valConstr, ind.parent_genome,ind.parent_obj,gen + 1])
        HVlog.append(hGen)
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)
            print("Current population hypervolume is " + str(HVlog[-1]))

    if verbose:
        print("Final population hypervolume is " + str(HVlog[-1]))
        # Note that we use a penalty to guide the search to feasible solutions,

        # but there is no guarantee that individuals are valid.

        # We expect the best individuals will be within bounds or very close.
        print("Final population:")
        print(numpy.asarray(strategy.parents))

    trIndLog = transLogs(indlogs)
    df = pandas.DataFrame(trIndLog)

    return trueParetoLog[-1], eval_log, df, HVlog, allTF, sigmas, Ases,sucrate,first_pc,detA

def transLogs(logs):
    trdlogs = list()
    for ind in logs:
        log = []
        for i in range(0, len(ind)):
            if isinstance(ind[i], list) or isinstance(ind[i], tuple):
                for j in range(0, len(ind[i])):
                    log.append(ind[i][j])
            else:
                log.append(ind[i])
        trdlogs.append(log)
    return trdlogs

if __name__ == "__main__":
    num_exc = 10
    for reps in range(0,num_exc):
        solutions, fitness_history, df, HVlog, tflog, siglog, Alog,sucrate,firstPC,detA = main()
        df.to_csv("MONES"+"%03.f"%(reps)+".csv")
        dh = pandas.DataFrame(HVlog)
        dh.to_csv("MONESHV"+"%03.f"%(reps)+".csv")
        itlog = transLogs(tflog)
        dtr = pandas.DataFrame(itlog)
        dtr.to_csv("MONESTF" + "%03.f" % (reps) + ".csv")
        sigtr = transLogs(siglog)
        dsig = pandas.DataFrame(sigtr)
        dsig.to_csv("NESSigmaLog" + "%03.f" % (reps) + ".csv")
        Atr = transLogs(Alog)
        Ad = pandas.DataFrame(Atr)
        Ad.to_csv("NESALog" + "%03.f" % (reps) + ".csv")
        tsucrate = transLogs(sucrate)
        tsucrated = pandas.DataFrame(tsucrate)
        tsucrated.to_csv("NESsuccessRateLog" + "%03.f" % (reps) + ".csv")
        tfpc = transLogs(firstPC)
        ttfpc = pandas.DataFrame(tfpc)
        ttfpc.to_csv("NESFirstPC" + "%03.f" % (reps) + ".csv")
        detAd = pandas.DataFrame(detA)
        detAd.to_csv("NESdetALog" + "%03.f" % (reps) + ".csv")

        fig = plt.figure()
        plt.title("Multi-objective minimization via MO-NES")
        plt.xlabel("f1")
        plt.ylabel("f2")
        # Limit the scale because our history values include the penalty.
        plt.xlim((0, 2))
        plt.ylim((0, 2))
        plt.grid(True)
        # Plot all history. Note the values include the penalty.
        fitness_history = numpy.asarray(fitness_history)
        plt.scatter(fitness_history[:,0], fitness_history[:,1],facecolors='none', edgecolors="lightblue",s=6)
        valid_front = numpy.array([ind.fitness.values for ind in solutions])
        plt.scatter(valid_front[:,0], valid_front[:,1], c="g",s=6)
        print("Writing cma_mo.png")
        plt.savefig("MONES"+"%03.f"%(reps)+".png", dpi=300)
        plt.close(fig)

        hvfig = plt.figure()
        plt.title("HV/Generation")
        plt.xlabel("Generation")
        plt.ylabel("HV")
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.ylim((0,1))
        GEN = numpy.array([i for i in range(1,len(HVlog)+1)])
        plt.plot(GEN,HVlog)
        plt.grid(True)
        plt.savefig("MONESHV" + "%03.f" % (reps) + ".png", dpi=300)
        plt.close(hvfig)