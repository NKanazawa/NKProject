#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

#    Special thanks to Nikolaus Hansen for providing major part of
#    this code. The CMA-ES algorithm is provided in many other languages
#    and advanced versions at http://www.lri.fr/~hansen/cmaesintro.html.

"""A module that provides support for the Covariance Matrix Adaptation
Evolution Strategy.
"""
import copy
from math import sqrt, log, exp
import numpy
from scipy.linalg import expm
from scipy import dot, array, randn, eye, outer, exp, trace, floor, log, sqrt
import scipy

import functools
from operator import itemgetter
from operator import attrgetter
from deap import tools
from deap.tools._hypervolume import pyhv
import emo
import problem

class NaturalStrategyMultiObjective(object):
    def __init__(self, population, sigma, **params):
        self.parents = population
        self.dim = len(self.parents[0])

        # Selection
        self.mu = params.get("mu", len(self.parents))
        self.lambda_ = params.get("lambda_")
        #Learning rate
        self.etasigma = (3+numpy.log(self.dim))/(4+3*numpy.log(self.dim))/pow(self.dim,1.5)
        self.etaA = self.etasigma
        self.eps = numpy.sqrt(self.dim)*(1-1/(4*self.dim)+1/(21*numpy.power(self.dim,2)))

        # Internal parameters associated to the mu parent
        self.initSigmas = sigma

        #集計パラメータ
        self.dominating_Success = 0
        self.less_constraint = 0
        self.success_outer = 0
        self.success = 0
        self.missed_both_alive_out = 0
        self.missed_both_alive_in = 0
        self.parentonly_alive_out = 0
        self.parentonly_alive_in = 0

        #進化パス
        self.evolpath = numpy.zeros(self.dim)

        self.indicator = params.get("indicator", tools.hypervolume)

    def generate(self, ind_init):
        """Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals with a private attribute :attr:`_ps`.
                  This last attribute is essential to the update function, it
                  indicates that the individual is an offspring and the index
                  of its parent.
        """

        arz = numpy.random.randn(self.lambda_, self.dim)
        individuals = list()

        # Make sure every parent has a parent tag and index
        self.parents=sorted(self.parents, key=lambda x: x[0])
        for i, p in enumerate(self.parents):
            p._ps = "p", i
            p.Rank = 0


        # Each parent produce an offspring
        for i in range(self.lambda_):
            # print "Z", list(arz[i])
            if self.parents[-1].A is None:
                self.parents[-1].A = numpy.identity(self.dim)
                self.parents[-1].sigma = self.initSigmas
                self.parents[-1].invA = numpy.identity(self.dim)
                self.parents[-1].logdetA = 0

            cparent = copy.deepcopy(self.parents[-1])
            individuals.append(ind_init(cparent + cparent.sigma * numpy.dot(cparent.A, arz[-1])))
            individuals[-1].theta = arz[-1]
            individuals[-1]._ps = "o", i
            individuals[-1].Rank = 0
            individuals[-1].contr = 0
            individuals[-1].A = cparent.A
            individuals[-1].sigma = cparent.sigma
            individuals[-1].parent_genome = []
            individuals[-1].parent_c = cparent.parent_c
            for mat in cparent:
                individuals[-1].parent_genome.append(mat)
            individuals[-1].parent_obj = cparent.fitness.values
        return individuals

    def _select(self, pop):
        """

        :type pop: list
        """
        candidates = self.parents + pop
        isPlenty = False
        if len(candidates) <= self.mu:
            return candidates, []

        pareto_fronts = emo.sortNondominated(candidates, len(candidates))
        if len(pareto_fronts[0]) > self.mu:
            isPlenty = True
        for i,front in enumerate(pareto_fronts):
            if len(front) == 1:
                for j in pareto_fronts[:i]:
                    if front[0]._ps[0] == "o":
                        pop[front[0]._ps[1]].Rank += len(j)
                    else:
                        self.parents[front[0]._ps[1]].Rank += len(j)
                if front[0]._ps[0] == "o":
                    pop[front[0]._ps[1]].Rank += 1
                    pop[front[0]._ps[1]].paretoRank = i + 1
                    pop[front[0]._ps[1]].contr = 0
                else:
                    self.parents[front[0]._ps[1]].Rank += 1
                    self.parents[front[0]._ps[1]].paretoRank = i + 1
                    self.parents[front[0]._ps[1]].contr = 0
            else:
                for m in range(0,len(front)):
                    for j in pareto_fronts[:i]:
                        if front[m]._ps[0] == "o":
                            pop[front[m]._ps[1]].Rank += len(j)
                        else:
                            self.parents[front[m]._ps[1]].Rank += len(j)
                wobjs = numpy.array([ind.fitness.wvalues for ind in front]) * -1
                refs = numpy.max(wobjs, axis=0) + 1

                def calContribution(idx, wobj, ref):
                    return pyhv.hypervolume(numpy.concatenate((wobj[:idx], wobj[idx + 1:])), ref)

                cont = list(map(functools.partial(calContribution,ref=refs,wobj=wobjs),list(range(len(front)))))
                for m in range(0,len(front)):
                    front[m].contr = cont[m]
                front.sort(key=attrgetter("contr"))
                for m in range(0,len(front)):
                    if front[m]._ps[0] == "o":
                        pop[front[m]._ps[1]].Rank += m+1
                        pop[front[m]._ps[1]].paretoRank = i + 1
                    else:
                        self.parents[front[m]._ps[1]].Rank += m+1
                        self.parents[front[m]._ps[1]].paretoRank = i + 1
        chosen = list()
        mid_front = None
        not_chosen = list()

        # Fill the next population (chosen) with the fronts until there is not enouch space
        # When an entire front does not fit in the space left we rely on the hypervolume
        # for this front
        # The remaining fronts are explicitely not chosen
        full = False
        for front in pareto_fronts:
            if len(chosen) + len(front) <= self.mu and not full:
                chosen += front
            elif mid_front is None and len(chosen) < self.mu:
                mid_front = front
                # With this front, we selected enough individuals
                full = True
            else:
                not_chosen += front

        # Separate the mid front to accept only k individuals
        k = self.mu - len(chosen)
        if k > 0:
            ref = numpy.array([ind.fitness.wvalues for ind in candidates]) * -1
            ref = numpy.max(ref, axis=0) + 1

            for _ in range(len(mid_front) - k):
                idx = self.indicator(mid_front, ref=ref)
                not_chosen.append(mid_front.pop(idx))
            chosen += mid_front
        return chosen, not_chosen,isPlenty



    def update(self, population,UPBOUNDS=None,LOWBOUNDS=None,evalfunc=None):
        """Update the current covariance matrix strategies from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """

        chosen, not_chosen,isPlenty = self._select(population)
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        # Update the internal parameters for successful offspring


        # Only the offspring update the parameter set
        if population[-1].Rank < self.parents[-1].Rank and population[-1].Rank <= self.mu:
            gm = numpy.outer(population[-1].theta, population[-1].theta) - numpy.identity(self.dim)
            gsigma = numpy.trace(gm) / self.dim
            ga = gm - gsigma * numpy.identity(self.dim)
            population[-1].sigma = population[-1].sigma * exp(self.etasigma * gsigma / 2.0)
            proc = 0.5 * (self.etaA * ga)
            GGA = scipy.linalg.expm(proc)
            population[-1].A = numpy.dot(population[-1].A, GGA)
            if self.dominates(population[-1],self.parents[-1]):
                count7 += 1
                chosen[population[-1].Rank-1] = self.mupdate(population[-1], self.parents[-1],LOWBOUNDS,UPBOUNDS,evalfunc)
            else:
                if numpy.sum(population[-1].valConstr[1:]) < numpy.sum(self.parents[-1].valConstr[1:]):
                    count8 += 1
                self.evolpath = (1-2/(self.dim+2))*self.evolpath
                chosen[population[-1].Rank - 1] = self.pullupdate(population[-1],LOWBOUNDS,UPBOUNDS,evalfunc)
            if gsigma > 0 :
                count1+=1
            else:count2 += 1

        elif population[-1].Rank > self.parents[-1].Rank and population[-1].Rank <= self.mu:
            gm = numpy.outer(population[-1].theta, population[-1].theta) - numpy.identity(self.dim)
            gsimga = numpy.trace(gm) / self.dim
            if gsimga > 0:
                count3 += 1
            else:
                count4 += 1
        elif population[-1].Rank > self.mu and self.parents[-1].Rank <= self.mu:
            gm = numpy.outer(population[-1].theta, population[-1].theta) - numpy.identity(self.dim)
            gsigma = numpy.trace(gm) / self.dim
            if gsigma > 0 :
                count5+=1
            else:count6 += 1
        else:
            print(str(population[-1].Rank)+" and parent achieved "+str(self.parents[-1].Rank))

        print(self.evolpath)
        self.dominating_Success = count7
        self.success_outer = count1
        self.success = count2
        self.missed_both_alive_out = count3
        self.missed_both_alive_out = count4
        self.parentonly_alive_out = count5
        self.parentonly_alive_in = count6
        self.less_constraint = count8
        self.parents = copy.deepcopy(chosen)

    def dominates(self,My, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        if other.isFeasible and not My.isFeasible:
            return False
        for self_wvalue, other_wvalue in zip(My.fitness.wvalues[obj], other.fitness.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    def mupdate(self,this,parent,LOW,UP,evals):
        thisgnm = numpy.array([this[i] for i in range(self.dim)])
        parentgnm = numpy.array([parent[i] for i in range(self.dim)])
        direction = (thisgnm-parentgnm) / numpy.linalg.norm(thisgnm-parentgnm)
        current = copy.deepcopy(this)
        dominate = False
        notdominate = False
        alpha = current.sigma
        while True:
            next = copy.deepcopy(current)
            for i,_ in enumerate(next):
                next[i] = next[i] + alpha * direction[i]
            next.fitness.values = evals(LOW,UP,next)
            if self.dominates(next,current):
                dominate = True
                current = next
            else:
                notdominate = True
                alpha = alpha * 0.5
            if dominate and notdominate:
                break
        self.evolpath=(1-2/(self.dim+2))*self.evolpath+numpy.sqrt(2/(self.dim+2)*(2-2/(self.dim+2)))* ((thisgnm-parentgnm) / current.sigma)
        C = numpy.dot(current.A,current.A.T)
        C = (1-2/(numpy.power(self.dim,2)+6))* C + (2/(numpy.power(self.dim,2)+6))*numpy.dot(self.evolpath,self.evolpath.T)
        current.A = numpy.linalg.cholesky(C)
        current.A = current.A / numpy.linalg.det(current.A)
        return current

    def pullupdate(self,this,LOW,UP,evals):
        if numpy.linalg.norm(self.evolpath) < 1e-32:
            return this
        direction = self.evolpath / numpy.linalg.norm(self.evolpath)
        current = copy.deepcopy(this)
        dominate = False
        notdominate = False
        alpha = current.sigma
        while True:
            next = copy.deepcopy(current)
            for i,_ in enumerate(next):
                next[i] = next[i] + alpha * direction[i]
            next.fitness.values = evals(LOW,UP,next)
            if self.dominates(next,current):
                dominate = True
                current = next
            else:
                notdominate = True
                alpha = alpha * 0.5
            if dominate and notdominate:
                break
        C = numpy.dot(current.A,current.A.T)
        C = (1-2/(numpy.power(self.dim,2)+6))* C + (2/(numpy.power(self.dim,2)+6))*(numpy.dot(self.evolpath,self.evolpath.T)+(2/(self.dim+2)*(2-2/(self.dim+2)))*C)
        current.A = numpy.linalg.cholesky(C)
        current.A = current.A / numpy.linalg.det(current.A)
        return current