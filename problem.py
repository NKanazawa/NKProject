import numpy

class zdt1():
    def __init__(self,LOW=None,UP=None):
        self.LOWBOUNDS = LOW
        self.UPBOUNDS = UP

    def zdt1(self,ind):
        gnm = []
        conresult = []
        for i, num in enumerate(ind):
            if num < self.LOWBOUNDS[i]:
                gnm.append(self.LOWBOUNDS[i])
                conresult.append(num - self.LOWBOUNDS[i])
            elif num > self.UPBOUNDS[i]:
                gnm.append(self.UPBOUNDS[i])
                conresult.append(self.UPBOUNDS[i] - num)
            else:
                gnm.append(num)
                conresult.append(numpy.minimum(num - self.LOWBOUNDS[i], self.UPBOUNDS[i] - num))
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