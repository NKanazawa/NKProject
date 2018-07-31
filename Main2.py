import csv
import numpy
import scipy
import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

fig = plt.figure()


def update(i,animdata,sucdata,sigdata,fpcdata,eigAdata,idx,xbound,ybound,startframes,endframes):
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    ax1 = fig.add_subplot(221)
    ax1.cla()
    ax1.set_xlim(0,1)
    ax1.set_ylim(0, 1)
    ax1.quiver(0,0,numpy.abs(fpcdata[i,0]),numpy.abs(fpcdata[i,idx]),angles="xy",scale_units="xy",scale=1)
    ax1.set_title("Principal Vector on Gen "+str(i+startframes+1))
    ax3 = fig.add_subplot(222)
    ax3.cla()
    ax3.plot(sigdata[0:i,0],sigdata[0:i,1],linewidth=2,color="blue")
    ax3.set_title("Sigma on Gen"+str(i+startframes+1))
    ax2 = fig.add_subplot(223)
    ax2.cla()
    ax2.set_xlim(xbound[0], xbound[1])
    ax2.set_ylim(ybound[0], ybound[1])
    ax2.hlines([0],xbound[0],xbound[1],"black",linestyles="dashed")
    blue = []
    if i > 0:
        for j in range(0,i):
            blue.append(animdata[j].tolist())
        blue = numpy.array(blue)
        #ax2.scatter(blue[:,0],blue[:,idx],facecolors='cyan',marker=".",s=10)
    ax2.scatter(animdata[i,-5],animdata[i,-4],facecolors='black', marker="s", s=20)
    ax2.add_artist(Ellipse((animdata[i,-5],animdata[i,-4])),numpy.sqrt(sigdata[i,1]*eigAdata[1]),numpy.sqrt(sigdata[i,1]*eigAdata[0]),
                   numpy.arctan2(fpcdata[i,2],fpcdata[i,3])*180/numpy.pi,color='cyan')
    if animdata[i,4] == 1 and sucdata[i,1] == 1 and sucdata[i,0] == 1:
        ax2.scatter(animdata[i, 0], animdata[i, idx], facecolors='red', marker="*", s=30)
    elif animdata[i, 4] == 1 and sucdata[i, 2] == 1 and sucdata[i,0] == 1:
        ax2.scatter(animdata[i, 0], animdata[i, idx], facecolors='blue', marker="*", s=30)
    elif animdata[i,4] == 1 and sucdata[i,1] == 1 and sucdata[i,0] == 0 and sucdata[i,7] == 1:
        ax2.scatter(animdata[i, 0], animdata[i, idx], facecolors='red', marker=".", s=30)
    elif animdata[i, 4] == 1 and sucdata[i, 2] == 1 and sucdata[i,0] == 0 and sucdata[i,7] == 1:
        ax2.scatter(animdata[i, 0], animdata[i, idx], facecolors='blue', marker=".", s=30)
    elif animdata[i,4] == 1 and sucdata[i,1] == 1 and sucdata[i,7] == 0:
        ax2.scatter(animdata[i, 0], animdata[i, idx], facecolors='red', marker="^", s=30)
    elif animdata[i, 4] == 1 and sucdata[i, 2] == 1 and sucdata[i,7] == 0:
        ax2.scatter(animdata[i, 0], animdata[i, idx], facecolors='blue', marker="^", s=30)
    elif animdata[i, 4] == 1 and (sucdata[i,3] == 1 or sucdata[i,5]==1):
        ax2.scatter(animdata[i, 0], animdata[i, idx], facecolors='red', marker="s", s=30)
    elif animdata[i, 4] == 1 and (sucdata[i,4] == 1 or sucdata[i,6]==1):
        ax2.scatter(animdata[i, 0], animdata[i, idx], facecolors='blue', marker="s", s=30)
    elif animdata[i, 4] == 0 and sucdata[i,5]==1:ax2.scatter(animdata[i,0],animdata[i,idx],facecolors='red',marker="x",s=30)
    elif animdata[i, 4] == 0 and sucdata[i,6]==1: ax2.scatter(animdata[i, 0], animdata[i, idx], facecolors='blue', marker="x", s=30)
    ax2.set_title("generation "+str(i+startframes+1))
    ax4 = fig.add_subplot(224)
    ax4.cla()
    suclog = []
    allsuclog = []
    for k in range(startframes, endframes - 1):
        if sucdata[k-startframes, 1] == 1 or sucdata[k-startframes, 2] == 1:
            allsuclog.append(animdata[k-startframes].tolist())
    sucxbound = [numpy.amin(allsuclog, axis=0)[2], numpy.amax(allsuclog, axis=0)[2]]
    sucybound = [numpy.amin(allsuclog, axis=0)[3], numpy.amax(allsuclog, axis=0)[3]]
    ax4.set_xlim(sucxbound[0], sucxbound[1])
    ax4.set_ylim(sucybound[0], sucybound[1])
    if i > 0:
        for j in range(0,i):
            if sucdata[j,1] == 1 or sucdata[j,2] == 1:
                suclog.append(animdata[j].tolist())
        suclog = numpy.array(suclog)
        allsuclog = numpy.array(allsuclog)
        #if len(suclog) > 1:
            #ax4.scatter(suclog[:,2],suclog[:,3],facecolors='cyan',marker=".",s=10)
    ax4.scatter(animdata[i, 9], animdata[i, 10], facecolors='cyan', marker=".", s=20)
    if sucdata[i, 1] == 1 or sucdata[i, 2] == 1:
        ax4.scatter(animdata[i,2],animdata[i,3],facecolors='red',marker=".",s=20)

if __name__ == "__main__":
    numframes = 650
    startframes = 600
    numidx = 1
    animfile = open("foranimation.csv", "r")
    animbase = csv.reader(animfile, delimiter=",")
    animlist = [row for row in animbase]
    animlist = [animlist[i] for i in range(startframes,numframes)]
    animdata = numpy.array(animlist, dtype="float64")
    anim_xbound=[numpy.amin(animdata,axis=0)[0],numpy.amax(animdata,axis=0)[0]]
    anim_ybound=[numpy.amin(animdata,axis=0)[numidx],numpy.amax(animdata,axis=0)[numidx]]

    sucfile = open("NESsuccessLog.csv", "r")
    sucbase = csv.reader(sucfile, delimiter=",")
    suclist = [row for row in sucbase]
    suclist = [suclist[i] for i in range(startframes,numframes)]
    sucdata = numpy.array(suclist, dtype="int32")

    fpcfile = open("FirstPC.csv", "r")
    fpcbase = csv.reader(fpcfile, delimiter=",")
    fpclist = [row for row in fpcbase]
    fpclist = [fpclist[i] for i in range(startframes, numframes)]
    fpcdata = numpy.array(fpclist, dtype="float64")

    eigAfile = open("NESA.csv", "r")
    eigAbase = csv.reader(eigAfile, delimiter=",")
    eigAlist = [row for row in eigAbase]
    eigAlist = [eigAlist[i] for i in range(startframes, numframes)]
    eigAdata = numpy.array(eigAlist, dtype="float64")

    sigfile = open("NESSigmaLog.csv", "r")
    sigbase = csv.reader(sigfile, delimiter=",")
    siglist = [row for row in sigbase]
    siglist = [siglist[i] for i in range(startframes, numframes)]
    sigdata = numpy.array(siglist, dtype="float64")

    anime = anim.FuncAnimation(fig, update, fargs=(animdata,sucdata,sigdata,fpcdata,eigAdata,numidx,anim_xbound,anim_ybound,startframes,numframes), interval=2000, frames=(numframes-startframes))
    anime.save('anime.mp4')
    #plt.show()