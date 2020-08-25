import plot_corner as pc
from glob import glob

fls = ['20181201T060953_M85_fullfit.dat', '20181201T044847_M87_fullfit.dat', '20181129T014719_M85_fullfit.dat']
fls = glob('/home/elliot/mcmcgemini/mcmcresults/20181207*')
for fl in fls:
    #pc.plot_corner('/home/elliot/mcmcgemini/mcmcresults/'+fl, -1000)
    pc.plot_corner(fl, -1000)
