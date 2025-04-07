import matplotlib as mpl
import matplotlib.pyplot as plt

def initialize():

    #initiliaze settings

    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",  # or "lualatex"
        "font.family": "serif",      # use serif/main font for text elements
        "text.usetex": True,         # use inline math for ticks
        "pgf.rcfonts": False, 
        "axes.labelsize" : 11,
        "legend.fontsize": 9,
        "lines.linewidth" : 0.7     # don't setup fonts from rc parameters
    })
