import sys,os
dir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)
from NalSuper import NalSuper
from PerceptualLoss import LossNetwork as PerLoss
