"""Supply code to
...
by B. E. Abali and A. F. Queiruga
"""
__author__ = "B. Emek Abali and Alejandro F. Queiruga"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

from fenics import *
import numpy as np


#
# Tensor operators we want in global space
#
i, j, k, l, m = indices(5)
delta = Identity(3)
#levicivita2 = as_matrix([ (0,1,-1) , (-1,0,1) , (1,-1,0) ])
levicivita3 = as_tensor([ ( (0,0,0),(0,0,1),(0,-1,0) ) , ( (0,0,-1),(0,0,0),(1,0,0) ) , ( (0,1,0),(-1,0,0),(0,0,0) ) ])
epsilon = levicivita3

