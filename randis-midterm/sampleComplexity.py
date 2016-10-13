import numpy as np
import math

class complexity:
    def findN(delta, N, dvc):
        out = 8/(delta*delta)
        ins = ((math.pow((2*N), dvc) + 1) * 4)/delta
        log = math.log(ins)
        newN = out * log        
        print("N >= ", newN)
        
    
def main():
    c = complexity
    c.findN(.1, 1000, 3)
    
main()