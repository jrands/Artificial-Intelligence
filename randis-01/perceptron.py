import numpy as np
import random
import os, subprocess
import matplotlib.pyplot as plt
 
class Perceptron:
    def __init__(self, N):
        # Random linearly separated data
        xA,yA,xB,yB = [random.uniform(-1, 1) for i in range(4)]
        self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        self.X = self.generate_points(N)
 
    def generate_points(self, N):
        X = []
        for i in range(N):
            x1,x2 = [random.uniform(-1, 1) for i in range(2)]
            x = np.array([1,x1,x2])
            s = int(np.sign(self.V.T.dot(x)))
            X.append((x, s))
            X, y = make_semi_circles()
        return X
        
    def make_semi_circles(n_samples=2000, thk=5, rad=10, sep=5, plot=True):
        """Make two semicircles circles
        A simple toy dataset to visualize classification algorithms.
        Parameters
        ----------
        n_samples : int, optional (default=2000)
            The total number of points generated.
        thk : int, optional (default=5)
            Thickness of the semi circles.
        rad : int, optional (default=10)
            Radious of the circle.
        sep : int, optional (default=5)
            Separation between circles.
        plot : boolean, optional (default=True)
            Whether to plot the data.
        Returns
        -------
        X : array of shape [n_samples, 2]
            The generated samples.
        y : array of shape [n_samples]
            The integer labels (-1 or 1) for class membership of each sample.
        """

        
        noisey = np.random.uniform(low=-thk/100.0, high=thk/100.0, size=(n_samples // 2))
        
        noisex = np.random.uniform(low=-rad/100.0, high=rad/100.0, size=(n_samples // 2))
        
        separation = np.ones(n_samples // 2)*((-sep*0.1)-0.6)
        
        
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
        
        # generator = check_random_state(random_state)
        
        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out)) + noisex
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out)) + noisey
        inner_circ_x = (1 - np.cos(np.linspace(0, np.pi, n_samples_in))) + noisex
        inner_circ_y = (1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5) + noisey + separation
        
        X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                       np.append(outer_circ_y, inner_circ_y))).T
        y = np.hstack([np.ones(n_samples_in, dtype=np.intp)*-1,
                       np.ones(n_samples_out, dtype=np.intp)])
        
        if plot:
            plt.plot(outer_circ_x, outer_circ_y, 'r.')
            plt.plot(inner_circ_x, inner_circ_y, 'b.')
            plt.show()

          
        
        return X, y
 
    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5,5))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        V = self.V
        a, b = -V[1]/V[2], -V[0]/V[2]
        l = np.linspace(-1,1)
        plt.plot(l, a*l+b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s]+'o')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'.')
        if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)),str(len(mispts))))
            plt.savefig('p_N%s' % (str(len(self.X))), \
                        dpi=200, bbox_inches='tight')
 
    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        return error
 
    def choose_miscl_point(self, vec):
        # Choose a random point among the misclassified
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]
 
    def pla(self, save=False):
        # Initialize the weigths to zeros
        w = np.zeros(3)
        X, N = self.X, len(self.X)
        it = 0
        # Iterate until all points are correctly classified
        while self.classification_error(w) != 0:
            it += 1
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s*x
            if save:
                self.plot(vec=w)
                plt.title('N = %s, Iteration %s\n' \
                          % (str(N),str(it)))
                plt.savefig('p_N%s_it%s' % (str(N),str(it)), \
                            dpi=200, bbox_inches='tight')
        self.w = w
        print(it)
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)
        
def main():
        p = Perceptron(1000)
        p.pla(save=False)
        p.plot
        plt.clf()
        plt.cla()

main()
