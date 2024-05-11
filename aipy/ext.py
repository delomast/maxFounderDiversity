import numpy as np

# scipy here
from scipy import optimize

import cvxopt as opt
# import mosek
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas

def extsolv(args, PLOT_PATH):    

    disp = args.debug # and not args.noplts  
  
    n = args.n
    x0 = np.ones((n,1)).flatten()/n
    Anp = args.A.detach().clone().numpy()

    #A function to define the space where scipy.minimize should 
    #confine its search:
    def apply_sum_constraint(inputs):
        #return value must come back as 0 to be accepted
        #if return value is anything other than 0 it's rejected
        #as not a valid answer.
        total = 1.0 - np.sum(inputs)
        return total
      
    def fcst(x):
        return (0.5*np.dot(x.T, np.dot(Anp, x))) # - np.dot(bnp.T, x) + 1)
      
    def jac(x):
        return (0.5*np.dot(Anp, x)) # - np.dot(bnp.T, x) + 1)

    def hess(x):
        return Anp

    answr = dict()
    allx = dict()
    allf = dict()
                
    def tcsqp():
      all_x_i = [x0]
      all_f_i = [fcst(x0)]          
      def store(x, flag=None):
        all_x_i.append(x)
        all_f_i.append(fcst(x))
        
      my_constraints = ({'type': 'eq', "fun": apply_sum_constraint })
      answr = optimize.minimize(fcst, x0, 
                            method='trust-constr', jac=jac, hess=hess, 
                            options={'disp': True, "gtol": 1e-5},
                            bounds=optimize.Bounds(0,1,True),
                            constraints=my_constraints, callback=store)         
      
      return answr, all_x_i, all_f_i  
                 
    def slsqp():
      all_x_i = [x0]
      all_f_i = [fcst(x0)]          
      def store(x, flag=None):
        all_x_i.append(x)
        all_f_i.append(fcst(x))
        
      my_constraints = ({'type': 'eq', "fun": apply_sum_constraint })
      answr = optimize.minimize(fcst, x0, 
                            method='SLSQP', jac=jac, 
                            options={'disp': disp},
                            bounds=optimize.Bounds(0,1,True),
                            constraints=my_constraints, callback=store )      
      
      return answr, all_x_i, all_f_i
                 
      
    print('\nSLSQP\n')
    answr['tcsqp'], allx['tcsqp'], allf['tcsqp'] = tcsqp()
    answr['slsqp'], allx['slsqp'], allf['slsqp'] = slsqp()
    
    print('\nCVX\n')
    # CVX
    Q = matrix(Anp.astype(np.double))
    r = matrix(np.zeros(n))
    A = matrix(np.ones(n)).T
    b = matrix(1.0)
    # G = matrix(-np.eye(n))
    # h = matrix(np.zeros(n))
    
   
    G = matrix(0.0, (2*n, n))
    h = matrix(0.0, (2*n,1)) 
    # G = matrix(-np.eye(n), np.eye(n))
    # h = matrix(np.zeros(n), np.ones(n))
    
    for k in range(n):
      G[k,k] = -1.0
      G[k+n,k] = 1.0
      h[k+n] = 1.0
    
    opts = opt.solvers.options
    options['show_progress'] = disp
    sol = qp(Q, -r, G, h, A, b, options=opts)
    maxit = sol['iterations']
    cvxf = []
    for it in range(1, maxit+1):
       opts = opt.solvers.options
       opts['maxiters'] = it 
       options['show_progress'] = False
       sol_it = qp(Q, -r, G, h, A, b, options=opts)
       cvxf.append(sol_it['primal objective'])

    return sol, answr, allx, allf, cvxf