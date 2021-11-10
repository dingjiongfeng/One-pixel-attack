# from scipy.optimize import rosen, differential_evolution
# from scipy.optimize import OptimizeResult, minimize
# from scipy.optimize.optimize import _status_message
# from scipy._lib._util import check_random_state
# import numpy as np
# import warnings

# # bounds = [(0, 36), (0, 36)]


# # def func(x):
# #     return (x[0]+1)**2 - np.log(x[1])


# # result = differential_evolution(func, bounds)
# # print(result.x, result.fun)  # x是参数 fun是值


# # predict_fn, bounds, maxiter, popsize, recombination=1, atol=-1, callback, polish=False, init
# def differential_evolution(func, bounds, args=(), strategy='best1bin',
#                            maxiter=1000, popsize=15, tol=0.01,
#                            mutation=(0.5, 1), recombination=0.7, seed=None,
#                            callback=None, disp=False, polish=True,
#                            init='latinhypercube', atol=0):
#     '''
#     Finds the global minimum of a multivariate function.
#     func: The objective function to be minimized
#     bounds: bounds for variables
#     args: tuple,optional

#     returns
#     res : OptimizeResult
#     important attributes are
#         'x' the solution array,
#         'success' a boolean flag indicating if the optimizer exited successfully
#         'message' the cause of the termination
#     '''

#     '''
#     Differential evolution is a stochastic population based method that is
#     useful for global optimization problems. At each pass through the population
#     the algorithm mutates each candidate solution by mixing with other candidate
#     solutions to create a trial candidate.
#     '''
#     solver = DifferentialEvolutionSolver(func, bounds, args=args,
#                                          strategy=strategy, maxiter=maxiter,
#                                          popsize=popsize, tol=tol,
#                                          mutation=mutation,
#                                          recombination=recombination,
#                                          seed=seed, polish=polish,
#                                          callback=callback,
#                                          disp=disp, init=init, atol=atol)
#     return solver.solve()


# class DifferentialEvolutionSolver():

#     def __init__(self, func, bounds, args=(),
#                  strategy='best1bin', maxiter=1000, popsize=15,
#                  tol=0.01, mutation=(0.5, 1), recombination=0.7,
#                  seed=None, maxfun=np.inf, callback=None, disp=False,
#                  polish=True, init='latinhypercube', atol=0):

#         self.mutation_func = '_best1'
#         self.strategy = strategy
#         self.callback = callback
#         self.polish = polish
#         self.tol, self.atol = tol, atol
#         self.scale = mutation

#         self.dither = None
#         if hasattr(mutation, '__iter__') and len(mutation) > 1:  # ??
#             self.dither = [mutation[0], mutation[1]]
#             self.dither.sort()

#         self.cross_over_probability = recombination

#         self.func = func
#         self.args = args
#         # bounds->np.array  size[5,2] 转置 ->size[2,5]
#         self.limits = np.array(bounds, dtype='float').T
#         self.maxiter = maxiter
#         self.maxfun = maxfun
#         self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
#         self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

#         self.parameter_count = np.size(self.limits, axis=1)  # 列数
#         # Turn seed into a np.random.RandomState instance.
#         self.random_number_generator = check_random_state(seed)
#         self.num_population_members = max(5, popsize * self.parameter_count)
#         self.population_shape = (self.num_population_members,
#                                  self.parameter_count)

#         self._nfev = 0
#         if isinstance(init, str):
#             if init == 'latinhypercube':
#                 self.init_population_lhs()
#             elif init == 'random':
#                 self.init_population_random()
#         else:
#             self.init_population_array(init)

#         self.disp = disp

#     def init_population_lhs(self):
