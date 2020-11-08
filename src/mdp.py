import random
import numpy as np
from functools import lru_cache

from .graph import *
from .nyc import *


class State(tuple):
    
    def __init__(self, tp):
        if len(tp) != 3:
            raise ValueError("Invalid number of arguments passed to State(_)")
        
        self.p = tp[0]
        self.POSITIONS = tp[0]
        self.d = tp[1]
        self.DELAYS = tp[1]
        self.q = tp[2]
        self.q_start = tp[2][0]
        self.q_end = tp[2][1]
        self.QUERY = tp[2]

        self.K = len(self.d)

        return super().__init__()

    def __repr__(self):
        return "State(\nPOSITIONS: "+str(self.p)+"\nDELAYS:    "+\
                str(self.d)+"\nQUERY: "+str(self.q)+"\n)"


class MarkowDP():

    def __init__(self, G, K=None, queries=None):
        '''Builds from a Graph G, number of taxis K (optional),
        and, optionally, a list of preset queries.

        Default values: K = sqrt(N) * G.avgd
        '''

        if K is None:
            K = int(np.sqrt(G.N) + G.avgd + 1)

        self.G = G
        self.N = G.N
        self.M = G.M
        self.K = K
        self.queries = queries
        self.current_query = 0

    def get_queries(self, n=1):
        '''Returns n queries'''

        if self.queries is None:
            # Default, return n random pairs of numbers
            return np.random.randint(0, self.N, size=(n, 2))

        # Return as many queries as you can from current list,
        #  then if necessary recursively call yourself
        n_valid = min(n, len(self.queries) - self.current_query)
        q_valid = self.queries[self.current_query:self.current_query+n_valid]

        n -= n_valid        
        self.current_query += n_valid
        if self.current_query >= len(self.queries):
            self.current_query = 0

        return q_valid if n == 0 else np.vstack((q_valid, self.get_queries(n)))

    def get_query(self):
        return tuple(self.get_queries()[0])

    def get_starting_state(self):
        return State((tuple(self.G.rand_v() for _ in range(self.K)),
                      tuple(0 for _ in range(self.K)),
                      self.get_query()))

    def C(self, state, action):
        '''Returns the wait time, plus distance to start point'''
        return state.d[action] + self.G.d[state.q_start, state.p[action]]

    def R(self, state, action):
        '''Returns the reward (negative cost)'''
        return -self.C(state, action)

    def T_p(self, state, action):
        '''Returns the new taxi positions after action'''
        new_p = [pos for pos in state.p]
        new_p[action] = state.q_end
        return tuple(new_p)

    def T_d(self, state, action):
        '''Returns the new taxi delays'''
        new_d = [max(dl - 1, 0) for dl in state.d]
        new_d[action] += self.G.d[state.p[action], state.q_start] # Wait time
        new_d[action] += self.G.d[state.q_start, state.q_end] # Trip time
        return tuple(new_d)

    def T(self, s, a):
        return State((self.T_p(s, a), self.T_d(s, a), self.get_query()))

    def eval(self, policy, no_iter=1000, gamma=1, 
             verbose=False, verbose_result=False, return_pct_trip=False):
        '''Evaluate a given policy by sampling, returns total (discounted) reward'''
        s = self.get_starting_state()
        if verbose: print("Starting state:", s, "\n-----------------")

        reward = 0
        for i in range(no_iter):
            a = policy(s)
            rw = self.R(s, a)
            s = self.T(s, a)

            if verbose:
                print("Step", i, "; Action:", a, "; Reward", rw, 
                      "New state:", s, "\n-----------------")
            reward = gamma * reward + rw
        if verbose or verbose_result:
            print("Total reward over", no_iter, "steps: ", reward)
            print("Approx", round(-100 * reward / no_iter / self.G.avgd, 1), " pct of avg trip time")

        return reward if not return_pct_trip else \
               (-100 * reward / no_iter / self.G.avgd)


MDP = {w: MarkowDP(G[w]) for w in ['small', 'medium', 'large', 'line', 'complete']}
MDP['nyc'] = MarkowDP(G['nyc'], queries=nyc_queries)
