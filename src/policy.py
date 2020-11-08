import random
import numpy as np

class Policy():

    def __init__(self, f, mdp=None):
        self.fn = f # Function state -> action
    def __call__(self, state):
        return self.fn(state)

# Return a random taxi driver
policy_random = Policy(lambda s: random.randint(0, s.K - 1))

# Return the earliest available taxi driver
#  If tied, choose at random
def random_lowest(s):
    min_wait = min(s.d)
    indices = [i for i in range(s.K) if s.d[i] == min_wait]
    return random.choice(indices)
policy_earliest_free = Policy(random_lowest)
# Earlier version: returns smallest index
#policy_earliest_free = Policy(lambda s: s.d.index(min(s.d)))

# Greedy policy: return action with lowest cost
def policy_greedy(mdp):

    def C(state, action):
        return state.d[action] + mdp.G.d[state.q_start, state.p[action]]
    def fn(s):
        mincost = 9999999
        besta = None
        for a in range(s.K):
            if C(s, a) < mincost:
                mincost = C(s, a)
                besta = a
        return besta
    return Policy(fn)