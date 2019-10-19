# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
q1_debug = True

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # utility values

        print("start! discount " + str(discount))
        print("iterations " + str(iterations))
        # input()

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # get
        # states:
        # ['TERMINAL_STATE', (0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        # get
        # actions:
        # ('north', 'west', 'south', 'east')
        # get
        # transition
        # states and probabilities:
        # [((0, 1), 0.8), ((1, 0), 0.1), ((0, 0), 0.1)]
        # get
        # reward:
        # 0.0
        # is terminal:
        # False

        # Calculates the utility of each state after a certain number of iterations

        states = mdp.getStates()

        for iter in range(iterations): # wasn't working before - but might work now?

            for s in states:

                # doing the sigma part of the equation
                actions_in_s = mdp.getPossibleActions(s) # e.g. ('north', 'west', 'south', 'east')
                actions_q_values = util.Counter()

                # find action with max p(x'x,a) U[s]
                for a in actions_in_s:

                    # add q_value to dictionary
                    actions_q_values[a] = self.computeQValueFromValues(s, a)
                #----------------------------------------a-------------------------------------

                # next action that gives the highest q value
                best_action = actions_q_values.argMax()

                # the q value of the next best action
                q_value_of_best_action = actions_q_values[best_action]

                # updated utility = reward of state + discount*q_value
                self.values[s] = mdp.getReward(s, None, None) + discount*q_value_of_best_action

            # ----------------------------------------s--------------------------------------------
        # ----------------------------------------iter--------------------------------------

        # now self.values has the utility of each state after speficied number of iterations

        # check:
        # for key in self.values.keys():
        #     print(str(key) + "    " + str(self.values[key]))

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        next_possible_states_and_prob = self.mdp.getTransitionStatesAndProbs(state, action)

        # q_value
        q_value = 0

        # loop to calculate the q_value
        for poss in next_possible_states_and_prob:

            # possibility to this state
            poss_to_state = poss[1]

            # to state
            to_state = poss[0] # tuple

            q_value += poss_to_state*self.values[to_state] # P(s'|s,a) * U[s']

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        # task: find action that gives the maximum q value

        actions = self.mdp.getPossibleActions(state)
        q_values = util.Counter()

        # loop to fill dictionary of q_values corresponding to all possible actions
        for a in actions:

            q_values[a] = self.computeQValueFromValues(state, a)

        # best action by our policy
        best_action = q_values.argMax()

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
