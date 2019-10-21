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

        # task: calculate true utilities using Bellman's update

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # utility values

        print("start! discount " + str(discount))
        print("iterations " + str(iterations))

        states = mdp.getStates()

        for iter in range(iterations): # wasn't working before - but might work now?

            # temporary dictionary to hold updated state values
            # update self.values when you complete the entire iteration
            temp_values = util.Counter()

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

                # updated temp utility = reward of state + discount*q_value
                temp_values[s] = mdp.getReward(s, None, None) + discount*q_value_of_best_action

            # ----------------------------------------s--------------------------------------------

            # update actual utility
            self.values = temp_values.copy()

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
            # note: make sure you are using the 'old' util values

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        # task: find action that gives the maximum expected utility

        actions = self.mdp.getPossibleActions(state)
        expected_utility = util.Counter()

        # loop to fill dictionary of expected utility corresponding to all possible actions
        for a in actions:

            expected_utility[a] = self.computeQValueFromValues(state, a)

        # best action by our policy
        best_action = expected_utility.argMax()

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
