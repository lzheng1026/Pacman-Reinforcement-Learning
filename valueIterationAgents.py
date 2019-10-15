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
        self.values = util.Counter() # A Counter is a dict with default 0

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

        for iter in range(10):

            for s in states:

                # doing the sigma part of the equation
                actions_in_s = mdp.getPossibleActions(s) # e.g. ('north', 'west', 'south', 'east')
                actions_q_values = util.Counter()

                for a in actions_in_s:

                    # transition states and associated probabilities
                    next_possible_states_and_prob = mdp.getTransitionStatesAndProbs(s, a)

                    for poss in next_possible_states_and_prob:

                        # possibility to this state
                        poss_to_state = poss[1]

                        # to state
                        to_state = poss[0] # tuple

                        if mdp.isTerminal(to_state):
                            print(str(mdp.getPossibleActions(s)))
                            print("current state: " + str(s) + "; action: " + str(a) + "; to_state: " + str(to_state))
                            print("reward: " + str(mdp.getReward(s, a, to_state)))

                        # q value to this state
                        q_val_to_state = self.values[to_state] # should be 0 to start with

                        actions_q_values[a] += poss_to_state*q_val_to_state
                #----------------------------------------a-------------------------------------

                # next action that gives the highest q value
                best_action = actions_q_values.argMax()
                # the q value of the next best action
                max_value_of_best_action = actions_q_values[best_action]

                # final value of right-side
                rs = self.discount*max_value_of_best_action

                # left-side
                # find next highest likely state

                if best_action is not None:
                    next_states = mdp.getTransitionStatesAndProbs(s, best_action)
                    nexts = util.Counter()
                    for next in next_states:
                        # next state
                        next_state = next[0]
                        possibility_to_next_state = next[1]
                        nexts[next_state] = possibility_to_next_state
                    highest_likely_state = nexts.argMax()
                    ls = mdp.getReward(s, best_action, highest_likely_state)
                else: # also am i using the policy from the previous step?
                    ls = 0#mdp.getReward(s, 'exit', "TERMINAL_STATE")

                self.values[s] = ls + rs
            # ----------------------------------------s--------------------------------------------
        # ----------------------------------------iter--------------------------------------

        # now self.values has the utility of each state after speficied number of iterations

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
        "*** YOUR CODE HERE ***"
        # Q(s, a) = expected discounted reward if perform a from s
        # and then follow optimal policy from then on.

    def Q_val(self, state, action, value=0):

        if self.mdp.isTerminal(state):
            return 1+value

        # find reward
        next_states_and_possibilities = self.mdp.getTransitionStatesAndProbs(state, action)
        probability = util.Counter()
        for next in next_states_and_possibilities:
            next_state = next[0]
            next_state_prob = next[1]
            probability[next_state] = next_state_prob
        most_likely_next_state = probability.argMax()
        reward = self.mdp.getReward(state, action, most_likely_next_state)

        # find max Q
        next_best_action = self.computeActionFromValues(most_likely_next_state)
        discounted_max_Q = self.Q_val(most_likely_next_state, next_best_action, value)

        return reward + discounted_max_Q


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        expected_utilities = util.Counter()
        actions = self.mdp.getPossibleActions(state)

        for a in actions:

            transition_states_and_probs = self.mdp.getTransitionStatesAndProbs(state, a)

            for transition_state_and_prob in transition_states_and_probs:

                transition_state = transition_state_and_prob[0]
                probability = transition_state_and_prob[1]

                # expected utilities
                expected_utilities[a] += probability*self.values[transition_state]
            # --------------transition_state_and_prob---------------------

        # -------------------------------a--------------------------------

        best_action = expected_utilities.argMax()

        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
