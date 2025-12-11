# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # qValores guarda Q(s,a); util.Counter devolve 0 para chaves não vistas
        self.qValores = util.Counter() # default 0.0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]
        ###util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        acoesLegais = self.getLegalActions(state)  # usa a função de ações definida em ReinforcementAgent
        if not acoesLegais:
            # estado terminal -> sem recompensa futura
            return 0.0
        # retorna o máximo Q entre ações legais
        return max(self.getQValue(state, a) for a in acoesLegais)
        ###util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        acoesLegais = self.getLegalActions(state)
        if not acoesLegais:
            return None
        melhorValor = float('-inf')
        melhoresAcoes = []
        # encontra o maior Q-valor e recolhe todas as ações que o atingem
        for a in acoesLegais:
            q = self.getQValue(state, a)
            if q > melhorValor:
                melhorValor = q
                melhoresAcoes = [a]
            elif q == melhorValor:
                melhoresAcoes.append(a)
        # desempate: escolha uniforme entre as melhores ações
        return random.choice(melhoresAcoes)
        ###util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        acoesLegais = self.getLegalActions(state)
        if not acoesLegais:
            return None
        # exploração: flipCoin devolve True com probabilidade epsilon
        if util.flipCoin(self.epsilon):
            return random.choice(acoesLegais)
        # exploração do conhecimento: escolhe a melhor ação segundo Q-vals
        return self.computeActionFromQValues(state)
        ###util.raiseNotDefined()

        ###return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # calcula max_a' Q(nextState, a') usando self.getLegalActions(nextState)
        acoesLegaisNext = self.getLegalActions(nextState)
        valorSeguinte = 0.0
        if acoesLegaisNext:
            valorSeguinte = max(self.getQValue(nextState, a) for a in acoesLegaisNext)
        # alvo de amostra
        amostra = reward + self.discount * valorSeguinte
        # atualização (média exponencial)
        qAntigo = self.getQValue(state, action)
        self.qValores[(state, action)] = (1 - self.alpha) * qAntigo + self.alpha * amostra
        ###util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        q = 0.0
        # produto escalar entre pesos e features
        for f, val in features.items():
            q += self.pesos[f] * val
        return q
        ###util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        acoesLegaisNext = self.getLegalActions(nextState)
        valorSeguinte = 0.0
        if acoesLegaisNext:
            valorSeguinte = max(self.getQValue(nextState, a) for a in acoesLegaisNext)
        # erro TD (difference)
        correcao = (reward + self.discount * valorSeguinte) - self.getQValue(state, action)
        # atualiza cada peso proporcional ao valor da feature
        for f, val in features.items():
            self.pesos[f] += self.alpha * correcao * val

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        # chamado no fim de cada jogo; executa final da classe pai
        PacmanQAgent.final(self, state)
        # ao terminar o treino poderíamos imprimir pesos para debug

        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
