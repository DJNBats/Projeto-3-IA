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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Implementamos a versão em lote (síncrona) da iteração de valores.
        # Cada iteração calcula novosValores com base apenas em self.values
        # (o vetor V^k) e depois substitui self.values por novosValores.
        for i in range(self.iterations):
            # novosValores guardará V^{k+1} calculado a partir de V^k (self.values)
            novosValores = util.Counter()
            # iterar por cada estado do MDP
            for estado in self.mdp.getStates():
                # estados terminais têm valor 0 (sem recompensas futuras)
                if self.mdp.isTerminal(estado):
                    novosValores[estado] = 0.0
                    continue
                # recolher Q(s,a) para todas as ações possíveis e tomar o máximo
                valoresAcoes = []
                for acao in self.mdp.getPossibleActions(estado):
                    # computeQValueFromValues usa self.values atuais (V^k)
                    q_valor = self.computeQValueFromValues(estado, acao)
                    valoresAcoes.append(q_valor)
                # se há ações, V^{k+1}(s) = max_a Q(s,a), senão 0.0
                novosValores[estado] = max(valoresAcoes) if valoresAcoes else 0.0
            # atualização síncrona: commit de V^{k+1}
            self.values = novosValores


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
        q_valor = 0.0
        # iterar por todos os estados sucessores possíveis e suas probabilidades
        for proximoEstado, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # recompensa imediata para a transição (state,action) -> proximoEstado
            recompensa = self.mdp.getReward(state, action, proximoEstado)
            # adiciona a contribuição esperada: prob * (recompensa + gamma * V(próximo))
            q_valor += prob * (recompensa + (self.discount * self.values[proximoEstado]))
        return q_valor
        ###util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        acoesPossiveis = self.mdp.getPossibleActions(state)
        # se não há ações legais, estamos num estado terminal
        if not acoesPossiveis:
            return None
        melhorAcao = None
        melhorValor = float('-inf')
        # avalia Q(s,a) para cada ação e mantém o argmax
        for acao in acoesPossiveis:
            q = self.computeQValueFromValues(state, acao)
            if q > melhorValor:
                melhorValor = q
                melhorAcao = acao
        return melhorAcao
        ###util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Passo 1: calcular predecessores para cada estado
        predecessores = {s: set() for s in self.mdp.getStates()}
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for proximoEstado, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                    # se há probabilidade não-nula de ir para proximoEstado a partir de s,
                    # então s é predecessor de proximoEstado
                    if prob > 0:
                        predecessores[proximoEstado].add(s)

        # Passo 2: inicializar uma fila de prioridade vazia
        filaPrioridade = util.PriorityQueue()

        # Passo 3: para cada estado não-terminal, calcular prioridade inicial e inserir na fila
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            valorAtual = self.values[s]
            melhorQ = max(self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s))
            diff = abs(valorAtual - melhorQ)
            # empurrar com prioridade -diff porque util.PriorityQueue é min-heap
            filaPrioridade.update(s, -diff)

        # Passo 4: loop principal — realizar até self.iterations atualizações
        for i in range(self.iterations):
            if filaPrioridade.isEmpty():
                break
            estado = filaPrioridade.pop()
            # atualiza o valor do estado se não for terminal
            if not self.mdp.isTerminal(estado):
                self.values[estado] = max(self.computeQValueFromValues(estado, a)
                                          for a in self.mdp.getPossibleActions(estado))
            # para cada predecessor, verificar se deve ser inserido/atualizado na fila
            for p in predecessores[estado]:
                if self.mdp.isTerminal(p):
                    continue
                valorAtual = self.values[p]
                melhorQ = max(self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p))
                diff = abs(valorAtual - melhorQ)
                # se a mudança for significativa, adicionar/atualizar na fila com prioridade -diff
                if diff > self.theta:
                    filaPrioridade.update(p, -diff)