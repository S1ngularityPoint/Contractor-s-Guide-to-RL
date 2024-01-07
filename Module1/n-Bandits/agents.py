from bandits import Bandit
import numpy as np
# Import libraries if you need them

class Agent:
    def __init__(self, bandit: Bandit) -> None:
        self.bandit = bandit
        self.banditN = bandit.getN()

        self.rewards = 0
        self.numiters = 0
    

    def action(self) -> int:
        '''This function returns which action is to be taken. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    def update(self, choice : int, reward : int) -> None:
        '''This function updates all member variables you may require. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    # dont edit this function
    def act(self) -> int:
        choice = self.action()
        reward = self.bandit.choose(choice)

        self.rewards += reward
        self.numiters += 1

        self.update(choice,reward)
        return reward

class GreedyAgent(Agent):
    def __init__(self, bandits: Bandit, initialQ : float) -> None:
        super().__init__(bandits)
        self.Q = np.full(self.banditN, initialQ)
        self.N = np.zeros(self.banditN)
        # add any member variables you may require
        
    # implement
    def action(self) -> int:
        return np.argmax(self.Q)

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.N[choice]+=1
        self.Q[choice]+=(reward-self.Q[choice])/self.N[choice]

class epsGreedyAgent(Agent):
    def __init__(self, bandits: Bandit, epsilon : float) -> None:
        super().__init__(bandits)
        self.epsilon = epsilon
        self.Q = np.zeros(bandits.getN())
        self.N = np.zeros(bandits.getN())
        # add any member variables you may require
    
    # implement
    def action(self) -> int:
        if(np.random.random()>self.epsilon):
            return np.argmax(self.Q)
        else:
            return np.random.randint(len(self.Q))

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.N[choice]+=1
        self.Q[choice]+=(reward-self.Q[choice])/self.N[choice]

class UCBAAgent(Agent):
    def __init__(self, bandits: Bandit, c: float) -> None:
        super().__init__(bandits)
        self.c = c
        self.Q = np.zeros(bandits.getN())
        self.N = np.zeros(bandits.getN())
        # add any member variables you may require

    # implement
    def action(self) -> int:
        if(self.numiters<len(self.Q)):
            return self.numiters
        else:
            U = np.sqrt(self.c * np.log(self.numiters)/self.N)
            return np.argmax(U+self.Q)

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.N[choice]+=1
        self.Q[choice]+=(reward-self.Q[choice])/self.N[choice]

class GradientBanditAgent(Agent):
    def __init__(self, bandits: Bandit, alpha : float) -> None:
        super().__init__(bandits)
        self.alpha = alpha
        self.Q = np.zeros(bandits.getN())
        self.N = np.zeros(bandits.getN())
        # add any member variables you may require

    # implement
    def action(self) -> int:
        scaled_Q=self.Q/self.alpha
        norm_Q=scaled_Q-max(scaled_Q)
        exp_Q=np.exp(norm_Q)
        probs=exp_Q/np.sum(exp_Q)
        return np.random.choice(np.arange(len(probs)),size=1,p=probs)[0]

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.N[choice]+=1
        self.Q[choice]+=(reward-self.Q[choice])/self.N[choice]

class ThompsonSamplerAgent(Agent):
    def __init__(self, bandits: Bandit) -> None:
        super().__init__(bandits)
        self.Q = np.zeros(bandits.getN())
        self.N = np.zeros(bandits.getN())
        # add any member variables you may require

    # implement
    def action(self) -> int:
        samples = np.random.normal(loc=self.Q, scale=1/(1+np.sqrt(self.N)))
        return np.argmax(samples)

    # implement
    def update(self, choice: int, reward: int) -> None:
        if(self.numiters<len(self.Q)):
            return self.numiters
        else:
            self.N[choice]+=1
            self.Q[choice]+=(reward-self.Q[choice])/self.N[choice]

# Implement other subclasses if you want to try other strategies