import sys
sys.path.insert(0, '/Users/carol/python/ThinkBayes2/thinkbayes2/')

import numpy as np
import matplotlib.pyplot as plt

from thinkbayes2 import Pmf, Suite, CredibleInterval, Beta

# PMF for 6-sided die
pmf = Pmf()
for x in [1, 2, 3, 4, 5, 6]:
    pmf.Set(x, 1/6)
print(pmf)

# How to build up a pmf from a list of strings
pmf2 = Pmf()
for word in ['a', 'in', 'or', 'to', 'a', 'me', 'in']:
    pmf2.Incr(word, 1)
pmf2.Normalize()
print(pmf2)
print("Probability of letter a:", pmf2.Prob('a'))   # Typo p12 print pmf.Prob('the') should read print(pmf.Prob('the'))

# PMF for the Cookie problem
pmf = Pmf()
# Prior:
pmf.Set("Bowl 1", 0.5)
pmf.Set("Bowl 2", 0.5)
# Posterior:
# First multiply prior by likelihood
pmf.Mult("Bowl 1", 0.75)
pmf.Mult("Bowl 2", 0.5)
# Then normalise (we can do this because the hypotheses are mutually exclusive and collectively exhaustive,
# i.e. only one of the hypotheses can be true and there can be no other hypothesis)
pmf.Normalize()
print(pmf)


# Create a Cookie class that inherits from Pmf and represents the Cookie problem
class Cookie(Pmf):

    proportions = {
        'Bowl 1':dict(vanilla=0.75, chocolate=0.25),
        'Bowl 2':dict(vanilla=0.5, chocolate=0.5),
        }

    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()
    
    def Update(self, data):
        for hypo in self.Values():
            likelihood = self.Likelihood(data, hypo)
            self.Mult(hypo, likelihood)
        self.Normalize()

    def Likelihood(self, data, hypo):
        proportion = self.proportions[hypo]
        likelihood = proportion[data]
        return likelihood

# Set up the hypotheses for the Cookie problem
hypos = ['Bowl 1', 'Bowl 2']
# Initialise the prior for the Cookie problem
pmf = Cookie(hypos)
print("Prior:")
for hypo, prob in pmf.Items():
    print(hypo, prob)     # Type p14: print hypo, prob should read print(hypo, prob)
# Update the prior given one data point (we drew a vanilla cookie)
pmf.Update('vanilla')
print("Posterior:")
for hypo, prob in pmf.Items():
    print(hypo, prob)
# Draw some more cookies (with replacement) and update the prior
dataset = ['vanilla', 'chocolate', 'vanilla']
for data in dataset:
    pmf.Update(data)
print("Posterior:")
for hypo, prob in pmf.Items():
    print(hypo, prob)


# Implement the Cookie problem by writing a class that inherits from Suite and providing the Likelihood method.
# Suite implements the Update and Print methods, which are the same for all Bayesian problems,
# but not the Likelihood method, which depends on the specification of the problem.
class CookieProblem(Suite):

    proportions = {
        hypos[0]:dict(vanilla=0.75, chocolate=0.25),
        hypos[1]:dict(vanilla=0.5, chocolate=0.5),
        }

    def Likelihood(self, data, hypo):
        proportion = self.proportions[hypo]
        likelihood = proportion[data]
        return likelihood

# Set up hypotheses for Cookie problem
hypos = ['Bowl 1', 'Bowl 2']
# Initialise prior
pmf = CookieProblem(hypos)
print("Prior:")
pmf.Print()
# Draw some cookies (with replacement) and update the prior
dataset = ['vanilla', 'vanilla', 'chocolate', 'vanilla']
for data in dataset:
    pmf.Update(data)
print("Posterior:")
pmf.Print()


# Write a class for the m & m problem
class M_and_M(Suite):

    mix94 = dict(brown=30,
                 yellow=20,
                 red=20,
                 green=10,
                 orange=10,
                 tan=10)

    mix96 = dict(blue=24,
                 green=20,
                 orange=16,
                 yellow=14,
                 red=13,
                 brown=13)

    hypoA = dict(bag1=mix94, bag2=mix96)
    hypoB = dict(bag1=mix96, bag2=mix94)

    hypotheses = dict(A=hypoA, B=hypoB)

    def Likelihood(self, data, hypo):
        bag, color = data
        mix = self.hypotheses[hypo][bag]
        likelihood = mix[color]
        return likelihood

# Implement the m & m problem
hypos = 'AB' # can also be written: hypos = ['A', 'B']
pmf = M_and_M(hypos)
pmf.Update(('bag1', 'yellow'))
pmf.Update(('bag2', 'green'))
pmf.Print()


# Implement the Cookie problem without replacement
class CookieGetsEaten(Suite):

    def __init__(self, hypos, Bowl1, Bowl2):
        Suite.__init__(self, hypos)
        self.Bowl1 = Bowl1
        self.Bowl2 = Bowl2

    def Likelihood(self, data, hypo):
        if (hypo == "Bowl 1") & (data == "vanilla"):
            likelihood = self.Bowl1.num_vanilla / (self.Bowl1.num_vanilla + self.Bowl1.num_chocolate)
            Bowl1.num_vanilla -= 1
        elif (hypo == "Bowl 1") & (data == "chocolate"):
            likelihood = self.Bowl1.num_chocolate / (self.Bowl1.num_vanilla + self.Bowl1.num_chocolate)
            Bowl1.num_chocolate -= 1
        elif (hypo == "Bowl 2") & (data == "vanilla"):
            likelihood = self.Bowl2.num_vanilla / (self.Bowl2.num_vanilla + self.Bowl2.num_chocolate)
            Bowl2.num_vanilla -= 1
        elif (hypo == "Bowl 2") & (data == "chocolate"):
            likelihood = self.Bowl2.num_chocolate / (self.Bowl2.num_vanilla + self.Bowl2.num_chocolate)
            Bowl2.num_chocolate -= 1
        return likelihood

# Set up a "Bowl" object
class Bowl():
    def __init__(self, num_vanilla=20, num_chocolate=20):
        self.num_vanilla = num_vanilla
        self.num_chocolate = num_chocolate

    def __str__(self):
        return "Vanilla: {}, Chocolate {}.".format(self.num_vanilla, self.num_chocolate)

# Set up hypotheses for Cookie problem
hypos = ['Bowl 1', 'Bowl 2']
# Create two Bowl objects with the right mix of cookies in each
Bowl1 = Bowl(30, 10)
Bowl2 = Bowl(20, 20)
# Initialise prior
pmf = CookieGetsEaten(hypos, Bowl1, Bowl2)
print("Prior:")
pmf.Print()
# Draw some cookies (with replacement) and update the prior
dataset = ['vanilla', 'vanilla', 'chocolate', 'vanilla']
#dataset = ['vanilla', 'chocolate', 'chocolate', 'chocolate', 'chocolate', 'chocolate', 'chocolate', 'chocolate', 'chocolate', 'chocolate', 'chocolate', 'chocolate', 'chocolate', 'chocolate']
for data in dataset:
    pmf.Update(data)
    print("Bowl 1:", Bowl1, "Bowl 2:", Bowl2)
    print("Posterior:")
    pmf.Print()
#print("Posterior:")
pmf.Print()


# Set up a class for doing Bayesian Inference on which of 5 different dice have been rolled:
class Dice(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1 / hypo

# Initialise a pmf for the Dice problem
pmf = Dice([4, 6, 8, 12, 20])  # The hypotheses are: 4-sided, 6-sided, 8-sided, 12-sided, 20-sided
print("Prior:")
pmf.Print()
pmf.Update(6)  # The data is that a 6 was rolled (using one of the dice - we don't know which one)
print("Posterior:")
pmf.Print()
# Calculate a confidence interval at the 90% level for which dice the data could come from
interval = CredibleInterval(pmf, 90)   # Error p28: function Percentile has been replaced with CredibleInterval
print("Confidence interval:", interval)   # typo p28: print interval should read print(interval)
# With more data, posterior distributions based on different priors tend to converge
for roll in [6, 8, 7, 7, 5, 4]:
    pmf.Update(roll)
print("Posterior:")
pmf.Print()
# Calculate a confidence interval at the 90% level for which dice the data could come from
interval = CredibleInterval(pmf, 90)
print("Confidence interval:", interval)
# Calculate the expectation value of the pmf
print("Mean:", pmf.Mean())
# Create a Cdf object from the pmf
cdf = pmf.MakeCdf()
interval = cdf.Percentile(5), cdf.Percentile(95)
print("Confidence interval:", interval)


# Create a coin class for analysing coin-tossing problems
class Coin(Suite):
    # Define the Likelihood method (this version is slow because there is one data point for every coin toss, 
    # so the method has to be called lots of times)
    def Likelihood(self, data, hypo):
        if data == "H":
            likelihood = hypo
        elif data == "T":
            likelihood = 1 - hypo
        return likelihood

# Function for plotting posterior
def plot_bias(pmf):
    bias = []
    prob = []
    for b, p in pmf.Items():
        bias.append(b)
        prob.append(p)
    plt.plot(bias, prob)

# Function for calculating maximum likelihood hypothesis
def MaximumLikelihood(pmf):
    """Returns the hypothesis with the highest probability."""
    prob, val = max((prob, val) for val, prob in pmf.Items())
    return val

# Create 101 hypotheses for the bias of the coin, ranging from 0 to 1
hypos = np.arange(0, 1.01, 0.01)
# Create pmf for coin bias hypotheses
pmf = Coin(hypos)
# Create data set of successive coin tosses
#dataset = "HHHHHTTTT"
dataset = 140 * "H" + 110 * "T"
# Update posterior
#for data in dataset:
#    pmf.Update(data)       # The Update method normalises the pmf for every data point.
pmf.UpdateSet(dataset)    # We can save time by performing all the updates first and only normalising at the end, using UpdateSet. 
# Compute 90% confidence interval
interval = CredibleInterval(pmf, 90)
print("90% confidence interval:", interval)
# Compute maximum likelihood hypothesis
print("Maximum likelihood hypothesis:", MaximumLikelihood(pmf))
# Plot posterior
plot_bias(pmf)
plt.show()


# Create a fast coin class for analysing coin-tossing problems efficiently
class CoinFast(Suite):
    # This verson of Likelihood is fast because the results of all coin tosses are included in one data point: data = (heads, tails)
    # Now, the update of the posterior takes the same amount of time, no matter how many coin tosses there are.
    def Likelihood(self, data, hypo):
        heads, tails = data
        likelihood = hypo**heads * (1-hypo)**tails  # We can multiply all the likelihoods because coin tosses are independent events.
        return likelihood

# Set up hypotheses for the coin-tossing problem
hypos = np.arange(0, 1.01, 0.01)
# Initialise a CoinFast pmf
pmf = CoinFast(hypos)
# Represent the data set as a tuple of (heads, tails)
data = (140, 110)
# Update posterior
pmf.Update(data)
# Plot posterior
plot_bias(pmf)
plt.show()


# Create a class for the Beta distribution
class BetaCHW(object):
    def __init__(self, alpha=1, beta=1):   # By default __init__ makes a uniform distribution
        self.alpha = alpha
        self.beta = beta
    
    # Update performs a Bayesian update:
    def Update(self, data):
        heads, tails = data
        self.alpha += heads
        self.beta += tails

    # Mean calculates the mean of the distribution using a formula that involves only alpha and beta
    def Mean(self):
        return float(self.alpha) / (self.alpha + self.beta)

    # EvalPdf evaluates the probability density function (PDF) of the beta distribution
    def EvalPdf(self, x):
        return x ** (self.alpha - 1)  *  (1 - x) ** (self.beta - 1)

beta = BetaCHW()
beta.Update((140, 110))
print("Mean hypothesis:", beta.Mean())    # Typo p40: print beta.Mean() should read print(beta.Mean())

# Cromwell’s rule: avoid giving a prior probability of 0 to any hypothesis that is even remotely possible.
# If the prior goes to zero, the posterior will always be zero thereafter.


# Ex 4.1: Coin tossing problem when the reader of the coin toss has a probability y of giving the wrong reading. 
class UncertainCoin(Suite):
    def Likelihood(self, data, hypo):
        heads, tails, y = data
        likelihood = (hypo * (1 - y) + (1 - hypo) * y) ** heads  *  ((1 - hypo) * (1 - y) + hypo * y) ** tails
        return likelihood

hypos = np.arange(0, 1.01, 0.01)
pmf = UncertainCoin(hypos)
data = (140, 110, 0)  # (heads, tails, y)
pmf.Update(data)
plot_bias(pmf)

pmf = UncertainCoin(hypos)
data = (140, 110, 0.1)  # (heads, tails, y)
pmf.Update(data)
plot_bias(pmf)

pmf = UncertainCoin(hypos)
data = (140, 110, 0.2)  # (heads, tails, y)
pmf.Update(data)
plot_bias(pmf)

pmf = UncertainCoin(hypos)
data = (140, 110, 0.3)  # (heads, tails, y)
pmf.Update(data)
plot_bias(pmf)

plt.show()


# Try using the Beta distribution to create a more realistic prior for the Belgain Euro coin problem
beta = Beta()               # Uniform prior
beta.Update((140, 110))
pmf = beta.MakePmf()
plot_bias(pmf)

beta = Beta(100, 100)       # Broadish prior centred on bias of 0.5
beta.Update((140, 110))
pmf = beta.MakePmf()
plot_bias(pmf)

beta = Beta(300, 300)       # Narrow prior centred on bias of 0.5 (if alpha, beta > 300, peak is just a spike and we get an error)
beta.Update((140, 110))
pmf = beta.MakePmf()
plot_bias(pmf)

interval = CredibleInterval(pmf, 90)
print("90% confidence interval:", interval)  # Since this comes out at (0.49, 0.55), it now seems less likely the coin is really biased.

plt.show()