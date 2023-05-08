#!/usr/bin/env python
# coding: utf-8

# #  Assignment_08.03.2023(Advance Statistic-1)

# ## Q1. What is the Probability density Function?
# 
# ## Ans.
#         A probability density function (PDF) is a mathematical function that describes the likelihood of a continuous random
#         variable taking a certain value within a specific range of values. In other words, it describes the distribution of
#         probabilities for a continuous random variable.The PDF can be used to calculate the probability of a random variable
#         falling within a specific range of values by integrating the PDF over that range.
#         
#         The PDF is an important concept in statistics and probability theory, as it provides a way to model and analyze 
#         continuous random variables. It is commonly used in fields such as physics, engineering, finance, and economics to 
#         describe the distribution of continuous data.

# ## Q2. What are the types of probability distribution?
# 
# ### Ans:-  
#     There are several types of probability distributions, each with its own properties and characteristics. Here are some of
#     the most common types of probability distributions:
#     
#     1.Normal distribution:- The normal distribution, also known as the Gaussian distribution, is a bell-shaped curve that is
#     symmetrical around the mean. Many natural phenomena, such as height or weight measurements, follow a normal distribution
#     This distribution part of the PDF.
#     
#     2.Binomial distribution:- The binomial distribution is used to model the number of successes in a fixed number of 
#     independent trials, where each trial has only two possible outcomes (success or failure). Examples include coin tosses
#     or the number of defective items in a sample.This distribution part of the PMF.
#     
#     3.Poisson distribution:- The Poisson distribution is used to model the number of occurrences of a rare event in a fixed
#     interval of time or space. Examples include the number of customers arriving at a store in a given hour or the number of
#     accidents on a highway in a day. This distribution part of the PMF.
#     
#     4.Uniform distribution:- The uniform distribution is used to model a random variable that is equally likely to take any
#     value within a given interval. Examples include the time it takes to complete a task or the distance between two points 
#     in a plane.
#     
#     5.Bernoulli distribution :-The Bernoulli distribution is a discrete probability distribution that models a random 
#     variable that can take on only two possible outcomes, usually labeled as success (1) and failure (0). 
#     it's part of thr PMF.
#     
#     6. Log normal distribution :- The log-normal distribution is a continuous probability distribution that describes a 
#     random variable whose logarithm is normally distributed.The log-normal distribution is commonly used to model 
#     variables that are expected to be positive but are skewed to the right, such as income or stock prices. It can also 
#     arise as a result of multiplying a large number of positive independent random variables. its part of the PDF.
#     
#     7.Power law distribution :- The power law distribution is a continuous probability distribution that describes a 
#     variable that has a relationship between its frequency and its magnitude that follows a power law.it is also known as a
#     Pareto distribution or a long-tailed distribution.
#                                         The power law distribution has been observed in many natural and social phenomena,
#                                         such as the frequency of earthquakes, the size of cities, the number of citations
#                                         received by scientific papers, and the popularity of websites. The power law 
#                                         distribution is also commonly used in network science and complex systems to model
#                                         the degree distribution of nodes in a network.
# 
# 
#    These are just a few examples of the many types of probability distributions that exist. Each distribution has its own set
#    of parameters and properties, which can be used to analyze and model different types of data.    

# ## Q3.Write a python function to calculate the probability density function of a normal distribution with given mean and Standard deviation at given point?
# 
# ### Ans.
#             The probability density function (PDF) of a normal distribution with mean μ and standard deviation σ is given:
#              Formula of PDF in Normal Distribution is = f(x) = (1 / (σ * sqrt(2π))) * e^(-(x-μ)^2 / (2σ^2))
#           
#                                           where where x is any real number, π is the mathematical constant pi,
#                                           e is the mathematical constant e (the base of the natural logarithm), 
#                                           and sqrt represents the square root function.
#                                           
#           The normal distribution is a bell-shaped curve that is symmetric around its mean. The value of the PDF at 
#           any given point x represents the relative likelihood of observing that value in a random sample from the 
#           normal distribution.
# 
#  

# In[1]:


import math
def normal_pdf(x,mean,std_dev):
    """Calculate the PDF of a normal distribution with a given mean 
    and standard deviation at a given point"""
    exponent = math.exp(-((x-mean)**2)/2*(std_dev**2))
    denominator = math.sqrt(2*math.pi)*std_dev
    pdf = exponent/denominator
    return pdf


# In[2]:


import math
def normal_pdf(x,mean,std_dev):
    """Calculate the PDF of a normal distribution with a given mean 
    and standard deviation at a given point"""
    exponent = math.exp(-((x-mean)**2)/2*(std_dev**2))
    denominator = math.sqrt(2*math.pi)*std_dev
    pdf = exponent/denominator
    return pdf
pdf = normal_pdf(2,0,1)
print(pdf)


# ## Q4. What are the Properties of Binomial distribution? Give two examples of events where binomial distribution can applied.
# 
# ### Ans.
#     The binomial distribution is a discrete probability distribution that describes the number of successes in a fixed 
#     number of independent trials, where each trial has the same probability of success. The properties of the binomial
#     distribution include:
# 
#     1.The trials are independent.
#     2.There are a fixed number of trials, denoted by n.
#     3.Each trial has only two possible outcomes: success or failure.
#     4.The probability of success on each trial is denoted by p, and the probability of failure is denoted by q=1-p.
#     5.The random variable X represents the number of successes in n trials.
#     6.The mean of the binomial distribution is μ = np.
#     7.The variance of the binomial distribution is σ² = npq.
#     8.The standard deviation of the binomial distribution is sqrt of npq.
#     9.The shape of the binomial distribution is bell-shaped, skewed to the right if p<0.5 and to the left if p>0.5.
#     
#     Two examples of events where the binomial distribution can be applied are:
# 
#     1.Flipping a coin: Suppose you flip a fair coin 10 times and count the number of times it lands heads. Each flip is
#     independent, and the probability of heads on each flip is 0.5. The number of heads follows a binomial distribution with
#     n=10 and p=0.5.
# 
#     2.Manufacturing defects: A manufacturer produces 1000 units of a product, and each unit has a 5% chance of having a 
#     defect. The number of defective units follows a binomial distribution with n=1000 and p=0.05.

# ## Q5. Generate a random sample of size 1000 from a binomial distribution with probability of Success 0.4 and plot a histogram of the result using matplotlib.
# 
# ## Ans.

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
# Set the parameters of the binomial distribution
n=1000
p= 0.4
# Generate a random sample of size 1000 from the binomial distribution
sample = np.random.binomial(n,p,size=1000)
# Plot a histogram of the sample using Matplotlib
plt.hist(sample,bins=20,alpha=0.5,density=True)
# Add labels and a title to the plot
plt.xlabel('NUMBER OF SUCESSES')
plt.ylabel('FREQUENCY')
plt.title('BINOMIAL DISTRIBUTION (n=1000,p=0.4)')
plt.grid()
plt.show()


# ## Q6. Write a python frunction to calculate the cumulative distribution function of a Poisson distribution with given mean at a given point.
# 
# ## Ans.

# In[4]:


import math
           
def cdf_poisson(mean,x):
    # Calculate the Poisson probability mass function (PMF) for each value up to x
    pmf_values =[math.exp(-mean) * mean**i/math.factorial(i) for i in range (x+i)]
    # Calculate the cumulative sum of the PMF values up to x
    cdf_values = sum(pmf_values)
    return cdf_values


# ## Q7. How Binomial distribution difference from Poisson distribution?
# 
# ## Ans.
#         The binomial and Poisson distributions are both discrete probability distributions, but they are used to model
#         different types of events.
#                   1. The binomial distribution is used to model the number of successes in a fixed number of independent
#                   trials, where each trial has the same probability of success. The trials are assumed to be independent
#                   and the probability of success remains constant from trial to trial. The binomial distribution has two
#                   parameters: the number of trials (n) and the probability of success (p). The mean and variance of the 
#                   binomial distribution are μ = np and σ² = np(1-p),respectively.
#                   
#                   2.the Poisson distribution is used to model the number of occurrences of an event in a fixed interval of
#                   time or space, where the events occur randomly and independently of each other, but the rate of occurrence
#                   (λ) is constant. The Poisson distribution has one parameter: the rate of occurrence (λ). The mean and
#                   variance of the Poisson distribution are both equal to λ.
#                   
#     So Main differences between the Binomial and Poisson distribution are :-
#     
#                     1. The binomial distribution models the number of successes in a fixed number of trials, while the 
#                     Poisson distribution models the number of occurrences in a fixed interval of time or space.
#                     
#                     2.The binomial distribution has two parameters (n and p), while the Poisson distribution has one 
#                     parameter (λ).
#                     
#                     3.The binomial distribution assumes that the trials are independent and the probability of success 
#                     remains constant from trial to trial, while the Poisson distribution assumes that the events occur 
#                     randomly and independently of each other, but the rate of occurrence is constant.
#                     
#                     4.As n approaches infinity and p approaches 0 in the binomial distribution, the distribution approaches
#                     the Poisson distribution with parameter λ = np.

# ## Q8. Generate a random Sample of size 1000 from a possion distribution with mean 5 and calculate the sample mean and variance.
# 
# ## Ans.

# In[5]:


import numpy as np
# Generate the random sample
sample= np.random.poisson(lam=5,size=1000)
# Calculate the sample mean and variance
sample_mean = np.mean(sample)
sample_variance = np.var(sample)

print("Sample Mean:",sample_mean)
print("Sample Variance:",sample_variance)


# ## Q9. How mean and variance are related in Binomial distribution and Poisson distribution.
# 
# ## Ans.
#     In the binomial distribution, the mean and variance are related by the formula:
#     
#     mean = n*p 
#     
#     Variance = n*p*(1-p)
#                             Where n = The number of trials
#                                   p = The probability of sucesses each trails
#                                   
#        In the poisson distribution, the mean and variance are equal by the formula:
#        
#        mean =  λ*t
#        
#        Variance =  λ*t 
#                            Where  λ =  The average rate of events occurring in the given time or space interval.
#                                   t =  Time interval 
#                                   
#      In both distributions, the mean and variance play an important role in determining the shape and spread of the 
#      distribution. The mean represents the center of the distribution, while the variance represents the spread or 
#      variability of the distribution. A higher variance means the distribution is more spread out, while a lower variance
#      means it is more tightly clustered around the mean.

# ## Q10. In normal distribution with respect to mean position, Where dose the least frequent data appear?
# 
# ## Ans.
#         In a normal distribution, the least frequent data appears at the tails of the distribution, which are the regions 
#         that are farthest away from the mean. Specifically, the data that are located beyond two standard deviations from 
#         the mean are considered to be in the tails of the distribution, and they represent the least frequent data.
#         
#         In a normal distribution, about 95% of the data falls within two standard deviations of the mean, and only about
#         2.5% of the data falls in each of the two tails beyond two standard deviations from the mean. Therefore, the data
#         in the tails are relatively rare compared to the data closer to the mean.
#         
#         It's important to note that the position of the mean itself does not determine the frequency of data in a normal 
#         distribution. The shape and spread of the distribution, as well as the position of the individual data points, all 
#         play a role in determining the frequency of data. However, in a symmetrical normal distribution, the mean is located
#         at the center of the distribution, and the frequency of data decreases as you move away from the center towards the
#         tails.
