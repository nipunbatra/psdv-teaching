#import "@preview/polylux:0.4.0": *
#import "@preview/metropolis-polylux:0.1.0" as metropolis
#import metropolis: new-section, focus

#show: metropolis.setup

#set page(
  margin: (top: 3cm, bottom: 3cm, left: 3cm, right: 3cm)
)

#show par: set block(spacing: 2em)
#show list: set block(spacing: 2em)
#new-section[Expectations]

#slide[
  = Expectation: The Big Idea

  - Expectation (or Expected Value) is a fundamental concept in probability theory.
  - It represents the long-run average value of a random variable over many trials.
  - For a discrete random variable, expectation is computed as a weighted sum.
]

#slide[
  = Definition

  - Given a discrete random variable $X$ with probability mass function (PMF) $P(X = x_i) = p_i$, the expectation is:
    $
    E[X] = sum x_i p_i
    $
  - This sum is taken over all possible values of $X$.
]

#slide[
  = Example: Rolling a Die

  - Let $X$ be the outcome of a fair 6-sided die.
  - The possible values are $\{1,2,3,4,5,6\}$ with equal probability $1/6$.
  - Compute expectation:
    $
    E[X] =  sum_(i=1)^6 (i/6)
    $
]

#slide[
  = Linearity of Expectation

  - If $X$ and $Y$ are discrete random variables, then:
    $$
    E[aX + bY] = aE[X] + bE[Y]
    $$
  - This property holds even if $X$ and $Y$ are dependent.
  - Useful for breaking down complex expectations.
]

#slide[
  = Example: Sum of Two Dice

  - Let $X_1$ and $X_2$ be two independent dice rolls.
  - By linearity:
    $$
    E[X_1 + X_2] = E[X_1] + E[X_2] = 3.5 + 3.5 = 7
    $$
  - No need to compute the full distribution of the sum!
]

#slide[
  = Expectation of a Function

  - If $g(X)$ is a function of a discrete random variable $X$:
    $
    E[g(X)] = sum g(x_i) p_i
    $
  - Example: If $X$ is a fair die roll, find $E[X^2]$:
    $
    E[X^2] =  sum_(i=1)^6 i^2/6 = 91/6 = 15.17
    $
]

#slide[
  = Summary

  - Expectation is the weighted sum of values of a random variable.
  - Key properties:
    - *Linearity:* $E[a X + b Y] = a E[X] + b E[Y]$
    - *Function Expectation:* $E[g(X)] = sum g(x_i) p_i$
  - Useful in probability, statistics, and machine learning.
]

#slide[
  #show: focus
  Thank You!  
  - Questions?
]
