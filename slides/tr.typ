#import "@preview/polylux:0.4.0": *
#import "@preview/metropolis-polylux:0.1.0" as metropolis
#import metropolis: new-section, focus
#show: metropolis.setup
#set page(
 margin: (top: 3cm, bottom: 3cm, left: 3cm, right: 3cm)
)
#show par: set block(spacing: 2em)
#show list: set block(spacing: 2em)

#new-section[Introduction]
#slide[
  **= Discrete Probability Distributions**
  - We'll explore common discrete probability distributions
  - For each distribution, we'll examine:
    - Definition and parameters
    - Probability mass function (PMF)
    - Expected value
    - Variance
    - Real-world applications
]

#new-section[Bernoulli Distribution]
#slide[
  **= Bernoulli Distribution: Basics**
  - Simplest discrete distribution
  - Models binary outcomes (success/failure)
  - Single parameter $p$ (probability of success)
  - PMF: $P(X = k) = p^k(1-p)^(1-k)$ for $k in {0,1}$
]

#slide[
  **= Bernoulli: Expectation**
  - Expected value: $E[X] = p$
  - Variance: $Var(X) = p(1-p)$
  - Example: Coin flip
    - Fair coin: $p = 1/2$, $E[X] = 1/2$
    - Biased coin: $p = 0.7$, $E[X] = 0.7$
]

#slide[
  **= Bernoulli: Applications**
  - Quality control (pass/fail)
  - Medical tests (positive/negative)
  - Customer conversion (buy/don't buy)
  - Email marketing (click/no click)
]

#new-section[Binomial Distribution]
#slide[
  **= Binomial Distribution: Basics**
  - Sum of $n$ independent Bernoulli trials
  - Parameters: $n$ (trials), $p$ (success probability)
  - PMF: $P(X = k) = binom(n,k)p^k(1-p)^(n-k)$
  - $k$ ranges from $0$ to $n$
]

#slide[
  **= Binomial: Expectation**
  - Expected value: $E[X] = np$
  - Variance: $Var(X) = np(1-p)$
  - Key insight: Linear scaling with $n$
]

#slide[
  **= Binomial: Examples**
  - Example 1: 10 coin flips
    - $n = 10$, $p = 1/2$
    - $E[X] = 5$ heads expected
  - Example 2: Quality control
    - Testing 100 items, 5% defect rate
    - $E[X] = 5$ defects expected
]

#new-section[Geometric Distribution]
#slide[
  **= Geometric Distribution: Basics**
  - Models trials until first success
  - Single parameter $p$ (success probability)
  - PMF: $P(X = k) = p(1-p)^(k-1)$
  - $k$ starts from 1
]

#slide[
  **= Geometric: Expectation**
  - Expected value: $E[X] = 1/p$
  - Variance: $Var(X) = (1-p)/p^2$
  - Memoryless property:
    - Past trials don't affect future probability
]

#slide[
  **= Geometric: Applications**
  - Number of attempts until:
    - First sale in cold calling
    - First defective item in inspection
    - First success in repeated experiments
]

#new-section[Poisson Distribution]
#slide[
  **= Poisson Distribution: Basics**
  - Models rare events in fixed interval
  - Single parameter $lambda$ (rate)
  - PMF: $P(X = k) = (lambda^k e^(-lambda))/k!$
  - $k$ ranges from 0 to infinity
]

#slide[
  **= Poisson: Expectation**
  - Expected value: $E[X] = lambda$
  - Variance: $Var(X) = lambda$
  - Unique property: $E[X] = Var(X)$
]

#slide[
  **= Poisson: Applications**
  - Website visitors per hour
  - Defects per unit area
  - Accidents per month
  - Mutations per DNA segment
]

#new-section[Negative Binomial]
#slide[
  **= Negative Binomial: Basics**
  - Models trials until $r$ successes
  - Parameters: $r$ (successes), $p$ (probability)
  - Generalizes geometric distribution
  - PMF: $P(X = k) = binom(k-1,r-1)p^r(1-p)^(k-r)$
]

#slide[
  **= Negative Binomial: Expectation**
  - Expected value: $E[X] = r/p$
  - Variance: $Var(X) = r(1-p)/p^2$
  - Note: Geometric is special case ($r = 1$)
]

#new-section[Comparison]
#slide[
  **= Distribution Summary**
  - Bernoulli: Single trial
    - $E[X] = p$
  - Binomial: Fixed trials, count successes
    - $E[X] = np$
  - Geometric: Trials until success
    - $E[X] = 1/p$
  - Poisson: Rate-based events
    - $E[X] = lambda$
]

#slide[
  **= Choosing Distributions**
  - Bernoulli: Single yes/no outcome
  - Binomial: Fixed number of independent trials
  - Geometric: Time/trials until first success
  - Poisson: Rate of rare events
  - Negative Binomial: Trials until r successes
]

#new-section[Properties of Expectation]
#slide[
  **= Key Properties**
  - Linearity: $E[aX + bY] = aE[X] + bE[Y]$
  - Non-negativity: If $X >= 0$, then $E[X] >= 0$
  - Monotonicity: If $X <= Y$, then $E[X] <= E[Y]$
  - Law of total expectation:
    $E[X] = E[E[X|Y]]$
]

#slide[
  **= Practical Tips**
  - Use linearity to break down complex problems
  - Consider conditional expectation for dependent events
  - Remember variance is $E[X^2] - (E[X])^2$
  - Use moment generating functions for advanced problems
]

#new-section[Applications]
#slide[
  **= Real-world Applications**
  - Quality Control
    - Defect rates and sampling
  - Finance
    - Investment returns
  - Biology
    - Genetic mutations
  - Computer Science
    - Algorithm analysis
]

#slide[
  **= Machine Learning Applications**
  - Loss function optimization
  - Dropout in neural networks
  - Random forest sampling
  - Cross-validation splitting
]

#slide[
#show: focus
  Thank You!
  - Questions?
  - References:
    - Ross, "A First Course in Probability"
    - Wasserman, "All of Statistics"
]