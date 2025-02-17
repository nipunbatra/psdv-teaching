---
author: Nipun Batra
badges: true
categories:
- ML
date: '2025-2-11'
title: PMF and some common discrete distributions
toc: true

---


```python
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn as nn

import pandas as pd
# Retina mode
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

# PMF of Bernoulli distribution

Let $X$ be a Bernoulli random variable. $X$ can take on one of two values, 0 or 1, with probabilities $1-p$ and $p$, respectively.

Example: Suppose we flip a coin with probability $p$ of landing heads. Let $X$ be the random variable that is 1 if the coin lands heads and 0 if the coin lands tails.

The probability mass function (PMF) of a Bernoulli random variable is given by:

$$
p_X(x) =
\begin{cases}
1-p, & \text{if } x = 0, \\
p, & \text{if } x = 1.
\end{cases}
$$


or, equivalently,

$$
p_X(x) = p^x(1-p)^{1-x}, \quad x \in \{0, 1\}.
$$


where $0 < p < 1$ is called the Bernoulli parameter. We write

$$
X \sim \text{Bernoulli}(p)
$$

to denote that $X$ is drawn from a Bernoulli distribution with parameter $p$.



The probability mass function (PMF) of a Bernoulli distribution is given by:
$$
\begin{equation}
f(x) = \begin{cases}
p & \text{if } x = 1 \\
1 - p & \text{if } x = 0
\end{cases}
\end{equation}
$$

where $p$ is the probability of success.





```python
## Plotting the PMF

def plot_pmf_bernoilli(p, title):
    x = np.array([0, 1])
    y = np.array([1-p, p])
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('P(X=x)')
    plt.xticks([0, 1])

```


```python
plot_pmf_bernoilli(0.3, 'Bernoulli PMF with p=0.3')    
```


```python
plot_pmf_bernoilli(0.8, 'Bernoulli PMF with p=0.3')    
```


```python
dist = torch.distributions.Bernoulli?
```

## Bernoulli Distribution in PyTorch


```python
# Create a Bernoulli distribution with p=0.9
dist = torch.distributions.Bernoulli(probs=0.9)

```


```python
dist
```


```python
# Print all attributes of the Bernoulli distribution -- do not have __ or _ in the beginning
attrs = [attr for attr in dir(dist) if not attr.startswith('_')]
pd.Series(attrs)
```


```python
dist.mean
```


```python
dist.probs
```


```python
dist.support
```


```python
dist.log_prob(torch.tensor(1.0)).exp()
```


```python
dist.log_prob(torch.tensor(0.0)).exp()
```


```python
try:
    dist.log_prob(torch.tensor(0.5)).exp()
except Exception as e:
    print(e)
```


```python
dist.sample()
```


```python
samples = dist.sample(torch.Size([1000]))
```


```python
samples[:10]
```


```python
samples.mean()
```

# Lime vs Lemon

![](../figures/pexels-solodsha-9009923.jpg)

https://www.healthline.com/nutrition/lime-vs-lemon

>Limes are small, green, and more tart than lemons, which are larger, oval-shaped, and yellow. Nutritionally, they’re almost identical and share many of the same potential health benefits.


>Lemons are usually bright yellow, while limes are typically a bright shade of green. However, certain types of limes will turn yellow as they ripen, making the distinction a little more difficult.

>Limes are also smaller and rounder than lemons. They can vary in size but are usually 1–2 inches (3–6 centimeters) in diameter.

>In comparison, lemons tend to be 2–4 inches (7–12 centimeters) in diameter and have a more oval or oblong shape.



## Main question

Given a fruit (lime or lemon) and its radius, we want to predict if it is a lime or a lemon.

Let us denote the radius of the fruit by $r$ and the type of the fruit by $y$ where $y=0$ if the fruit is a lime and $y=1$ if the fruit is a lemon.

We want to model the probability of the fruit being a lemon given its radius, i.e., we want to model $p(y=1|r)$.

## Generative process


```python
# Set random seed
torch.manual_seed(42)
```


```python
radius_array = torch.distributions.Uniform(0, 3).sample((1000,))
```


```python
radius_array[:10]
```


```python
_ = plt.hist(radius_array)
```

We start by modeling the generative process of the data.

We assume if w*r + b > 0, then the fruit is a lemon, otherwise it is a lime.

Let us assume that `w_true` = 1.2 and `b_true` = -2.0.


```python
def linear(r, w, b):
    return w * r + b

w_true = 1.2
b_true = -2.0

logits = linear(radius_array, w_true, b_true)
```


```python
pd.Series(logits.numpy()).describe()
```

Can we use `logits` to model the probability of the fruit being a lemon given its radius? 

No! These logits can be any real number, but we want to model the probability of the fruit being a lemon given its radius, which is a number between 0 and 1.

We can use the sigmoid function to map the logits to a number between 0 and 1.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$




```python
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

probs = sigmoid(logits)
```


```python
df = pd.DataFrame({
    'radius': radius_array.numpy(),
    'logits': logits.numpy(),
    'probabilities': probs.numpy()
})
```


```python
df.head()
```


```python
df.query('radius < 0.2').head()
```

We can observe as per our model, smaller fruits are more likely to be limes (probability of being a lemon is less) and larger fruits are more likely to be lemons (probability of being a lemon is more).

## Generate a dataset


```python
y_true = torch.distributions.Bernoulli(probs=probs).sample()

```


```python
df['y_true'] = y_true.numpy()
```


```python
df.query('y_true == 0').head(10)
```


```python
df.query('y_true == 1').head(10)
```

We can notice that even though the probability of the event is very low, it still happens. This is the nature of the Bernoulli distribution.


```python
df.query('y_true == 0').head(10)
```


```python
# Plot the data
plt.scatter(radius_array, y_true, alpha=0.1, marker='|', color='k')
plt.xlabel('Radius')

# Use Limes and Lemon markers only on y-axis
plt.yticks([0, 1], ['Limes', 'Lemons'])
plt.ylabel('Fruit')
```


```python
# Logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)
    
model = LogisticRegression()

# Training the model
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Convert the data to PyTorch tensors
radius_tensor = radius_array.unsqueeze(1)
y_true_tensor = y_true.unsqueeze(1)

# Training loop
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(radius_tensor)
    
    # Compute loss
    loss = criterion(y_pred, y_true_tensor)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')



```


```python
# Learned weights and bias
w_learned = model.linear.weight.item()
b_learned = model.linear.bias.item()

# Compare the true and learned weights and bias
print(f'True weights: {w_true}, Learned weights: {w_learned}')
print(f'True bias: {b_true}, Learned bias: {b_learned}')

```


```python
# Test if a new fruit is a lime or a lemon

def predict_fruit(radius, model):
    model.eval()
    radius_tensor = torch.tensor([[radius]])
    logits = model(radius_tensor)
    prob = sigmoid(logits).item()
    fruit = ['Lime', 'Lemon'][int(prob > 0.5)]
    return fruit, prob
    
```


```python
predict_fruit(0.5, model)
```


```python
predict_fruit(1.5, model)
```


```python
predict_fruit(2.0, model)
```


```python
# Decision surface
radius_values = torch.linspace(0, 3, 100).unsqueeze(1)
probs = sigmoid(model(radius_values)).detach()

plt.plot(radius_values, probs)
plt.xlabel('Radius')
plt.ylabel('Probability of being a lemon')
plt.title('Decision surface')
plt.ylim(0, 1)
plt.axhline(0.5, color='r', linestyle='--')


```

# Categorical distribution

Say, we have a random variable $X$ that can take on one of $K$ possible values, $1, 2, \ldots, K$. The probability mass function (PMF) of a categorical distribution is given by:

$$
p_X(x) = \begin{cases}
\theta_1, & \text{if } x = 1, \\
\theta_2, & \text{if } x = 2, \\
\vdots \\
\theta_K, & \text{if } x = K.
\end{cases}
$$

where $\theta_1, \theta_2, \ldots, \theta_K$ are the parameters of the categorical distribution and satisfy the following constraints:

$$
0 \leq \theta_i \leq 1, \quad \sum_{i=1}^K \theta_i = 1.
$$

We write

$$
X \sim \text{Categorical}(\theta_1, \theta_2, \ldots, \theta_K)
$$

to denote that $X$ is drawn from a categorical distribution with parameters $\theta_1, \theta_2, \ldots, \theta_K$. The categorical distribution is a generalization of the Bernoulli distribution to more than two outcomes.


If we had a fair 6-sided die, the PMF of the die roll would be given by:

$$
p_X(x) = \frac{1}{6}, \quad x \in \{1, 2, 3, 4, 5, 6\}.
$$



## Imagenet

The ImageNet project is a large visual database designed for use in visual object recognition research. 



![](https://blog.roboflow.com/content/images/2021/06/image-18.png)


```python
theta_vec = torch.tensor([0.1, 0.2, 0.3, 0.4])

#
ser = pd.Series(theta_vec.numpy())
ser.plot(kind='bar', rot=0)
plt.xlabel('Outcome')
plt.ylabel('Probability')
```


```python
dist = torch.distributions.Categorical(probs=theta_vec)
print(dist)
```


```python
dist.support
```


```python
dist.log_prob(torch.tensor(0.0)).exp()
```


```python
try:
    dist.log_prob(torch.tensor(4.0)).exp()
except Exception as e:
    print(e)
```


```python
samples = dist.sample(torch.Size([1000]))

```


```python
samples[:10]
```


```python
pd.value_counts(samples.numpy(), normalize=True).sort_index()
```

# Quality control in factories

A factory produces electronic chips, and each chip has a probability $p$ of being defective due to manufacturing defects. The quality control team randomly selects 10 chips from a large production batch and checks how many are defective.

How can we model the number of defective chips in the sample?

We can model the number of defective chips in the sample using a binomial distribution.

Let $X$ be the number of defective chips in the sample. $X$ can take on values $0, 1, 2, \ldots, 10$. The probability mass function (PMF) of a binomial distribution is given by:

$$
p_X(x) = \binom{n}{x} p^x(1-p)^{n-x}, \quad x \in \{0, 1, 2, \ldots, 10\}.
$$

where $n$ is the number of chips in the sample, $0 < p < 1$ is the probability of a chip being defective, and $\binom{n}{x}$ is the binomial coefficient, which is the number of ways to choose $x$ defective chips from $n$ chips.




```python
p_failure = 0.1
n_chips = 10
dist = torch.distributions.Binomial(n_chips, p_failure)
```


```python
dist
```


```python
dist.support
```


```python
x = torch.arange(0, n_chips+1)
y = dist.log_prob(x).exp()
df_prob_binomial = pd.DataFrame({
    'x': x.numpy(),
    'P(X=x)': y.numpy().round(5)
})

df_prob_binomial
```


```python
df_prob_binomial.plot(kind='bar', x='x', y='P(X=x)', rot=0)
plt.xlabel('x')
plt.ylabel('P(X=x)')
```


```python
samples = dist.sample(torch.Size([1000]))
```


```python
samples[:5]
```


```python
pd.Series(samples.numpy()).value_counts().sort_index()
```


```python

```

# Number of SMS rceived per day

Hat tip: Bayesian Methods for Hackers book



```python
url = "https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/refs/heads/master/Chapter1_Introduction/data/txtdata.csv"
data = pd.read_csv(url, header=None)
```


```python
data.index.name = 'Day'
data.columns = ['Count']
```


```python
data
```


```python
fig, ax = plt.subplots(figsize=(21, 7))
data.plot(kind='bar', rot=0, ax=ax)
```

How can you model the number of SMS messages you receive per day?

We can model the number of SMS messages you receive per day using a Poisson distribution.

Let $X$ be the number of SMS messages you receive per day. $X$ can take on values $0, 1, 2, \ldots$. The probability mass function (PMF) of a Poisson distribution is given by:

$$
p_X(x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x \in \{0, 1, 2, \ldots\}.
$$

where $\lambda > 0$ is the average number of SMS messages you receive per day.



```python
rate_param = 6
dist = torch.distributions.Poisson(rate_param)

```


```python
dist
```


```python
dist.support
```


```python
x_range = torch.arange(0, 20)
y = dist.log_prob(x_range).exp()
df_prob_poisson = pd.DataFrame({
    'x': x_range.numpy(),
    'P(X=x)': y.numpy().round(6)
})

df_prob_poisson
```


```python
df_prob_poisson.plot(kind='bar', x='x', y='P(X=x)', rot=0)
```

# Sales Calls Before a Successful Sale
A sales representative makes cold calls to potential customers. Each call has a probability $p$ of resulting in a successful sale. We want to model the number of calls needed before achieving a successful sale.

We can model the number of calls needed before achieving a successful sale using a geometric distribution.

Let $X$ be the number of calls needed before achieving a successful sale. $X$ can take on values $1, 2, 3, \ldots$. The probability mass function (PMF) of a geometric distribution is given by:

$$
p_X(x) = (1-p)^{x-1}p, \quad x \in \{1, 2, 3, \ldots\}.
$$

where $0 < p < 1$ is the probability of a successful sale on each call.




```python
p = 0.3
dist = torch.distributions.Geometric(p)
```


```python
dist
```


```python
dist.support   
```


```python
dist.sample(torch.Size([10]))   
```


