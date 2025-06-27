# 1 Normal Distribution

**Example 1**: Conduct 100 groups of coin toss experiments, 10 times per group, and record the number of heads. Represent the frequency of the 100 results (integers from 1 to 10) using a histogram.

- This is a "100 trials of 10-fold Bernoulli experiments."

**Example 2**: Estimate the average English test score of 5,000 freshmen by sampling 5 students each time, calculating the mean, repeating 1,000 times, and plotting the distribution of means in a histogram.

- This is a "1,000 trials of 5-fold Bernoulli experiments."

**Conclusion**: The results of a large number of n-fold Bernoulli experiments produce a normal distribution curve.

# 2 Hypothesis Testing

### 2.1 Basic Concepts

- **Scenario**: Determine whether there is a significant difference in the average English test scores between freshmen and sophomores.
- **Objective**: Determine whether there exists a statistically significant difference between the average college entrance examination scores in English for freshmen and sophomores, when complete academic records of the freshman cohort are unavailable.
- **Hypotheses**:
  - $H_0$ (Null Hypothesis): $\mu_1 = \mu_2$ (no significant difference)
  - $H_1$(Alternative Hypothesis): $\mu_1 \neq \mu_2$(significant difference exists)

### 2.2 Testing Process

1. Set the **significance level $\alpha=0.05$** (typically 0.01 or 0.001).
2. Determine the rejection region (critical region):
   - Two-tailed test: $\frac{\alpha}{2}$ on each side
   - One-tailed test: $\alpha$ on one side
3. Decision rule:
   - If a single sampling of freshmen yields a sample mean that falls within the extreme region (i.e., a statistically rare event occurs), this suggests potential issues with our initial hypothesis.
   - So, if the sample statistic falls in the rejection region ⇒ Reject $H_0$. Otherwise ⇒ Fail to reject $H_0$

## 3 t-Distribution and t-Test

**To eliminate the necessity of performing repeated sampling for every hypothesis test, we standardize these distribution patterns.**

### 3.1 t-Value Formula

$$
t = \frac{\overline{x} - \mu}{s / \sqrt{n}}
$$

Where:

- $\mu$: Hypothesized population mean (e.g., sophomores' average score)
- $\overline{x}$: Sample mean
- $s$: Sample standard deviation
- $n$: Sample size

### 3.2 Example Test

Given $t = 1.77, n = 20, df = 19, \alpha = 0.05$ (one-tailed):

- The area where $t > 1.729$ is 0.05, which is called the p-value. If $p<alpha$, then reject $H_0$ ; otherwise, fail to reject $H_0$ .
- Critical value from table is $1.729$. Since $1.77 > 1.729$ ⇒ Reject $H_0$

### 3.3 Definition of p-Value

"p-value is, under the assumption that the Null Hypothesis is True, the probability of obtaining test results At Least As Extreme As the Results actually observed."

### 3.4 Types of Hypothesis Tests

1. **One-Sample Two-tailed test**:
   - $H_0$：$\mu=x$   $H_1$：$\mu\neq x$
2. **One-Sample One-tailed test**:
   - $H_0$：$\mu\leq x$   $H_1$：$\mu>x$
   - $H_0$：$\mu\geq x$   $H_1$：$\mu<x$

## 4 Confidence Interval

### 4.1 Basic Concept

- **Definition**: An interval that contains the true parameter with a confidence level of $1-\alpha$.
- **Example**: A $95%$ confidence interval corresponds to $\alpha = 0.05$ .

### 4.2 Construction Method

Critical value formula:
$$
X_{\text{critical}} = \mu_{\text{hypothesized population}} + t_{\text{critical}} \cdot \frac{s}{\sqrt{n}}
$$

### 4.3 Interpretation

If infinite samples are drawn from the same population and confidence intervals are constructed:

- $(1-\alpha)\%$ of the intervals will contain the true population mean.
- A specific confidence interval has a $(1-\alpha)\%$ probability of containing the true parameter.
- https://seeing-theory.brown.edu/frequentist-inference/cn.html#section2

### 4.4 Application

When using a sample mean (point estimate) to estimate the population mean, the confidence interval provides the possible range of variation for the estimate. 