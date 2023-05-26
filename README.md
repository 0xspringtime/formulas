[Standard Deviation:](./stdev.py)

$$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

[Covariance:](./cov.py)

$${cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

[Pearson's r:](./pearsonr.py)

$$r = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}$$

[Binomial Distribution:](./binomial.py)

$$P(X=k) = \binom{n}{k} \cdot p^k \cdot (1-p)^{n-k}$$

[Poisson Distribution:](poisson.py)

$$P(X=k) = \frac{e^{-\lambda} \cdot \lambda^k}{k!}$$

[Normal Distribution:](normal.py)

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

[Central Limit Theorem:](./clt.py)

Let $X_1, X_2, \ldots, X_n$ be a sequence of independent and identically distributed random variables with mean $\mu$ and standard deviation $\sigma$. If $n$ is sufficiently large, the distribution of the sample mean $\bar{X}$ approaches a Gaussian distribution with mean $\mu$ and standard deviation $\sigma/\sqrt{n}$ as $n$ tends to infinity.

[Confidence Interval:](./conf.py)

$$\bar{x} \pm Z \left(\frac{\sigma}{\sqrt{n}}\right)$$


