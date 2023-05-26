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

[Linear Regression:](./linreg.py)

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n$$

[Logistic Regression:](./logreg.py)

$$P(y=1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n)}}$$

[Naive Bayes:](./nbayes)

$$P(y | x_1, x_2, \ldots, x_n) = \frac{P(y) \cdot P(x_1 | y) \cdot P(x_2 | y) \cdot \ldots \cdot P(x_n | y)}{P(x_1) \cdot P(x_2) \cdot \ldots \cdot P(x_n)}$$

[Decision Trees:](./dtree.py)

Recursively partitioning the data based on feature values and creating a tree-like model of decisions and their outcomes

[Random Forest:](./rforest.py)

Ensemble learning method that combines multiple decision trees to make predictions, improves upon the individual decision tree's performance by reducing overfitting and increasing robustness

[Support Vector Machine:](svm.py)

$$f(x) = \text{sign}(\sum_{i=1}^{n} w_i x_i + b)$$

[K-Nearest Neighbors:](knn.py)

Classifies or predicts the target variable based on the k nearest neighbors in the feature space

[Mean Squared Error:](./mse.py)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

[Mean Absolute Error:](./mae.py)

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

[R-squared (Coefficient of Determination):](./rsquare.py)

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$
