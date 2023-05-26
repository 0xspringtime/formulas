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

[K-Means Clustering:](./kmc.py)

1. Initialize the cluster centroids $\mu_1, \mu_2, \ldots, \mu_K$
2. Assign each data point $x_i$ to the nearest centroid $\mu_j$ based on the Euclidean distance $\text{argmin}_j ||x_i - \mu_j||^2$
3. Update the centroids by computing the mean of the data points assigned to each centroid $\mu_j = \frac{1}{N_j} \sum_{i=1}^{N_j} x_i$ 
4. Repeat steps 2 and 3 until convergence or a maximum number of iterations is reached.

[Within-Cluster Sum of Squares:](./wcss.py)

$$WCSS = \sum_{i=1}^{n} \sum_{j=1}^{k} w_{ij} \cdot ||x_i - \mu_j||^2$$

[Principal Component Analysis:](./pca.py)

1. Standardize the data (optional but recommended): $$X_{\text{std}} = \frac{X - \text{mean}(X)}{\text{std}(X)}$$ where X is the original data, mean(X) is the mean along each feature, and std(X) is the standard deviation along each feature.
2. Compute the covariance matrix: $$\text{covariance} = \frac{1}{n} X_{\text{std}}^T X_{\text{std}}$$ where n is the number of data points.
3. Compute the eigenvectors and eigenvalues of the covariance matrix: $$\text{eigenvectors}, \text{eigenvalues} = \text{eig}(\text{covariance})$$
4. Sort the eigenvalues in descending order and select the top k eigenvectors corresponding to the largest eigenvalues.
5. Project the data onto the selected eigenvectors: $$\text{transformed\_data} = X_{\text{std}} \times \text{selected\_eigenvectors}$$

[Gradient Descent:](./grad.py)

1. Initialize the parameters $\theta = [\theta_0, \theta_1, \ldots, \theta_n]$
2. Calculate the gradient of the cost function: $$\nabla J(\theta) = \left[ \frac{\partial J}{\partial \theta_0}, \frac{\partial J}{\partial \theta_1}, \ldots, \frac{\partial J}{\partial \theta_n} \right]$$
3. Update the parameters using the learning ratei $\alpha$: $\theta := \theta - \alpha \cdot \nabla J(\theta)$

[Rectified Linear Unit (ReLU):](./relu.py)

$$f(x) = \max(0, x)$$

[Forward Propagation:](./fprop.py)

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

[Bavkward Propogation:](./bprop.py)

1. Compute the error gradient at the output layer: $$dZ^{[L]} = A^{[L]} - Y$$
2. Propagate the error gradients backward through the layers: $$dZ^{[l]} = (W^{[l+1]})^T \cdot dZ^{[l+1]} \cdot g'^{[l]}(Z^{[l]})$$ $$dW^{[l]} = \frac{1}{m} \cdot dZ^{[l]} \cdot (A^{[l-1]})^T$$ $$db^{[l]} = \frac{1}{m} \cdot \text{np.sum}(dZ^{[l]}, \text{axis}=1, \text{keepdims}=True)$$
3. Update the parameters using the error gradients and a learning rate: $$W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]}$$ $$b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]}$$

[Cross-Entropy Loss](./crossloss.py)

$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

[Sigmoid Acrivation:](./sigmoid)

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

[Convolutional Neural Networks:](./cnn.py)

1. Convolutional Layer:
-Convolution Operation: It involves convolving a set of learnable filters (also called kernels) with the input data to produce feature maps.
-Activation Function: Typically, an activation function such as ReLU is applied element-wise to the output of the convolution operation.
2. Pooling Layer:
-Pooling Operation: It downsamples the feature maps by extracting the most prominent features, reducing the spatial dimensions and making the network more robust to variations in position.
-Common pooling operations include Max Pooling and Average Pooling.
3. Fully Connected Layers:
-Flatten: The output feature maps from the last convolutional/pooling layer are flattened into a 1D vector.
-Fully Connected Layers: These layers operate similar to those in a traditional neural network, where each neuron is connected to all neurons in the previous layer.
-Activation Functions: Activation functions like ReLU are applied to the outputs of the fully connected layers.

[Batch Normalization:](./batchn)

$$\text{BN}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$ with $\mu_B$ mean of the input batch, $\sigma_B^2$ variance of the input batch, $\gamma$ and $\beta$ learnable parameters
