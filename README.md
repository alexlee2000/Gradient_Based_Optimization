# Gradient Based Optimization
We will be exploring some algorithms for gradient based optimization such as backpropagation via stochastic gradient descent and steepest descent without using any existing imlpementations of gradient methods (i.e. from scratch). Furthermore, we will also look at how we can use an automatic differentiation library called JAX to compute the gradient of the loss function at each step.

## Part 1 - Gradient Descent (from scratch) 
Consider the optimization problem 

![Screen Shot 2021-08-28 at 7 22 09 am](https://user-images.githubusercontent.com/43845085/131189951-c41a6ca4-1e77-48ca-a2b3-3ffa17c63dd7.png)

where 

![Screen Shot 2021-08-28 at 7 22 12 am](https://user-images.githubusercontent.com/43845085/131189972-8035440e-e060-4172-8551-0a9772dea7d6.png)

and where A ∈ R^m×n, b ∈ R^m are defined as

![Screen Shot 2021-08-28 at 7 23 20 am](https://user-images.githubusercontent.com/43845085/131190031-5befd4f5-1978-4a0d-8798-00309d033c2e.png)

We will be running gradient descent on f using a step size of alpha = 0.1, starting at point of x(0) = (1, 1, 1, 1). The algorithm will terminate when the following condition is met: 

![Screen Shot 2021-08-28 at 7 25 01 am](https://user-images.githubusercontent.com/43845085/131190202-c69a8223-ee7d-4fcc-8e97-706d5b1312e7.png)

This termination condition is reasonable as it ensures that we stop near the minimum i.e. when the gradient is less than 0.001 and thus very close to zero. Since gradients can be positive or negative depending on where we are on the plane, the L2-Norm ensures that the gradients are positive which enables us to code this termination condition so eloquently.

The gradient update can be derived as shown below

![image](https://user-images.githubusercontent.com/43845085/131192068-065ca0f2-bec1-46b3-b74f-bf6c13e545f5.png)

Using the above, we can now build this algorithm in python. I have shown the output of the weights of the first 5 iterations and the last 5 iterations below using the provided code. 

![Screen Shot 2021-08-28 at 7 55 35 am](https://user-images.githubusercontent.com/43845085/131192689-325d7aee-1295-49ef-9d63-15157fe6e849.png)

We can see that the algorithm converged in 221 iterations (k = 222).


## Part 2 - Steepest Gradient Descent (from scratch)
Taking a constant step size is sub-optimal. Ideally, we would take larger steps at the start and take smaller steps as we move closer towards the minimum. This would speed up convergence and reduce the number of iterations required in most cases. 

The method of steepest descent is almost identical to that of gradient descent as shown in part 1, except at each iteration k, we choose 

![Screen Shot 2021-08-28 at 7 42 31 am](https://user-images.githubusercontent.com/43845085/131191595-47200f54-a31c-4d08-807d-209f40848a71.png)

The step size (alpha) is chosen to minimize an objective at each iteration of the gradient method. The objective is different at each step since it depends on the current x-value. 

The step size (alpha) can be derived as shown below 

![image](https://user-images.githubusercontent.com/43845085/131191910-00abda30-0d10-4e56-a224-d5ae52528bfc.png)

Using the above, we can now build this algorithm. I have shown the output of the weights of the first 5 iterations and the last 5 iterations below using the provided code.

![Screen Shot 2021-08-28 at 7 55 26 am](https://user-images.githubusercontent.com/43845085/131192709-9bc4245f-4ce5-4cfa-ac11-7a6e13e0ec2d.png)

We can see that the algorithm converged in 89 iterations (k = 90).


I have also plotted alpha over all iterations to highlight how the step size changes over time

![image](https://user-images.githubusercontent.com/43845085/131192249-68813943-96e1-46d8-adc0-de4d7ddcce69.png)

## Part 3 - Discussion (Gradient Descent vs Steepest Descent) 
Steepest gradient descent required less iterations to converge compared to the gradient descent with a constant step size as expected. Thus, we can see that steepest descent is more efficient at finding the minimizer of a function as this method takes large steps at the beginning when we are away from the optimum, adjusting the steps as we get closer to the minimum. However, computing the extra gradient at each iteration might be expensive, especially when we do not have an easy f as in this case. Furthermore, since the steps in normal gradient descent is fixed, we don’t have to worry about learning rate decay whereas steepest gradient descent is prone to issues revoling around learning rate decay.

## Part 4 - Gradient Descent (using JAX) 

