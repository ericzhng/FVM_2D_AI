# Optimization Algorithms with Details

## Gradient-Based (First-Order) Methods
- **Steepest Descent**: Follows the negative gradient direction with a line search for step size. Simple but slow convergence for ill-conditioned problems like Rosenbrock.
- **Conjugate Gradient (CG)**: Improves steepest descent by using conjugate directions, reducing iterations. Efficient for large-scale, smooth problems with available gradients.
- **Stochastic Gradient Descent (SGD)**: Uses random data subsets for gradient estimation, ideal for large datasets. Fast but noisy convergence.
- **Adam**: Combines momentum and adaptive learning rates for robust, fast convergence. Popular in deep learning for noisy gradients.
- **RMSprop**: Adapts learning rate using a moving average of squared gradients. Effective for non-stationary objectives.
- **Adagrad**: Scales learning rates inversely with past gradient magnitudes. Suits sparse data but may stall for non-convex problems.
- **Adadelta**: Extends Adagrad with a moving average to avoid diminishing learning rates. Robust for long-term optimization.

## Quasi-Newton Methods
- **BFGS**: Approximates the Hessian using gradient differences, updating a full matrix. Fast convergence for smooth problems but memory-intensive.
- **L-BFGS**: Limited-memory version of BFGS, storing only a few vectors. Ideal for high-dimensional problems with fast convergence.
- **Damped BFGS**: Modifies BFGS to stabilize Hessian updates, improving robustness for non-convex problems.
- **Symmetric Rank-One (SR1)**: Updates Hessian with rank-one matrices. Less stable than BFGS but can handle indefinite Hessians.

## Newton-Based Methods
- **Newton’s Method**: Uses exact Hessian for quadratic convergence. Fast near optima but computationally expensive for large problems.
- **Truncated Newton**: Approximates Newton steps using iterative solvers (e.g., CG). Scales well for large-scale problems.

## Derivative-Free (Direct Search) Methods
- **Nelder-Mead (Simplex)**: Manipulates a simplex of points to search for minima. Robust for noisy or non-smooth functions but slow for high dimensions.
- **Pattern Search**: Explores a mesh of points around the current solution. Flexible for constrained or noisy problems.
- **Powell’s Method**: Uses conjugate directions without derivatives. Efficient for small-scale, smooth problems.
- **COBYLA**: Approximates constraints linearly for constrained optimization. Suitable for small-scale problems.
- **DIRECT**: Divides search space systematically. Global search for black-box functions but evaluation-heavy.

## Evolutionary and Metaheuristic Algorithms
- **Genetic Algorithms (GA)**: Mimics natural selection with crossover and mutation. Global search for non-convex problems but requires many evaluations.
- **Differential Evolution (DE)**: Uses vector differences for mutation. Robust for global optimization with moderate evaluation counts.
- **Particle Swarm Optimization (PSO)**: Simulates social behavior of particles. Simple, effective for continuous global optimization.
- **Simulated Annealing (SA)**: Probabilistic search inspired by annealing. Good for escaping local minima in noisy problems.
- **Ant Colony Optimization (ACO)**: Mimics ant foraging, best for discrete or combinatorial problems.
- **Evolutionary Strategies (ES)**: Focuses on mutation and selection. Scalable for high-dimensional global optimization.

## Bayesian Optimization
- **Gaussian Process-based BO**: Models function with a Gaussian Process, minimizing evaluations. Ideal for expensive black-box functions.
- **Tree-structured Parzen Estimator (TPE)**: Uses probabilistic models for hyperparameter tuning. Efficient for high-dimensional spaces.
- **SMAC**: Sequential model-based configuration. Combines random forests with BO for robust optimization.

## Trust-Region Methods
- **Trust-Region Newton**: Solves subproblems within a trust region using Hessian. Robust for non-convex problems.
- **Dogleg Method**: Combines Newton and gradient steps in trust region. Balances efficiency and stability.
- **Cauchy Point Method**: Simplest trust-region approach using gradient steps. Fast but less accurate.

## Interior-Point Methods
- **Primal-Dual Interior Point**: Solves constrained problems by balancing primal and dual variables. Efficient for large-scale constrained optimization.
- **Log-Barrier Method**: Transforms constraints into logarithmic penalties. Robust for convex constrained problems.

## Augmented Lagrangian Methods
- **Augmented Lagrangian**: Combines penalties and Lagrange multipliers for constraints. Flexible for non-linear constraints.
- **LANCELOT**: Specialized for large-scale constrained problems with augmented Lagrangian framework.

## Coordinate Descent Methods
- **Cyclic Coordinate Descent**: Optimizes one variable at a time in a fixed order. Simple, effective for separable problems.
- **Randomized Coordinate Descent**: Randomly selects variables to optimize. Scalable for high-dimensional problems.

## Surrogate-Based Methods
- **Radial Basis Function (RBF) Optimization**: Interpolates function with RBFs. Efficient for black-box functions with moderate dimensions.
- **Kriging**: Similar to Gaussian Process BO, models function uncertainty. Used in engineering design optimization.

## Hybrid Methods
- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: Adapts covariance matrix for mutation. Robust for non-convex, high-dimensional problems.
- **Memetic Algorithms**: Combines evolutionary algorithms with local search (e.g., CG or L-BFGS). Balances global and local search.