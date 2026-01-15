import numpy as np
import matplotlib.pyplot as plt

# --- 1. Objective Function (The "Food Source") ---
# We use the Sphere function: f(x) = sum(x^2). Goal is to find the minimum (0,0)
def objective_function(x):
    return np.sum(x**2)

# --- 2. Krill Herd Algorithm Class ---
class KrillHerd:
    def __init__(self, pop_size=25, dimensions=2, max_iter=100):
        self.pop_size = pop_size
        self.dim = dimensions
        self.max_iter = max_iter
        
        # KHA Constants (standard values from Gandomi & Alavi)
        self.Vf = 0.02    # Foraging speed
        self.Nmax = 0.01  # Max induced speed
        self.Dt = 0.005   # Diffusion speed
        self.Cr = 0.2     # Crossover probability
        self.Mu = 0.05    # Mutation probability
        
        # Initialize Krill positions randomly between -10 and 10
        self.positions = np.random.uniform(-10, 10, (self.pop_size, self.dim))
        self.fitness = np.array([objective_function(p) for p in self.positions])
        
        # Best solution found
        self.best_idx = np.argmin(self.fitness)
        self.best_pos = self.positions[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        
        self.history = []

    def solve(self):
        for iteration in range(self.max_iter):
            new_positions = np.zeros_like(self.positions)
            
            # Identify Food Position (approximate as the current best)
            food_pos = self.best_pos 
            
            for i in range(self.pop_size):
                # 1. Induced Movement (Social influence)
                # Simplified: Move toward the best and away from the worst
                alpha_induced = (food_pos - self.positions[i]) * self.Nmax
                
                # 2. Foraging Motion (Food attraction)
                beta_foraging = (food_pos - self.positions[i]) * self.Vf
                
                # 3. Physical Diffusion (Random walk)
                diffusion = self.Dt * np.random.uniform(-1, 1, self.dim)
                
                # Calculate Velocity
                velocity = alpha_induced + beta_foraging + diffusion
                
                # Update Position
                new_pos = self.positions[i] + velocity
                
                # 4. Genetic Operators (Crossover & Mutation)
                # Crossover with the best krill
                if np.random.rand() < self.Cr:
                    new_pos = (new_pos + self.best_pos) / 2
                
                # Mutation
                if np.random.rand() < self.Mu:
                    new_pos += np.random.normal(0, 0.1, self.dim)
                
                new_positions[i] = new_pos

            # Update population and evaluate fitness
            self.positions = new_positions
            self.fitness = np.array([objective_function(p) for p in self.positions])
            
            # Find new best
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_pos = self.positions[current_best_idx].copy()
            
            self.history.append(self.best_fitness)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best Fitness = {self.best_fitness:.6f}")

        return self.best_pos, self.best_fitness, self.history

# --- 3. Execution and Visualization ---
if __name__ == "__main__":
    # Initialize KHA
    kha = KrillHerd(pop_size=30, dimensions=2, max_iter=50)
    best_sol, best_fit, history = kha.solve()

    print(f"\nOptimization Finished!")
    print(f"Best Coordinates: {best_sol}")
    print(f"Minimum Value Found: {best_fit}")

    # Plotting the Convergence Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='blue', linewidth=2)
    plt.title('Krill Herd Algorithm: Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Objective Value)')
    plt.yscale('log') # Use log scale to see progress clearly
    plt.grid(True)
    plt.show()