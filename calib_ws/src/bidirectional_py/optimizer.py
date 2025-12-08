import numpy as np
from scipy.optimize import minimize
from .loss import LossCalculator
from .config import Config

class CalibrationOptimizer:
    def __init__(self, config=Config):
        self.config = config
        self.loss_calc = LossCalculator(config)

    def optimize(self, lidar_points, stereo_cloud, init_rvec, init_tvec):
        """
        Optimize extrinsics (rvec, tvec) to minimize bidirectional loss.
        """
        # Initial guess: concatenation of rvec and tvec (6 parameters)
        x0 = np.concatenate([init_rvec, init_tvec])
        
        def objective_function(x):
            rvec = x[:3]
            tvec = x[3:]
            loss = self.loss_calc.bidirectional_loss(lidar_points, stereo_cloud, rvec, tvec)
            # print(f"Loss: {loss:.4f}")
            return loss
            
        print("Starting optimization...")
        res = minimize(
            objective_function,
            x0,
            method='Nelder-Mead', # Derivative-free method, robust but slow
            options={'maxiter': self.config.NUM_ITERATIONS, 'disp': True}
        )
        
        optimized_rvec = res.x[:3]
        optimized_tvec = res.x[3:]
        
        print("Optimization finished.")
        print(f"Final Loss: {res.fun:.4f}")
        
        return optimized_rvec, optimized_tvec

    def optimize_semantic(self, lidar_clusters, car_mask, init_rvec, init_tvec):
        """
        Optimize extrinsics using semantic loss.
        """
        x0 = np.concatenate([init_rvec, init_tvec])
        
        def objective_function(x):
            rvec = x[:3]
            tvec = x[3:]
            loss = self.loss_calc.semantic_loss(lidar_clusters, car_mask, rvec, tvec)
            print(f"Semantic Loss: {loss:.4f}")
            return loss
            
        print("Starting semantic optimization...")
        res = minimize(
            objective_function,
            x0,
            method='Nelder-Mead',
            options={'maxiter': self.config.NUM_ITERATIONS, 'disp': True}
        )
        
        optimized_rvec = res.x[:3]
        optimized_tvec = res.x[3:]
        
        print("Semantic Optimization finished.")
        print(f"Final Semantic Loss: {res.fun:.4f}")
        
        return optimized_rvec, optimized_tvec
