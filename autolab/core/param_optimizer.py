import numpy as np
from scipy.optimize import minimize

class ParamOptimizer:
    """智能参数优化模块"""
    
    def __init__(self):
        self.history = []
        
    def optimize(self, objective_func, initial_params, bounds):
        """
        使用贝叶斯优化调整参数
        :param objective_func: 目标函数
        :param initial_params: 初始参数(dict)
        :param bounds: 参数边界(dict)
        :return: 优化后的参数
        """
        # 转换参数格式
        param_names = list(initial_params.keys())
        x0 = np.array([initial_params[name] for name in param_names])
        bounds_list = [bounds[name] for name in param_names]
        
        # 定义优化目标
        def wrapped_func(x):
            params = {name: x[i] for i, name in enumerate(param_names)}
            score = objective_func(params)
            self.history.append((params, score))
            return -score  # 最小化负得分
            
        # 运行优化
        result = minimize(
            wrapped_func,
            x0=x0,
            bounds=bounds_list,
            method='L-BFGS-B',
            options={'maxiter': 50}
        )
        
        # 返回最佳参数
        best_params = {name: result.x[i] for i, name in enumerate(param_names)}
        return best_params
