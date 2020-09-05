import numpy as np
import json
from .target_space import TargetSpace


class DomainTransformer():
    '''The base transformer class'''

    def __init__(self, **kwargs):
        pass

    def initialize(self, target_space: TargetSpace):
        raise NotImplementedError

    def transform(self, target_space: TargetSpace):
        raise NotImplementedError


class SequentialDomainReductionTransformer(DomainTransformer):
    """
    A sequential domain reduction transformer bassed on the work by Stander, N. and Craig, K:
    "On the robustness of a simple domain reduction scheme for simulation‐based optimization"
    """

    def __init__(
        self,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9
    ) -> None:
        self.gamma_osc = gamma_osc
        self.gamma_pan = gamma_pan
        self.eta = eta
        pass

    def initialize(self, target_space: TargetSpace) -> None:
        """Initialize all of the parameters"""
        print(f"Init: Target Space.Bounds: {json.dumps(target_space.bounds)} ({type(target_space.bounds)})")
        self.original_bounds = np.copy(target_space.bounds)
        self.bounds = [self.original_bounds]
        print(f"Init: original_bounds: {json.dumps(self.original_bounds)} ({type(self.original_bounds)})")

        self.previous_optimal = np.mean(target_space.bounds, axis=1)
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        print(f"Init: previous_optimal: {json.dumps(self.previous_optimal)} ({type(self.previous_optimal)})")
        print(f"Init: current_optimal: {json.dumps(self.current_optimal)} ({type(self.current_optimal)})")
        self.r = target_space.bounds[:, 1] - target_space.bounds[:, 0]
        print(f"Init: self.r: {json.dumps(self.r)} ({type(self.r)})")
        self.previous_d = 2.0 * \
            (self.current_optimal - self.previous_optimal) / self.r
        print(f"Init: previous_d: {json.dumps(self.previous_d)} ({type(self.previous_d)})")
        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        print(f"Init: current_d: {json.dumps(self.current_d)} ({type(self.current_d)})")
        self.c = self.current_d * self.previous_d
        print(f"Init: c: {json.dumps(self.c)} ({type(self.c)})")
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)
        print(f"Init: c_hat: {json.dumps(self.c_hat)} ({type(self.c_hat)})")

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))
        print(f"Init: gamma: {json.dumps(self.gamma)} ({type(self.gamma)})")

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)
        print(f"Init: contraction_rate: {json.dumps(self.contraction_rate)} ({type(self.contraction_rate)})")

        self.r = self.contraction_rate * self.r
        print(f"Init: self.r: {json.dumps(self.r)} ({type(self.r)})")

    def _update(self, target_space: TargetSpace) -> None:

        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        self.current_optimal = target_space.params[
            np.argmax(target_space.target)
        ]

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _trim(self, new_bounds: np.array, global_bounds: np.array) -> np.array:
        for i, variable in enumerate(new_bounds):
            if variable[0] < global_bounds[i, 0]:
                variable[0] = global_bounds[i, 0]
            if variable[1] > global_bounds[i, 1]:
                variable[1] = global_bounds[i, 1]

        return new_bounds

    def _create_bounds(self, parameters: dict, bounds: np.array) -> dict:
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    def transform(self, target_space: TargetSpace) -> dict:

        self._update(target_space)

        new_bounds = np.array(
            [
                self.current_optimal - 0.5 * self.r,
                self.current_optimal + 0.5 * self.r
            ]
        ).T

        self._trim(new_bounds, self.original_bounds)
        self.bounds.append(new_bounds)
        return self._create_bounds(target_space.keys, new_bounds)
