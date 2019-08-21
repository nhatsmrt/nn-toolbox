from torch import nn, Tensor
from typing import List, Tuple, Union, Optional


__all__ = ['NeuralAbstractionPyramid']


class NeuralAbstractionPyramid(nn.Module):
    """
    Neural Abstraction Pyramid Module. Sharing weights both spatially and temporally:

    a^t_l = norm(activation(f_l(a^{t - 1}_l) + g_l(a^{t - 1}_{l - 1}) + h_l(a^{t - l}_{l + 1})))

    If f_l, g_l and h_l are repeated for all layers, then we can also share weights across depth dimension.

    (UNTESTED)

    References:

        Sven Behnke and Ralil Rojas. "Neural Abstraction Pyramid: A hierarchical image understanding architecture."
        http://page.mi.fu-berlin.de/rojas/1998/pyramid.pdf

        Sven Behnke. "Hierarchical Neural Networks for Image Interpretation."
        https://www.ais.uni-bonn.de/books/LNCS2766.pdf

        Sven Behnke. "Face Localization and Tracking in the Neural Abstraction Pyramid."
        https://www.ais.uni-bonn.de/behnke/papers/nca04.pdf
    """

    def __init__(
            self, lateral_connections: List[nn.Module], forward_connections: List[nn.Module],
            backward_connections: List[nn.Module], activation_function: nn.Module,
            normalization: nn.Module, duration: int
    ):
        """
        Note that here we assume the forward direction increase the resolution and the backward direction
        reverse the resolution. This can always be reversed.

        :param lateral_connections: consist of depth + 1 conv layers, each with output of same dimension as input.
        Aggregate information from a local neighborhood of the same resolution from previous timestep.
        :param forward_connections: consist of depth downsampling conv layers.
        Transform information from a region of larger resolution (i.e previous layer) from the previous timestep.
        :param backward_connections: consist of depth upsampling layers
        Retrieve feedback from a region of smaller resolution (i.e next layer) from the previous timestep.
        :param duration: default number of timesteps to process data
        """
        assert len(lateral_connections) - 1 == len(forward_connections) == len(backward_connections)
        super().__init__()
        self.depth = len(forward_connections)
        self.duration = duration
        self.lateral_connections = nn.ModuleList(lateral_connections)
        self.forward_connections = nn.ModuleList(forward_connections)
        self.backward_connections = nn.ModuleList(backward_connections)
        self.activ_norm = nn.Sequential(activation_function, normalization)

    def forward(
            self, input: Tensor, return_all_states: bool=False, duration: Optional[int]=None
    ) -> Union[List[Tensor], Tuple[List[Tensor], List[List[Tensor]]]]:
        """
        :param input:
        :param return_all_states: whether to return output of all timesteps
        :param duration: number of timesteps to process data
        :return: the output of last time steps and outputs of all time steps
        """
        if duration is None: duration=self.duration
        assert duration > 0

        states = self.get_initial_states(input)
        all_states = [states]
        for t in range(duration):
            new_states = []
            for l in range(self.depth + 1):
                new_state = self.lateral_connections[l](states[l])
                if l > 0: new_state = new_state + self.forward_connections[l - 1](states[l - 1])
                if l < self.depth: new_state = new_state + self.backward_connections[l](states[l + 1])
                new_state = self.activ_norm(new_state)
                new_states.append(new_state)
            states = new_states
            all_states.append(states)
        if return_all_states:
            return states, all_states
        else:
            return states
        # return states, all_states if return_all_states else states

    def get_initial_states(self, input: Tensor) -> List[Tensor]:
        ret = [input]
        for layer in self.forward_connections:
            input = layer(input)
            ret.append(input)
        return ret
