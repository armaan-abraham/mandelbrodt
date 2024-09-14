import torch

# TODO: how to initialize the state?
# TODO: dtypes


class InputGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_weight = torch.nn.Linear(input_size, hidden_size)
        self.hidden_weight = torch.nn.Linear(hidden_size, hidden_size)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h):
        return torch.sigmoid(self.input_weight(x) + self.hidden_weight(h) + self.bias)


class ForgetGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_weight = torch.nn.Linear(input_size, hidden_size)
        self.hidden_weight = torch.nn.Linear(hidden_size, hidden_size)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h):
        return torch.sigmoid(self.input_weight(x) + self.hidden_weight(h) + self.bias)


class CellGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_weight = torch.nn.Linear(input_size, hidden_size)
        self.hidden_weight = torch.nn.Linear(hidden_size, hidden_size)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h):
        return torch.tanh(self.input_weight(x) + self.hidden_weight(h) + self.bias)


class OutputGate(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_weight = torch.nn.Linear(input_size, hidden_size)
        self.hidden_weight = torch.nn.Linear(hidden_size, hidden_size)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h):
        return torch.sigmoid(self.input_weight(x) + self.hidden_weight(h) + self.bias)


class HaltGate(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = torch.nn.Linear(hidden_size, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, h):
        return torch.sigmoid(self.weight(h) + self.bias)


class ACT(torch.nn.Module):
    def __init__(self, n_iter=None, tau=1e-3, epsilon=1e-3, initial_state=None):
        """If n_iter is None, we use adaptive computation time."""
        # TODO: add hyperparameters as arguments
        super().__init__()
        self.n_iter = n_iter
        self.tau = tau

        # LSTM state update parameters
        self.input_size = 2
        self.hidden_size = 1024
        self.output_size = 1

        self.input_gate = InputGate(self.input_size, self.hidden_size)
        self.forget_gate = ForgetGate(self.input_size, self.hidden_size)
        self.cell_gate = CellGate(self.input_size, self.hidden_size)
        self.output_gate = OutputGate(self.hidden_size, self.hidden_size)

        # adaptive computation time parameters
        self.halt_gate = HaltGate(self.hidden_size)

        self.output_module_size = 128
        self.state_to_output_module = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.output_module_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.output_module_size, self.output_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        p_sum = torch.zeros(1)
        iter = 0

        if self.initial_state is None:
            state = torch.zeros(1, self.hidden_size)
        else:
            state = self.initial_state.clone()

        outputs = []
        halt_probs = []

        while True:
            input_g = self.input_gate(x, state)
            forget_g = self.forget_gate(x, state)
            cell_g = self.cell_gate(x, state)
            output_g = self.output_gate(state)

            cell_candidate = forget_g * state + input_g * cell_g
            state = output_g * torch.tanh(cell_candidate)

            outputs.append(self.state_to_output_module(state))

            halt_prob = self.halt_gate(state)
            p_sum += halt_prob
            halt_probs.append(halt_prob)

            iter += 1

            # stopping condition
            if self.n_iter is None:
                # adaptive computation time
                if p_sum >= (1 - self.epsilon):
                    halt_probs[-1] = 1 - (p_sum - halt_probs[-1])
                    break
            else:
                if iter >= self.n_iter:
                    break

        return (
            torch.sum(torch.stack(outputs) * torch.stack(halt_probs)[:, None], dim=0),
            len(outputs),
            halt_probs[-1],
        )
