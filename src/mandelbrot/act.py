import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


class LSTMGate(nn.Module):
    def __init__(self, input_size, hidden_size, activation=torch.sigmoid):
        super().__init__()
        self.input_weight = nn.Linear(input_size, hidden_size)
        self.hidden_weight = nn.Linear(hidden_size, hidden_size)
        self.activation = activation

    def forward(self, x, h):
        return self.activation(self.input_weight(x) + self.hidden_weight(h))


class ACT(nn.Module):
    def __init__(
        self,
        input_size=2,
        hidden_size=1024,
        output_size=1,
        output_module_size=128,
        epsilon=1e-3,
        initial_state=None,
        max_iter=1000,
        adaptive_time=True,
    ):
        super().__init__()
        self.output_size = output_size
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.initial_state = initial_state
        self.max_iter = max_iter
        self.adaptive_time = adaptive_time

        # LSTM gates
        self.input_gate = LSTMGate(input_size, hidden_size)
        self.forget_gate = LSTMGate(input_size, hidden_size)
        self.cell_gate = LSTMGate(input_size, hidden_size, activation=torch.tanh)
        self.output_gate = LSTMGate(input_size, hidden_size)

        # Halt gate
        self.halt_gate = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

        # Output module
        self.state_to_output = nn.Sequential(
            nn.Linear(hidden_size, output_module_size),
            nn.ReLU(),
            nn.Linear(output_module_size, output_size),
            nn.Sigmoid(),
        )

    def compute_gates(self, x, state):
        return (
            self.input_gate(x, state),
            self.forget_gate(x, state),
            self.cell_gate(x, state),
            self.output_gate(x, state),
        )

    def update_last_halt_prob(self, halt_probs, p_sum, finished_mask, active_mask):
        halt_probs[-1][1][finished_mask[active_mask]] += 1 - p_sum[finished_mask & active_mask].squeeze(1)
        p_sum_copy = p_sum.clone()
        p_sum_copy[finished_mask & active_mask] = 1
        return p_sum_copy

    def forward(self, x):
        assert x.ndim == 2
        batch_size = x.shape[0]

        p_sum = torch.zeros(batch_size, 1, device=x.device, dtype=torch.float32)
        outputs = []
        halt_probs = []

        if self.initial_state is None:
            state = torch.zeros(batch_size, self.hidden_size, device=x.device)
            cell = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            state, cell = self.initial_state

        active_mask = torch.ones(batch_size, 1, device=x.device, dtype=torch.bool)
        active_mask_squeezed = active_mask.squeeze(1)

        iter = 0

        while active_mask.any() and (iter < self.max_iter or self.max_iter is None):
            input_g, forget_g, cell_g, output_g = self.compute_gates(
                x[active_mask_squeezed], state[active_mask_squeezed]
            )

            new_cell = cell.clone()
            new_cell[active_mask_squeezed] = (
                forget_g * cell[active_mask_squeezed] + input_g * cell_g
            )
            cell = new_cell

            new_state = state.clone()
            new_state[active_mask_squeezed] = output_g * torch.tanh(
                cell[active_mask_squeezed]
            )
            state = new_state

            output = self.state_to_output(state[active_mask_squeezed])
            outputs.append((active_mask_squeezed.clone(), output))

            halt_prob = self.halt_gate(state[active_mask_squeezed])
            p_sum_new = p_sum.clone()
            p_sum_new[active_mask_squeezed] += halt_prob
            p_sum = p_sum_new

            halt_probs.append((active_mask_squeezed.clone(), halt_prob.squeeze(1)))

            if self.adaptive_time:
                finished = p_sum >= (1 - self.epsilon)

                # Update halt probabilities for finished sequences
                if (finished.squeeze(1) & active_mask_squeezed).any():
                    p_sum = self.update_last_halt_prob(
                        halt_probs, p_sum, finished.squeeze(1), active_mask_squeezed
                    )

                # Update active_mask
                new_active_mask = active_mask.clone()
                new_active_mask[finished] = False
                active_mask = new_active_mask
                active_mask_squeezed = active_mask.squeeze(1)

            iter += 1

        if self.max_iter is not None and self.adaptive_time and iter >= self.max_iter:
            # update halt_probs for the remaining active elements
            # TODO: check this
            p_sum = self.update_last_halt_prob(
                halt_probs, p_sum, active_mask_squeezed, active_mask_squeezed
            )
            active_mask = torch.zeros_like(active_mask)

        # Aggregate outputs and halt_probs
        outputs_tensor = torch.zeros(
            batch_size, iter, self.output_size, device=x.device
        )
        halt_probs_tensor = torch.zeros(batch_size, iter, device=x.device)

        iter_taken = torch.sum(
            torch.stack([active_idx for active_idx, _ in outputs]), dim=0
        )

        for iter, (active_idx, out) in enumerate(outputs):
            outputs_tensor[active_idx, iter] = out

        for iter, (active_idx, h_prob) in enumerate(halt_probs):
            halt_probs_tensor[active_idx, iter] = h_prob

        if self.adaptive_time:
            with torch.no_grad():
                assert torch.allclose(
                    torch.sum(halt_probs_tensor, dim=1),
                    torch.ones(batch_size, device=x.device),
                    atol=1e-5,
                )
            final_output = torch.sum(
                outputs_tensor * halt_probs_tensor.unsqueeze(-1), dim=1
            )
            self.iter_taken = iter_taken
            self.remainder = halt_probs_tensor[:, iter_taken - 1]
        else:
            final_output = outputs_tensor[:, self.max_iter - 1]
            self.iter_taken = iter_taken  # should be max_iter
            self.remainder = torch.ones(batch_size, device=x.device)

        return final_output


# Example usage:
if __name__ == "__main__":
    model = ACT()
    x = torch.randn(5, 2)
    output, iterations, last_halt_prob = model(x)
    print("Output:", output)
    print("Iterations:", iterations)
    print("Last Halt Probabilities:", last_halt_prob)
