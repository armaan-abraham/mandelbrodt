import torch
import torch.nn as nn


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
        output_module_hidden_layers=2,
        halt_gate_size=128,
        halt_gate_hidden_layers=1,
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
        assert output_module_hidden_layers >= 1
        assert halt_gate_hidden_layers >= 1

        # LSTM gates
        self.input_gate = LSTMGate(input_size, hidden_size)
        self.forget_gate = LSTMGate(input_size, hidden_size)
        self.cell_gate = LSTMGate(input_size, hidden_size, activation=torch.tanh)
        self.output_gate = LSTMGate(input_size, hidden_size)

        # Halt gate
        self.halt_gate = nn.Sequential(
            nn.Linear(hidden_size, halt_gate_size),
            nn.ReLU(),
            *[
                nn.Linear(halt_gate_size, halt_gate_size)
                for _ in range(halt_gate_hidden_layers - 1)
            ],
            nn.Linear(halt_gate_size, 1),
            nn.Sigmoid(),
        )

        # Output module
        self.state_to_output = nn.Sequential(
            nn.Linear(hidden_size, output_module_size),
            nn.ReLU(),
            *[
                nn.Linear(output_module_size, output_module_size)
                for _ in range(output_module_hidden_layers - 1)
            ],
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

    def update_last_halt_prob(self, halt_probs_list, p_sum, finished_mask, active_mask):
        halt_probs_list[-1][finished_mask & active_mask] += 1 - p_sum[
            finished_mask & active_mask
        ].squeeze(1)
        p_sum_copy = p_sum.clone()
        p_sum_copy[finished_mask & active_mask] = 1
        return p_sum_copy

    def forward(self, x):
        assert x.ndim == 2
        batch_size = x.shape[0]

        p_sum = torch.zeros(batch_size, 1, device=x.device, dtype=torch.float32)
        outputs_list = []
        halt_probs_list = []
        active_masks_list = []

        if self.initial_state is None:
            state = torch.zeros(batch_size, self.hidden_size, device=x.device)
            cell = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            state, cell = self.initial_state

        active_mask = torch.ones(batch_size, device=x.device, dtype=torch.bool)

        iter = 0

        while active_mask.any() and (iter < self.max_iter or self.max_iter is None):
            input_g, forget_g, cell_g, output_g = self.compute_gates(
                x[active_mask], state[active_mask]
            )

            new_cell = cell.clone()
            new_cell[active_mask] = forget_g * cell[active_mask] + input_g * cell_g
            cell = new_cell

            new_state = state.clone()
            new_state[active_mask] = output_g * torch.tanh(cell[active_mask])
            state = new_state

            output = self.state_to_output(state[active_mask])
            outputs_full = torch.zeros(batch_size, self.output_size, device=x.device)
            outputs_full[active_mask] = output
            outputs_list.append(outputs_full)

            halt_prob = self.halt_gate(state[active_mask])

            halt_probs_full = torch.zeros(batch_size, device=x.device)
            halt_probs_full[active_mask] = halt_prob.squeeze(1)
            halt_probs_list.append(halt_probs_full)

            p_sum_new = p_sum.clone()
            p_sum_new[active_mask] += halt_prob
            p_sum = p_sum_new

            active_masks_list.append(active_mask)

            if self.adaptive_time:
                finished = (p_sum >= (1 - self.epsilon)).squeeze(1)

                # Update halt probabilities for finished sequences
                if (finished & active_mask).any():
                    p_sum = self.update_last_halt_prob(
                        halt_probs_list, p_sum, finished, active_mask
                    )

                # Update active_mask
                new_active_mask = active_mask.clone()
                new_active_mask[finished] = False
                active_mask = new_active_mask

            iter += 1

        if self.max_iter is not None and self.adaptive_time and iter >= self.max_iter:
            # update halt_probs for the remaining active elements
            p_sum = self.update_last_halt_prob(
                halt_probs_list, p_sum, active_mask, active_mask
            )

        # Stack outputs and halt_probs
        outputs_tensor = torch.stack(outputs_list, dim=1)
        halt_probs_tensor = torch.stack(halt_probs_list, dim=1)

        iter_taken = torch.sum(
            torch.stack([active_mask for active_mask in active_masks_list]), dim=0
        ).to(torch.int32)

        self.mean_iter_taken = iter_taken.float().mean()

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
