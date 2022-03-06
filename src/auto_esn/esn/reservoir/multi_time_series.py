import torch
from torch import nn, Tensor


class MultiTimeSeriesHandler(nn.Module):
    def __init__(self, esn_cell):
        super().__init__()
        self.esn_cell = esn_cell

    def forward(self, input: Tensor, washout=0) -> Tensor:
        tensor_size = input.shape
        if len(tensor_size) == 2:
            return self.esn_cell(input, washout)
        elif len(tensor_size) == 3:
            num_of_time_series = tensor_size[0]
            result = None
            for i in range(num_of_time_series):
                self.esn_cell.reset_hidden()
                ith_result = self.esn_cell(input[i], washout)
                if result is None:
                    result = torch.empty((tensor_size[0], ith_result.shape[0], ith_result.shape[1]))
                result[i, :, :] = ith_result
            return result
        else:
            raise ValueError(f"Only Matrices of dim 2 and 3 are supported but dim {len(tensor_size)} was passed.")
    
    def to_cuda(self):
        self.esn_cell.to_cuda()
