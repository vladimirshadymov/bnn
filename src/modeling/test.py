import unittest
import torch
from bnn_modules import BinarizedLinear

class TestBnnModules(unittest.TestCase):

  def test_binarized_linear(self):
      layer = BinarizedLinear(3, 3)
      layer.weight.data = torch.Tensor([[0.5, -0.5, 0.5],
                                        [-0.5, -0.5, 0.5],
                                        [0.5, 0.5, 0.5]])
      layer.bias.data = torch.Tensor([0, 0, 0])
      input_tsr = torch.Tensor([[5., 2., 7.],
                                [1., -3., 16.],
                                [24., 0., 3.]])
      output_tsr = layer(input_tsr)
      expected_tsr = torch.Tensor([[10., 0., 14.],
                                   [20., 18., 14.],
                                   [27., -21., 27.]])
      self.assertTrue(torch.all(torch.eq(output_tsr, expected_tsr)))


if __name__ == '__main__':
    unittest.main()