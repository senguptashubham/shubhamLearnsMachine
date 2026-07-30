import pytest
import torch
import torch.nn as nn
from my_conv2d import MyConv2d
from typing import Callable
import torch.utils.benchmark as benchmark
import logging

@pytest.fixture
def make_conv_pair() -> Callable[..., tuple]:
  def _make(in_channels, out_channels, kernel_size, padding=0, stride=1, batch=2, H=8, W=8):
    torch.manual_seed(0)
    x = torch.randn(batch, in_channels, H, W)
    my_conv = MyConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    ref_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    ref_conv.weight.data = my_conv.weight.data.clone()
    assert ref_conv.bias is not None
    ref_conv.bias.data = my_conv.bias.data.clone()
    return x, my_conv, ref_conv
  return _make


def test_output_shape(make_conv_pair):
  x, my_conv, ref_conv = make_conv_pair(3, 4, kernel_size=3, padding=1, stride=2)
  out_mine = my_conv(x)
  out_ref = ref_conv(x)
  assert out_mine.shape == out_ref.shape

def test_matches_reference_conv2d(make_conv_pair):
  x, my_conv, ref_conv = make_conv_pair(3, 4, kernel_size=3, padding=1, stride=2)
  out_mine = my_conv(x)
  out_ref = ref_conv(x)
  assert torch.allclose(out_mine, out_ref, atol=1e-5)

def test_gradients_flow(make_conv_pair):
  x, my_conv, ref_conv = make_conv_pair(3, 4, kernel_size=3, padding=1, stride=2)
  out_mine = my_conv(x)
  out_mine.sum().backward()
  assert my_conv.weight.grad is not None
  assert not torch.isnan(my_conv.weight.grad).any()

def test_speed(make_conv_pair):
  x, my_conv, ref_conv = make_conv_pair(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, batch=2, H=32, W=32)
  timer_mine = benchmark.Timer(
    stmt='my_conv(x)',
    globals={'my_conv': my_conv, 'x':x}
  )
  timer_ref = benchmark.Timer(
    stmt='ref_conv(x)',
    globals={'ref_conv': ref_conv, 'x':x}
  )
  result_mine = timer_mine.timeit(100)
  result_ref = timer_ref.timeit(100)

  logger = logging.getLogger(__name__)
  logger.info(result_mine)
  logger.info(result_ref)