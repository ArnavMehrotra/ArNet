import numpy as np
import time
import ctypes
import op

MAX_SIZE = 9000

def test_gemm(lib: ctypes.CDLL):
  j = np.random.randint(2, MAX_SIZE)
  k = np.random.randint(2, MAX_SIZE)
  m = k
  n = np.random.randint(2, MAX_SIZE)

  hi = np.random.rand(j, k).astype(np.float32) * 100
  hello = np.random.rand(m, n).astype(np.float32) * 100

  start = time.time()
  correct = hi @ hello
  end = time.time()

  print(f"Total time for numpy: {(end - start)*1000:.3f} ms")

  start = time.time()
  test = op.gemm(hi, hello, lib)
  end = time.time()

  print(f"Total time for young arn: {(end - start)*1000:.3f} ms")

  start = time.time()
  test2 = op.gemm2(hi, hello, lib)
  end = time.time()

  print(f"Total time for young arn (optimized): {(end - start)*1000:.3f} ms")

  print(f"input 1: {j}x{k} matrix")
  print(f"input 2: {m}x{n} matrix")
  print(f"output: {j}x{n} matrix")


  good = np.allclose(correct, test, rtol=1e-3, atol=1e-3) and np.allclose(correct, test2, rtol=1e-3, atol=1e-3)

  return good

def test_biasAdd(lib: ctypes.CDLL):
  j = np.random.randint(2, MAX_SIZE)
  k = np.random.randint(2, MAX_SIZE)

  a = np.random.rand(j, k).astype(np.float32)
  b = np.random.rand(k).astype(np.float32)

  correct = a + b
  test = op.biasAdd(a, b, lib)
  good = np.allclose(correct, test, rtol=1e-3, atol=1e-3)

  print(f"input 1: {j}x{k} matrix")
  print(f"input 2: 1x{k} matrix")
  print(f"output: {j}x{k} matrix")

  return good

def test_scalarAdd(lib: ctypes.CDLL):

  j = np.random.randint(2, MAX_SIZE)
  k = np.random.randint(2, MAX_SIZE)

  a = np.random.rand(j, k).astype(np.float32)
  s = np.random.rand()

  correct = a + s
  test = op.scalarAdd(a, s, lib)
  good = np.allclose(correct, test, rtol=1e-3, atol=1e-3)


  return good

def test_matAdd(lib: ctypes.CDLL):

  j = np.random.randint(2, MAX_SIZE)
  k = np.random.randint(2, MAX_SIZE)

  a = np.random.rand(j, k).astype(np.float32)
  b = np.random.rand(j, k).astype(np.float32)

  correct = a + b
  test = op.matAdd(a, b, lib)

  good = np.allclose(correct, test, rtol=1e-3, atol=1e-3)

  print(f"input 1: {j}x{k} matrix")
  print(f"input 2: {j}x{k} matrix")
  print(f"output: {j}x{k} matrix")

  return good

def test_relu(lib: ctypes.CDLL):

  j = np.random.randint(2, MAX_SIZE)
  k = np.random.randint(2, MAX_SIZE)

  a = np.random.randn(j, k).astype(np.float32)

  correct = np.maximum(a, 0)
  test = op.relu(a, lib)

  good = np.allclose(correct, test, rtol=1e-3, atol=1e-3)

  print(f"input: {j}x{k} matrix")

  return good

def numpy_softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def test_softmax(lib: ctypes.CDLL):
  j = np.random.randint(2, MAX_SIZE)
  k = np.random.randint(2, MAX_SIZE)

  a = np.random.randn(j, k).astype(np.float32) * 100

  test = op.softmax(a, lib)

  check = numpy_softmax(a)

  good = np.allclose(test, check, rtol=1e-3, atol=1e-3)

  return good

def test_gradient(lib: ctypes.CDLL):
  j = np.random.randint(2, MAX_SIZE)
  k = np.random.randint(2, MAX_SIZE)

  a = np.random.rand(j, k).astype(np.float32)
  y = np.random.randint(0, k, size=(j, 1)).astype(np.uint32)

  test = op.gradient(a, y, lib)

  check = numpy_softmax(a)
  check[np.arange(j), y.squeeze()] -= 1

  good = np.allclose(test, check, rtol=1e-3, atol=1e-3)

  return good

def run_tests(lib: ctypes.CDLL):
    assert test_gemm(lib), "gemm failed"
    assert test_matAdd(lib), "matAdd failed"
    assert test_scalarAdd(lib), "scalarAdd failed"
    assert test_relu(lib), "relu failed"
    assert test_softmax(lib), "softmax failed"
    assert test_gradient(lib), "gradient failed"
    assert test_biasAdd(lib), "biasAdd failed"

    print("All tests passed!")