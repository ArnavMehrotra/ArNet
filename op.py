import ctypes
import numpy as np


def gemm(a: np.array, b: np.array, lib: ctypes.CDLL) -> np.array:

  if(a.dtype != np.float32 or b.dtype != np.float32):
    print("data type must be float32")
  j, k = a.shape
  m, n = b.shape

  if(m != k):
    print("matrix dimensions do not match")
    return

  N = j * n

  out = np.zeros(N, dtype=np.float32)

  lib.launchMult.argtypes =  [ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int]

  a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  c_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  lib.launchMult(a_ptr, b_ptr, c_ptr, j, k, m, n)

  c_np = np.ctypeslib.as_array(c_ptr, (N,)).reshape(j, n)

  return c_np


def gemm2(a: np.array, b: np.array, lib: ctypes.CDLL) -> np.array:
  if(a.dtype != np.float32 or b.dtype != np.float32):
    print("data type must be float32")
  j, k = a.shape
  m, n = b.shape

  if(m != k):
    print("matrix dimensions do not match")
    return

  N = j * n

  out = np.zeros(N, dtype=np.float32)

  lib.launchMult2.argtypes =  [ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int]

  a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  c_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  lib.launchMult2(a_ptr, b_ptr, c_ptr, j, k, m, n)

  c_np = np.ctypeslib.as_array(c_ptr, (N,)).reshape(j, n)

  return c_np

def gradient(a: np.array, y: np.array, lib: ctypes.CDLL):
  if a.dtype != np.float32:
    print("data type must be float32")
    return
  
  if y.dtype != np.uint32:
    print("label index type must be uint32")
    return

  j, k = a.shape
  N = j * k

  out = np.zeros(N, dtype=np.float32)

  lib.launchGradient.argtypes =  [ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_uint32),
                                  ctypes.POINTER(ctypes.c_float),
                                  ctypes.c_int,
                                  ctypes.c_int]


  a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
  b_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  lib.launchGradient(a_ptr, y_ptr, b_ptr, j, k)

  b_np = np.ctypeslib.as_array(b_ptr, (N,)).reshape(j, k)

  return b_np

def biasAdd(a: np.array, b: np.array, lib: ctypes.CDLL) -> np.array:
  if a.dtype != np.float32 or b.dtype != np.float32:
    print("data type must be float32")

  j, k = a.shape
  n = b.shape[0]

  if k != n:
    print("matrix dimensions do not match")
    return

  N = j * k

  out = np.zeros(N, dtype=np.float32)

  lib.launchBiasAdd.argtypes =  [ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.c_int,
                              ctypes.c_int]
  
  a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  c_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  lib.launchBiasAdd(a_ptr, b_ptr, c_ptr, j, k)

  c_np = np.ctypeslib.as_array(c_ptr, (N,)).reshape(j, k)

  return c_np

def scalarAdd(a: np.array, s: float, lib: ctypes.CDLL) -> np.array:
  if a.dtype != np.float32:
    print("data type must be float32")

  j, k = a.shape
  N = j * k

  out = np.zeros(N, dtype=np.float32)

  lib.launchScalarAdd.argtypes =  [ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.c_float,
                              ctypes.c_int]
  
  a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  c_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  lib.launchScalarAdd(a_ptr, c_ptr, s, N)

  c_np = np.ctypeslib.as_array(c_ptr, (N,)).reshape(j, k)

  return c_np

def matAdd(a: np.array, b: np.array, lib: ctypes.CDLL) -> np.array:

  if(a.dtype != np.float32 or b.dtype != np.float32):
    print("data type must be float32")

  j, k = a.shape
  m, n = b.shape

  if m != j or n != k:
    print("matrix dimensions do not match")
    return

  N = j * k

  out = np.zeros(N, dtype=np.float32)

  lib.launchAdd.argtypes =  [ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.c_int,
                              ctypes.c_int]

  a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  c_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  lib.launchAdd(a_ptr, b_ptr, c_ptr, j, k)

  c_np = np.ctypeslib.as_array(c_ptr, (N,)).reshape(j, k)

  return c_np

def relu(a: np.array, lib: ctypes.CDLL) -> np.array:

  if(a.dtype != np.float32):
    print("data type must be float32")

  j, k = a.shape

  N = j * k

  out = np.zeros(N, dtype=np.float32)

  lib.launchRelu.argtypes =  [ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float),
                              ctypes.c_int]

  a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  b_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  lib.launchRelu(a_ptr, b_ptr, N)

  c_np = np.ctypeslib.as_array(b_ptr, (N,)).reshape(j, k)

  return c_np

def softmax(a: np.array, lib: ctypes.CDLL) -> np.array:
  if a.dtype != np.float32:
    print("data type must be float32")
  
  j, k = a.shape
  N = j * k

  out = np.zeros(N, dtype=np.float32)

  lib.launchSoftmax.argtypes =  [ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.c_int,
                                 ctypes.c_int]
  a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  b_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  lib.launchSoftmax(a_ptr, b_ptr, j, k)
  c_np = np.ctypeslib.as_array(b_ptr, (N,)).reshape(j, k)

  return c_np