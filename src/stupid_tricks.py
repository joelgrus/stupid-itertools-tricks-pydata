from itertools import count, repeat, cycle, islice, tee, repeat, accumulate, chain

head = next

def tail(it):
  next(it)
  return it

def take(n, it):
  return [x for x in islice(it, n)]

def drop(n, it):
  return islice(it, n, None)

def iterate(f, x):
  """return (x, f(x), f(f(x)), ...)"""
  return accumulate(repeat(x), lambda fx, _: f(fx))

def until_convergence(it):
  """returns elements of it until the same element appears twice in a row,
  then stops"""
  def no_repeat(prev, curr):
    if prev == curr: raise StopIteration
    else: return curr

  return accumulate(it, no_repeat)
