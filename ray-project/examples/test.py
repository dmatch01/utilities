import ray
import time
import numpy as np
@ray.remote(num_cpus=1)
def estimate_pi(num_samples):
    time.sleep(3)
    xs = np.random.uniform(low=-1.0, high=1.0, size=num_samples)
    ys = np.random.uniform(low=-1.0, high=1.0, size=num_samples)
    xys = np.stack((xs, ys), axis=-1)
    inside = xs*xs + ys*ys <= 1.0
    xys_inside = xys[inside]
    in_circle = xys_inside.shape[0]
    approx_pi = 4.0*in_circle/num_samples
    return approx_pi
ray.init(address=‘auto’)
num_samples = 10000
num_tasks = 128
refs = [estimate_pi.remote(num_samples) for _ in range(num_tasks)]
ray.get(refs)
