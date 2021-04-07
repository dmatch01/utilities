import os
import ray
import time
import numpy as np

HEAD_SERVICE_IP_ENV = "EXAMPLE_CLUSTER_RAY_HEAD_SERVICE_HOST"
HEAD_SERVICE_CLIENT_PORT_ENV = "EXAMPLE_CLUSTER_RAY_HEAD_SERVICE_PORT_CLIENT"

@ray.remote
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

head_service_ip = os.environ[HEAD_SERVICE_IP_ENV]
client_port = os.environ[HEAD_SERVICE_CLIENT_PORT_ENV]
ray.util.connect(f"{head_service_ip}:{client_port}")

num_samples = 10000
num_tasks = 4
refs = [estimate_pi.remote(num_samples) for _ in range(num_tasks)]
ray.get(refs)
