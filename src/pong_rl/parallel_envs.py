"""Subprocess environments supporting individual lane pauses and resets."""

import multiprocessing as mp
import traceback

import numpy as np
from gymnasium.vector.utils import CloudpickleWrapper


def _worker(connection, factory):
    env = None
    try:
        env = factory()
        connection.send((True, (env.observation_space, env.action_space)))
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command == "step":
                result = env.step(argument)
            elif command == "reset":
                result = env.reset(**argument)
            else:
                raise ValueError(f"Unknown environment command: {command}")
            connection.send((True, result))
    except EOFError:
        pass
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        connection.close()


class _RemoteEnv:
    def __init__(self, connection):
        self.connection = connection

    def receive(self):
        try:
            success, result = self.connection.recv()
        except EOFError as error:
            raise RuntimeError("Environment worker exited unexpectedly") from error
        if not success:
            raise RuntimeError(f"Environment worker failed:\n{result}")
        return result

    def reset(self, **kwargs):
        self.connection.send(("reset", kwargs))
        return self.receive()


class ParallelEnvPool:
    """A worker per environment; only explicitly active lanes receive steps.

    Spawn avoids inheriting the parent's CUDA context. Unlike ordinary vector
    stepping, completed lanes remain untouched until the collector resets them.
    """

    def __init__(self, factories):
        self.envs, self.processes = [], []
        self.closed = False
        context = mp.get_context("spawn")
        try:
            for factory in factories:
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, CloudpickleWrapper(factory)), daemon=True)
                process.start()
                child.close()
                self.envs.append(_RemoteEnv(parent))
                self.processes.append(process)
            spaces = [env.receive() for env in self.envs]
            self.single_observation_space, self.single_action_space = spaces[0]
            if any(pair != spaces[0] for pair in spaces[1:]):
                raise ValueError("Parallel environments must have matching spaces")
        except Exception:
            self.close()
            raise

    def reset(self, *, seed=None):
        for i, env in enumerate(self.envs):
            env.connection.send(("reset", {"seed": None if seed is None else seed + i}))
        results = [env.receive() for env in self.envs]
        return np.stack([obs for obs, _ in results]), {}

    def step_active(self, active, actions):
        for i, action in zip(active, actions):
            self.envs[i].connection.send(("step", action))
        return [self.envs[i].receive() for i in active]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for env in self.envs:
            try:
                env.connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()
        for env in self.envs:
            env.connection.close()
