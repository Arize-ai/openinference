from agents import weather_agent
from swarm.repl import run_demo_loop

if __name__ == "__main__":
    run_demo_loop(weather_agent, stream=True)
