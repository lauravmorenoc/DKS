import matplotlib.pyplot as plt
import time
import random

class BinaryStateVisualizer:
    def __init__(self):
        self.states = [0, 0, 0, 0]  # Initial states

        # Set up the figure
        self.fig, self.ax = plt.subplots()
        self.circles = [plt.Circle((i, 0), 0.4, fc='red', edgecolor='black') for i in range(4)]

        for circle in self.circles:
            self.ax.add_patch(circle)

        self.ax.set_xlim(-1, 4)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        plt.ion()  # Turn on interactive mode
        plt.show()

    def update_states(self, new_states):
        """Update the binary states (list of 4 elements: 0 or 1)."""
        if len(new_states) == 4:
            self.states = new_states
            for i, circle in enumerate(self.circles):
                circle.set_facecolor('green' if self.states[i] else 'red')  # Change color
            self.fig.canvas.draw()  # Ensure the figure updates
            plt.pause(0.1)  # Small pause to allow GUI refresh

# Example Usage
if __name__ == "__main__":
    visualizer = BinaryStateVisualizer()

    try:
        for _ in range(50):  # Simulating a loop in the main code
            new_values = [random.randint(0, 1) for _ in range(4)]  # Generate random 0s and 1s
            print("high1")
            visualizer.update_states(new_values)
            print("high2")
            time.sleep(0.5)  # Main loop delay

    except KeyboardInterrupt:
        pass