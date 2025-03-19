import tkinter as tk

def visualize_binary_state(states):
    """
    Displays a window with four color blocks in two rows based on binary input.

    Parameters:
        states (tuple or list of 4 elements): Binary values (0 or 1) for the four colors.
    """
    if len(states) != 4:
        raise ValueError("Input must be a tuple or list with exactly 4 binary values (0 or 1).")

    root = tk.Tk()
    root.title("Binary State Visualization")

    # Define variable names
    variable_names = ["Variable 1", "Variable 2", "Variable 3", "Variable 4"]

    # Create a frame for better layout
    frame = tk.Frame(root)
    frame.pack()

    # Create labels and canvases for each variable in a 2-row layout
    for i in range(4):
        row = 0 if i < 2 else 2  # First two variables go to row 0, last two to row 2
        col = i % 2  # Arrange in two columns

        label = tk.Label(frame, text=variable_names[i], font=("Arial", 12))
        label.grid(row=row, column=col, pady=5)

        color = "green" if states[i] == 1 else "red"
        canvas = tk.Canvas(frame, width=100, height=100, bg=color)
        canvas.grid(row=row+1, column=col, padx=10, pady=5)  # Place below the label

    root.mainloop()

# Example Usage
visualize_binary_state((0, 1, 1, 0))  # Example input: 0 -> Red, 1 -> Green
