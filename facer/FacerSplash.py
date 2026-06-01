import tkinter as tk

SPLASH_DURATION_MS = 3000

class FacerSplash:
    splash = None

    def __init__(self, root):
        """Show a borderless 3-second splash, then call on_done() to start the app."""
        self.splash = tk.Toplevel(root)
        self.splash.overrideredirect(True)  # no title bar / borders

        frame = tk.Frame(self.splash, bg="#1e1e2e", padx=60, pady=40)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame, text="Facer", bg="#1e1e2e", fg="white",
            font=("Helvetica", 36, "bold"),
        ).pack()
        tk.Label(
            frame, text="by marcohern", bg="#1e1e2e", fg="#a6adc8",
            font=("Helvetica", 14),
        ).pack(pady=(8, 0))

        # Center on screen.
        self.splash.update_idletasks()
        w, h = self.splash.winfo_width(), self.splash.winfo_height()
        x = (self.splash.winfo_screenwidth() - w) // 2
        y = (self.splash.winfo_screenheight() - h) // 2
        self.splash.geometry(f"+{x}+{y}")

    def close(self, on_done):
        self.splash.destroy()
        on_done()

    def show(self, on_done):
        """Show the splash, then call on_done() after a delay."""
        self.splash.after(SPLASH_DURATION_MS, lambda: (self.close(on_done)))
