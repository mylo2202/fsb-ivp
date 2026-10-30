import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

class GLSDemo:
    def __init__(self, img_path='micro.tif'):
        # Load and handle image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback synthetic image if file is missing
            print(f"Warning: Could not load '{img_path}'. Using fallback default image.")
            img = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))

        self.img = img

        # Default control points (x, y) in [0, 255] range
        self.xpts = [0.0, 255.0]
        self.ypts = [0.0, 255.0]

        # State tracking for mouse interactions
        self.active_pt_idx = None
        self.press_pt = None
        self.dragged = False

        # Create GUI layout using Matplotlib
        self.fig = plt.figure(figsize=(12, 4.5))
        self.fig.canvas.manager.set_window_title('Gray Level Slicing Demo')

        self.ax_img1 = self.fig.add_subplot(1, 3, 1)
        self.ax_slice = self.fig.add_subplot(1, 3, 2)
        self.ax_img2 = self.fig.add_subplot(1, 3, 3)

        # Show Original Image
        self.ax_img1.imshow(self.img, cmap='gray', vmin=0, vmax=255)
        self.ax_img1.set_title('Original Image')
        self.ax_img1.axis('off')

        # Configure Slicing Axes
        self.ax_slice.set_xlim(0, 255)
        self.ax_slice.set_ylim(0, 255)
        self.ax_slice.set_title('Transformation Function')
        self.ax_slice.grid(True, linestyle=':')

        # Setup plot handles
        self.xi = np.arange(256)
        self.line, = self.ax_slice.plot([], [], 'b-', linewidth=2)
        self.points_plot, = self.ax_slice.plot([], [], 'go', markersize=7, markeredgecolor='k')

        # Show Adjusted Image Placeholder
        self.im_display = self.ax_img2.imshow(self.img, cmap='gray', vmin=0, vmax=255)
        self.ax_img2.set_title('Adjusted Image')
        self.ax_img2.axis('off')

        # Add "Continuous Update" Checkbox
        ax_check = plt.axes([0.42, 0.02, 0.18, 0.05], frameon=False)
        self.check = CheckButtons(ax_check, ['Continuous Update'], [True])

        # Connect Event Handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)

        # Initial plot & image update
        self.update_all()
        plt.tight_layout()
        plt.show()

    def get_interpolated_lut(self):
        """Replicates plot_data: interpolates xpts & ypts to 0-255 lookup table."""
        # Sort points by X coordinate to ensure valid 1D linear interpolation
        sorted_indices = np.argsort(self.xpts)
        sorted_x = np.array(self.xpts)[sorted_indices]
        sorted_y = np.array(self.ypts)[sorted_indices]

        yi = np.interp(self.xi, sorted_x, sorted_y)
        return np.clip(yi, 0, 255).astype(np.uint8)

    def update_all(self):
        """Updates line, points, and target image display."""
        lut = self.get_interpolated_lut()

        # Update curve plot
        self.line.set_data(self.xi, lut)
        self.points_plot.set_data(self.xpts, self.ypts)

        # Replicate update_img: Map original image through Lookup Table (LUT)
        adjusted_img = cv2.LUT(self.img, lut)
        self.im_display.set_data(adjusted_img)

        self.fig.canvas.draw_idle()

    def on_button_press(self, event):
        if event.inaxes != self.ax_slice or event.xdata is None or event.ydata is None:
            return

        click_x, click_y = event.xdata, event.ydata
        self.press_pt = (click_x, click_y)
        self.dragged = False

        # Check proximity to existing points
        distances = [np.hypot(x - click_x, y - click_y) for x, y in zip(self.xpts, self.ypts)]
        min_dist = min(distances) if distances else float('inf')

        if min_dist < 10.0:  # Clicked near an existing point
            self.active_pt_idx = int(np.argmin(distances))
        else:  # Clicked on line/axes background: add a new point
            self.xpts.append(click_x)
            self.ypts.append(click_y)
            self.active_pt_idx = len(self.xpts) - 1
            self.update_all()

    def on_motion(self, event):
        if self.active_pt_idx is None or event.inaxes != self.ax_slice:
            return

        new_x, new_y = event.xdata, event.ydata
        if new_x is None or new_y is None:
            return

        self.dragged = True
        idx = self.active_pt_idx

        # Enforce end-point constraints (X fixed for x=0 and x=255 endpoints)
        if idx == 0:
            self.xpts[idx] = 0.0
        elif idx == 1:
            self.xpts[idx] = 255.0
        else:
            self.xpts[idx] = np.clip(new_x, 0.0001, 254.9999)

        self.ypts[idx] = np.clip(new_y, 0.0001, 255.0)

        # Update if continuous update checkbox is checked
        if self.check.get_status()[0]:
            self.update_all()

    def on_button_release(self, event):
        if self.active_pt_idx is not None:
            # Delete point if it was a quick click without dragging (and not endpoints 0 or 1)
            if not self.dragged and self.active_pt_idx not in (0, 1):
                self.xpts.pop(self.active_pt_idx)
                self.ypts.pop(self.active_pt_idx)

            self.active_pt_idx = None
            self.press_pt = None
            self.dragged = False
            self.update_all()


if __name__ == '__main__':
    # Launch demo
    GLSDemo('micro.tif')