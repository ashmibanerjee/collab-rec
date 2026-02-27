import shutil

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def set_paper_style():
    """Set matplotlib style for publication-quality figures with improved legibility."""
    sns.set_theme(style="whitegrid")

    # Font settings - larger sizes for better readability
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 20  # Increased from 14
    plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True

    # Axes settings - bolder and larger for visibility
    plt.rcParams['axes.labelsize'] = 18  # Increased from 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 20  # Increased from 18
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2.0  # Increased from 1.5
    plt.rcParams['axes.labelpad'] = 8  # More space between label and axis

    # Tick settings - larger and bolder
    plt.rcParams['xtick.labelsize'] = 24 # Increased from 14
    plt.rcParams['ytick.labelsize'] = 24  # Increased from 14
    plt.rcParams['xtick.major.size'] = 6  # Longer tick marks
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['xtick.major.width'] = 1.5  # Thicker tick marks
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams.update({
        "axes.labelweight": "bold",
        "axes.titleweight": "bold"
    })

    # Legend settings - more prominent
    plt.rcParams['legend.fontsize'] = 16 # Increased from 13
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.95  # Slightly more opaque
    plt.rcParams['legend.edgecolor'] = 'black'  # Changed from gray to black
    plt.rcParams['legend.fancybox'] = False  # Square corners for cleaner look
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['legend.borderpad'] = 0.5  # Padding inside legend box
    plt.rcParams['legend.labelspacing'] = 0.5  # Space between legend entries
    plt.rcParams['legend.handlelength'] = 2.0  # Length of legend lines
    plt.rcParams['legend.handleheight'] = 0.7
    plt.rcParams['legend.handletextpad'] = 0.8  # Space between line and text
    plt.rcParams['legend.columnspacing'] = 1.0

    # Line and marker settings - more prominent
    plt.rcParams['lines.linewidth'] = 3.0  # Increased from 2.5
    plt.rcParams['lines.markersize'] = 10  # Increased from 8
    plt.rcParams['lines.markeredgewidth'] = 1.5  # Thicker marker edges

    # Grid settings - subtle but visible
    plt.rcParams['grid.alpha'] = 0.4  # Increased from 0.3
    plt.rcParams['grid.linewidth'] = 0.9  # Increased from 0.8
    plt.rcParams['grid.linestyle'] = '--'  # Dashed for distinction

    # Figure settings
    plt.rcParams['figure.titlesize'] = 20  # Increased from 18
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['figure.dpi'] = 300  # Higher resolution
    plt.rcParams['savefig.dpi'] = 300  # Higher resolution for saved figures
    plt.rcParams['savefig.bbox'] = 'tight'  # Ensure nothing is cut off
    plt.rcParams['savefig.pad_inches'] = 0.1

    # Misc settings
    plt.rcParams.update({'figure.max_open_warning': 0})


def save_plots(file_name, subfolder=None,
               extensions=None, copy_to_paper=False,
               paper_location=None):
    if extensions is None or "pdf" not in extensions:
        extensions = ['pdf', 'png']

    # Get project root directory (parent of analysis folder)
    project_root = Path(__file__).resolve().parent.parent
    plots_dir = project_root / 'plots'

    for extension in extensions:
        file_name_new = file_name + '.' + extension
        print("new file name: ", file_name_new)

        if subfolder is None:
            src_file_path = plots_dir / extension / file_name_new
        else:
            src_file_path = plots_dir / extension / subfolder / file_name_new

        # Create directory if it doesn't exist
        src_file_path.parent.mkdir(parents=True, exist_ok=True)

        print(str(src_file_path))
        plt.tight_layout()
        plt.savefig(str(src_file_path), bbox_inches='tight')

        if extension == "pdf" and copy_to_paper:
            print("copying to paper location: ", paper_location)
            dest_path = Path(paper_location) / 'plots' / subfolder
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src_file_path), str(dest_path / file_name_new))
