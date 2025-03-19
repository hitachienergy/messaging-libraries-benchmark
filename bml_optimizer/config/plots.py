lw = 5.04

PLOT_SETTINGS = {
    'normal': {
        'dpi': 300,
        'figsize_bar': (10, 6),
        'figsize_scatter': (12, 6),
        'figsize_optimal': (10, 6),
        'figsize_gp': [(16, 8)],
        'format': 'png',
        'output_dir': 'figures-normal'
    },
    'paper': {
        'dpi': 600,
        'figsize_bar': (lw, lw*0.8),
        'figsize_scatter': (lw*2, lw),
        'figsize_optimal': (lw, lw*0.8),
        'figsize_gp': [(lw, lw), (lw*2, lw*1.2)],
        'format': 'pdf',
        'output_dir': 'figures-paper'
    }
}