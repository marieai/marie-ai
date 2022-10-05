import matplotlib.pyplot as plt


# helper function for data visualization
def visualize(imgpath=None, **images):
    """PLot images in one row."""

    if True:
        return

    n = len(images)
    # Subplots are organized in a Rows x Cols Grid
    number_of_subplots = n
    number_of_columns = 2

    total = number_of_subplots
    cols = number_of_columns

    # Compute Rows required
    rows = total // cols
    rows += total % cols

    # Create a Position index
    position = range(1, total + 1)

    # Create main figure
    fig = plt.figure(1)
    for k, (name, image) in enumerate(images.items()):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.set_title(" ".join(name.split("_")).title())
        ax.imshow(image)

    if imgpath:
        plt.savefig(imgpath, dpi=600)

    plt.show()
