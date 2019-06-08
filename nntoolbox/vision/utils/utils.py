def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2)) / (c * h * w)

def is_image(filename):
    return filename.endswith(".jpg") or filename.endswith(".png")