import cv2


def color_picker(color="red"):
    if color == "red":
        return [255, 0, 102]
    elif color == "lime_green":
        return [50, 205, 50]
    elif color == "light_green":
        return [153, 255, 153]
    elif color == "white":
        return [255, 255, 255]
    elif color == "black":
        return [0, 0, 0]
    elif color == "light_pink":
        return [255, 204, 255]
    elif color == "yellow":
        return [30, 144, 255]
    elif color == "blue":
        return [250, 21, 0]
    elif color == "magenta":
        return [255, 0, 255]
    elif color == "purple":
        return [153, 102, 255]
    elif color == "mediumorchid":
        return [186, 85, 211]
    else:
        raise ValueError("Choice does not exist.")