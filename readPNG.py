from PIL import Image


def open_image(path):
    new_image = Image.open(path)
    return new_image


def get_pixel(image, i_index, j_index):
    width, height = image.size
    if i_index > width or j_index > height:
        return None

    pixel = image.getpixel((i_index, j_index))
    return pixel


def convert_image_to_array(path):
    w, h = 28, 28
    arr = [[0 for x in range(w)] for y in range(h)]
    imported_image = open_image(path)
    imported_image = imported_image.convert('1')
    for i in range(0, 28):
        for j in range(0, 28):
            if get_pixel(imported_image, j, i) == 255:
                arr[i][j] = 1
            else:
                arr[i][j] = 0
    new_arr = sum(arr, [])
    return new_arr
