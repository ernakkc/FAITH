def UAP_renk(pixel):
    red = (pixel[2] > 100) and (pixel[0] < 50) and (pixel[1] < 50)
    white = (pixel[0] > 200) and (pixel[1] > 200) and (pixel[2] > 200)
    return red or white

def UAI_renk(pixel):
    blue = (pixel[0] > 100) and (pixel[2] < 50) and (pixel[1] < 50)
    white = (pixel[0] > 200) and (pixel[1] > 200) and (pixel[2] > 200)
    return blue or white