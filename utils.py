import numpy as np
import matplotlib.pyplot as plt
import math

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def plot_gray(img, save=None):
    plt.imshow(img, cmap='gray')
    plt.show()

def crop_image_circle(image,center,radius):
    threshold = 3
    xmin = max(0,center[0]-radius-threshold)
    xmax = min(image.shape[0],center[0]+radius+threshold)
    ymin = max(0,center[1]-radius-threshold)
    ymax = min(image.shape[1],center[1]+radius+threshold)
    gray_tmp = np.copy(image)
    for i in range(gray_tmp.shape[0]):
        for j in range(gray_tmp.shape[1]):
            distance_2 = math.sqrt(pow(center[0]-i,2) + pow(center[1]-j,2))
            if distance_2 > radius:
                gray_tmp[i][j] = 255
    return gray_tmp[xmin:xmax,ymin:ymax]

def generate_nail_positions(radius, number, shape, center_nail = False):
    x_center = 0.5*shape[0]
    y_center = 0.5*shape[1]
    x = [x_center + radius*math.cos(n*2*math.pi/number) for n in range(number)]
    y = [y_center + radius*math.sin(n*2*math.pi/number) for n in range(number)]
    res = {}
    for i in range(number):
        res[i] = [int(x[i]),int(y[i])]
    if center_nail:
        res[-1] = [int(x_center),int(y_center)]
    return res

def generate_line_matrix(line,shape, thickness = 0.01):
    line_matrix = np.ones(shape=(shape[0],shape[1]))
    point1 = line[0]
    point2 = line[1]
    if abs(point2[0]-point1[0]) > 0.1:
        a = (point2[1]-point1[1])/(point2[0]-point1[0])
        b = point1[1]-a*point1[0]
        x = [i for i in range(min(point1[0],point2[0]),max(point1[0],point2[0])+1)]
        y = [round(a*i+b) for i in x]
        for x_ in x:
            y_ = round(a*x_+b)
            line_matrix[x_][y_] = thickness
    else:
        if abs(point2[1]-point1[1]) < 0.1:
            print('none')
            return line_matrix
        y = [i for i in range(min(point1[1],point2[1]),max(point1[1],point2[1])+1)]
        for y_ in y:
            line_matrix[point1[0]][y_] = thickness       
    return line_matrix

def add_line(current_idx,next_idx, nail_positions, image, thickness = 0.01):
    line = [nail_positions[current_idx],nail_positions[next_idx]]
    line_matrix = generate_line_matrix(line,image.shape,thickness)
    return np.multiply(image,line_matrix)

def distance_image(image1, image2):
    return np.linalg.norm(image1-image2)

def possible_next_nails(idx, nail_number, threshold = 10, center_nail = False):
    # 0 <= idx <= nail_number-1
    if center_nail:
        nail_number -= 1
    if idx == -1:
        return [i for i in range(nail_number)]
    if (idx>=nail_number):
        print('idx not correct')
        return []
    if ((idx>=threshold) and (idx+threshold<nail_number)):
        left = [i for i in range(1,idx-threshold)]
        right = [i for i in range(idx+threshold+1,nail_number)]
        result = left + right
    elif (idx<=threshold):
        result = [i for i in range(idx+threshold+1,nail_number-threshold+idx)]
    elif (idx>=nail_number-threshold):
        result = [i for i in range(threshold,idx-threshold)]
    if (center_nail):
        result.append(-1)
    return result  


