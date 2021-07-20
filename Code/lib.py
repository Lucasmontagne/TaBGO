# Libraries
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from Code import vocab


# Returns the background image when the camera is 461mm above the scene
def crop_start_image(img):
    return img[int((img.shape[0] * 0.08)):int((img.shape[0] * 0.95)), int((img.shape[1] * 0.24)):int((img.shape[1] *
                                                                                                      0.77))]


# Rescale image by multiplying X and Y coordinates by a scale percentage
def rescale_frame(frame, scale_percent=75, back=1):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgX = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    if back == 1:
        bg = imgX[int((height * 0.08)):int((height * 0.95)),
                  int((width * 0.24)):int((width * 0.77))]
    else:
        bg = imgX
    return bg


# Returns the distance between two points
def dist(pt1, pt2):
    return np.sqrt(((pt1[0] - pt2[0]) ** 2) + (pt1[1] - pt2[1]) ** 2)


# Returns the coordinates of the closest point, the width and the height of a box defined by its vertices
def get_width_and_height(box, image_ratio=100):
    p1 = box[0]
    p2 = box[1]
    p3 = box[2]
    p4 = box[3]
    d12 = dist(p1, p2)
    d13 = dist(p1, p3)
    d14 = dist(p1, p4)
    d23 = dist(p3, p2)
    d24 = dist(p4, p2)
    d34 = dist(p3, p4)
    dst = [(d12, "Dist12"), (d13, "Dist13"), (d14, "Dist14"), (d23, "Dist32"), (d24, "Dist42"), (d34, "Dist34")]
    dst.sort()
    height = ((dst[0][0] + dst[1][0]) / 2) * (image_ratio / 100)
    width = ((dst[2][0] + dst[3][0]) / 2) * (image_ratio / 100)
    return [p1[0], p1[1], width, height]


# Sort the vertices points of the box in the right order (BotR = 0, BotL = 1, TopL = 2, TopR = 3)
def reorderBox(bx):
    myBox = np.empty([4, 2])
    littleBox = np.empty([2, 2])
    nbrMin = 0
    nbrMax = math.inf
    rg = 0
    for k in range(4):
        nbr = bx[k][0] + bx[k][1]
        if nbr > nbrMin:
            nbrMin = nbr
            myBox[0][0] = bx[k][0]
            myBox[0][1] = bx[k][1]
        if nbr < nbrMax:
            nbrMax = nbr
            myBox[2][0] = bx[k][0]
            myBox[2][1] = bx[k][1]
    for k in range(4):
        nbr = bx[k][0] + bx[k][1]
        if nbr != nbrMin and nbr != nbrMax:
            littleBox[rg][0] = bx[k][0]
            littleBox[rg][1] = bx[k][1]
            rg = 1
    if littleBox[0][0] > littleBox[1][0]:
        myBox[1][0] = littleBox[0][0]
        myBox[1][1] = littleBox[0][1]
        myBox[3][0] = littleBox[1][0]
        myBox[3][1] = littleBox[1][1]
    else:
        myBox[1][0] = littleBox[1][0]
        myBox[1][1] = littleBox[1][1]
        myBox[3][0] = littleBox[0][0]
        myBox[3][1] = littleBox[0][1]
    return myBox


def create_box_from_contour(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    for nb, [a, b] in enumerate(box):
        box[nb] = [int(a), int(b)]
    box = reorderBox(box)
    return box


# Creates the list of blocks
def get_info_from_contour(ncontour, name, image_ratio=65, areaMin=1000):
    listOfContoursForOneColor = []
    for _, contour in enumerate(ncontour):
        area = cv2.contourArea(contour)
        if area > areaMin:
            box = create_box_from_contour(contour)
            x, y, w, h = get_width_and_height(box, image_ratio=image_ratio)
            intraclass, unicity, nbSlots = get_type_mid(name, w, h)
            listOfContoursForOneColor.append([name, x, y, w, h, box, intraclass, unicity, nbSlots])
    return listOfContoursForOneColor


# Defines binary mask image from hsvImage in the range of low-upp threshs
def define_mask(hsvImg, low, upp):
    # Brightness between 50 & 700 lum
    kernal = np.ones((3, 3), "uint8")
    mask = cv2.inRange(hsvImg, low, upp)
    mask = cv2.erode(mask, kernal)
    mask = cv2.dilate(mask, kernal)
    mask = cv2.dilate(mask, kernal)
    mask = cv2.dilate(mask, kernal)
    return mask


# Extract an element from the list
def extract(lst, get=vocab.NAME):
    return [item[get] for item in lst]


# Creates the histogram for hsv image
def histoHSV(img):
    seuil = 10000
    hg = [0] * 180
    niv = [i for i in range(180)]
    dim = img.shape

    for i in range(dim[0]):
        for j in range(dim[1]):
            hg[img[i][j][0]] += 1

    for i in range(len(hg)):
        if hg[i] > seuil:
            hg[i] = 0

    plt.bar(niv, hg)
    plt.yticks(hg)
    plt.xlabel('Nombre px')
    plt.ylabel('Niveau Teinte (HSV)')
    plt.show()
    return hg


# Creates the histogram for gray image
def histoGRAY(img):
    hg = [0] * 256
    niv = [i for i in range(256)]
    dim = img.shape

    for i in range(dim[0]):
        for j in range(dim[1]):
            hg[img[i][j]] += 1

    plt.bar(niv, hg)
    plt.yticks(hg)
    plt.xlabel('Nombre px')
    plt.ylabel('Niveau Teinte (GRAY)')
    plt.show()
    return hg


# Defines the histerized mask of a gray image
def manual_threshold(img, threshmin=100, threshmax=200):
    threshed = img.copy()
    dim = img.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if threshmin <= img[i, j] <= threshmax:
                threshed[i, j] = 255
            else:
                threshed[i, j] = 0
    return threshed


# Crops and rotates the image
def get_warped(img, w, h, box):
    width = int(w)
    height = int(h)
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (width, height))


# Isolates each block in diff images
def cropRotatedBox(img, list_blc):
    list_out = []
    for bloc in list_blc:
        warped = get_warped(img, bloc[vocab.WIDTH], bloc[vocab.HEIGHT], bloc[vocab.BOX])
        list_out.append([bloc[0], bloc[1], bloc[2], bloc[3], bloc[4], bloc[5],
                         bloc[6], bloc[7], bloc[8], warped, vocab.NONE, vocab.NONE, 0])
    return list_out


# Returns a masked image in hsv mode
def get_masked(img, hsv, masklow, maskhigh, approx=cv2.CHAIN_APPROX_NONE):
    oneMask = define_mask(hsv, masklow, maskhigh)
    contours, _ = cv2.findContours(oneMask, cv2.RETR_LIST, approx)
    masked = cv2.bitwise_and(img, img, mask=oneMask)
    return masked, contours


# With color, width and height finds the type and parameters of one block
def get_type_mid(col, w, h):
    intraclass = vocab.UNKNOWN
    unicity = vocab.UNKNOWN
    nbslot = vocab.SLOTDEFAULT

    if col == vocab.BLUE:
        if vocab.ASKw - vocab.ERRw < w < vocab.ASKw + vocab.ERRw \
                and vocab.ASKh - vocab.ERRh < h < vocab.ASKh + vocab.ERRh:
            intraclass = vocab.ASK
            unicity = vocab.UNIQUE
        if vocab.SAYw - vocab.ERRw < w < vocab.SAYw + vocab.ERRw \
                and vocab.SAYh - vocab.ERRh < h < vocab.SAYh + vocab.ERRh:
            intraclass = vocab.SAY
            unicity = vocab.COMMON
    if col == vocab.GREEN:
        if vocab.SHOWw - vocab.ERRw < w < vocab.SHOWw + vocab.ERRw \
                and vocab.SHOWh - vocab.ERRh < h < vocab.SHOWh + vocab.ERRh:
            intraclass = vocab.SHOW
            unicity = vocab.UNIQUE
        if vocab.BUTTONw - vocab.ERRw < w < vocab.BUTTONw + vocab.ERRw \
                and vocab.BUTTONh - vocab.ERRh < h < vocab.BUTTONh + vocab.ERRh:
            intraclass = vocab.BUTTON
            unicity = vocab.COMMON
        if vocab.PINw - vocab.ERRw < w < vocab.PINw + vocab.ERRw \
                and vocab.PINh - vocab.ERRh < h < vocab.PINh + vocab.ERRh:
            intraclass = vocab.PIN
            unicity = vocab.UNIQUE
            nbslot = vocab.SLOTPIN
        if vocab.SHAKEDw - vocab.ERRw < w < vocab.SHAKEDw + vocab.ERRw \
                and vocab.SHAKEDh - vocab.ERRh < h < vocab.SHAKEDh + vocab.ERRh:
            intraclass = vocab.SHAKED
            unicity = vocab.UNIQUE
        if vocab.TILTEDw - vocab.ERRw < w < vocab.TILTEDw + vocab.ERRw \
                and vocab.TILTEDh - vocab.ERRh < h < vocab.TILTEDh + vocab.ERRh:
            intraclass = vocab.TILTED
            unicity = vocab.COMMON
    if col == vocab.YELLOW:
        if vocab.ENDIFw - vocab.ERRw < w < vocab.ENDIFw + vocab.ERRw \
                and vocab.ENDIFh - vocab.ERRh < h < vocab.ENDIFh + vocab.ERRh:
            intraclass = vocab.ENDIF
            unicity = vocab.UNIQUE
        if vocab.ELSEw - vocab.ERRw < w < vocab.ELSEw + vocab.ERRw \
                and vocab.ELSEh - vocab.ERRh < h < vocab.ELSEh + vocab.ERRh:
            intraclass = vocab.ELSE
            unicity = vocab.UNIQUE
        if vocab.IFw - vocab.ERRw < w < vocab.IFw + vocab.ERRw \
                and vocab.IFh - vocab.ERRh < h < vocab.IFh + vocab.ERRh:
            intraclass = vocab.IF
            unicity = vocab.UNIQUE
            nbslot = vocab.SLOTIF
        if vocab.SENDMSGw - vocab.ERRw < w < vocab.SENDMSGw + vocab.ERRw \
                and vocab.SENDMSGh - vocab.ERRh < h < vocab.SENDMSGh + vocab.ERRh:
            intraclass = vocab.SENDMSG
            unicity = vocab.UNIQUE
        if vocab.RECMSGw - vocab.ERRw < w < vocab.RECMSGw + vocab.ERRw \
                and vocab.RECMSGh - vocab.ERRh < h < vocab.RECMSGh + vocab.ERRh:
            intraclass = vocab.RECMSG
            unicity = vocab.UNIQUE
        if vocab.ENDWHILEw - vocab.ERRw < w < vocab.ENDWHILEw + vocab.ERRw \
                and vocab.ENDWHILEh - vocab.ERRh < h < vocab.ENDWHILEh + vocab.ERRh:
            intraclass = vocab.ENDWHILE
            unicity = vocab.UNIQUE
        if vocab.WHILEw - vocab.ERRw < w < vocab.WHILEw + vocab.ERRw \
                and vocab.WHILEh - vocab.ERRh < h < vocab.WHILEh + vocab.ERRh:
            intraclass = vocab.WHILE
            unicity = vocab.UNIQUE
        if vocab.TOUCHw - vocab.ERRw < w < vocab.TOUCHw + vocab.ERRw \
                and vocab.TOUCHh - vocab.ERRh < h < vocab.TOUCHh + vocab.ERRh:
            intraclass = vocab.TOUCH
            unicity = vocab.COMMON
    if col == vocab.ORANGE:
        if vocab.VARw - vocab.ERRw < w < vocab.VARw + vocab.ERRw \
                and vocab.VARh - vocab.ERRh < h < vocab.VARh + vocab.ERRh:
            intraclass = vocab.VAR
            unicity = vocab.UNIQUE
        if vocab.ALEAw - vocab.ERRw < w < vocab.ALEAw + vocab.ERRw \
                and vocab.ALEAh - vocab.ERRh < h < vocab.ALEAh + vocab.ERRh:
            intraclass = vocab.ALEA
            unicity = vocab.UNIQUE
            nbslot = vocab.SLOTALEA
    if col == vocab.PURPLE:
        if vocab.PLAYUNTILw - vocab.ERRw < w < vocab.PLAYUNTILw + vocab.ERRw \
                and vocab.PLAYUNTILh - vocab.ERRh < h < vocab.PLAYUNTILh + vocab.ERRh:
            intraclass = vocab.PLAYUNTIL
            unicity = vocab.COMMON
            nbslot = vocab.SLOTPLAYUNTIL
        if vocab.PLAYw - vocab.ERRw < w < vocab.PLAYw + vocab.ERRw \
                and vocab.PLAYh - vocab.ERRh < h < vocab.PLAYh + vocab.ERRh:
            intraclass = vocab.PLAY
            unicity = vocab.COMMON

    return intraclass, unicity, nbslot


# TODO Aggressive filtration to separate consecutive same-colored blocks
# NOT FUNCTIONNING
def separate(listofblocks):
    """
    for _, bloc in enumerate(listofblocks):
        if bloc[vocab.HEIGHT] >= vocab.MAXHEIGHT:
            bigBlock = cv2.Canny(bloc[vocab.IMAGE], 0, 50)
            contours, _ = cv2.findContours(bigBlock, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            horizontal = np.copy(bigBlock)
            cols = horizontal.shape[1]
            horizontal_size = cols // 30
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=5)

            cv2.imshow("Canny" + str(bloc[vocab.NAME]), bigBlock)
            cv2.imshow("Big" + str(bloc[vocab.NAME]) + "Block", horizontal)
            cv2.drawContours(bigBlock, contours, -1, (255, 255, 255))
    """
    return listofblocks


# Extract the values of cubarithms present on each block
def get_cubarithm(listofblocks, blk_low, blk_upp):

    for nbr, bloc in enumerate(listofblocks):

        if bloc[vocab.SLOTNUMBER] > 0:

            out = [[0, 0]] * 50
            nbit = 0
            nbrCub = bloc[vocab.SLOTNUMBER]

            imageHSV = cv2.cvtColor(bloc[vocab.IMAGE], cv2.COLOR_BGR2HSV)
            black_masked, contours = get_masked(bloc[vocab.IMAGE], imageHSV, blk_low, blk_upp,
                                                approx=cv2.CHAIN_APPROX_SIMPLE)
            cv2.imshow("Blk", black_masked)
            for it, cntrs in enumerate(contours):
                areaC = cv2.contourArea(cntrs)
                if areaC > 120:

                    box = create_box_from_contour(cntrs)
                    x, y, wid, hei = get_width_and_height(box)
                    cubarithm = get_warped(bloc[vocab.IMAGE], wid, hei, box)
                    warped = cv2.cvtColor(cubarithm, cv2.COLOR_BGR2GRAY)

                    """identify(warped)"""

                    warped = manual_threshold(warped, 200, 255)
                    h, w = warped.shape[:2]
                    mask = np.zeros((h + 2, w + 2), np.uint8)
                    w, h = warped.shape[1], warped.shape[0]

                    for i in range(w):
                        cv2.floodFill(warped, mask, (i, 0), (0, 0, 0))
                        cv2.floodFill(warped, mask, (i, 1), (0, 0, 0))
                        cv2.floodFill(warped, mask, (i, h - 2), (0, 0, 0))
                        cv2.floodFill(warped, mask, (i, h - 1), (0, 0, 0))
                    for j in range(h):
                        cv2.floodFill(warped, mask, (0, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (1, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (2, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (3, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (4, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (w - 9, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (w - 8, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (w - 7, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (w - 6, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (w - 5, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (w - 4, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (w - 3, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (w - 2, j), (0, 0, 0))
                        cv2.floodFill(warped, mask, (w - 1, j), (0, 0, 0))

                    pourcentage = [0, 0, 0, 0, 0, 0]
                    for i in range(w - 1):
                        for j in range(h - 1):
                            if 0 <= j < h / 3:
                                if 0 <= i < w / 2:
                                    if warped[j, i] > 200:
                                        pourcentage[0] += 1
                                else:
                                    if warped[j, i] > 200:
                                        pourcentage[1] += 1
                            if h / 3 <= j < 2 * h / 3:
                                if 0 <= i < w / 2:
                                    if warped[j, i] > 200:
                                        pourcentage[2] += 1
                                else:
                                    if warped[j, i] > 200:
                                        pourcentage[3] += 1
                            if 2 * h / 3 <= j < h:
                                if 0 <= i < w / 2:
                                    if warped[j, i] > 200:
                                        pourcentage[4] += 1
                                else:
                                    if warped[j, i] > 200:
                                        pourcentage[5] += 1

                    key = ""
                    for val in pourcentage:
                        # print(val)
                        if val > 3:
                            key += "1"
                        else:
                            key += "0"

                    out[it] = [vocab.CUBEVALUE.get(key, vocab.UNKNOWN), x]
                    # print(key)
                    # cv2.imshow(bloc[vocab.NAME] + "_" + str(vocab.CUBEVALUE.get(key, "Inconnu")), warped)
                    nbit += 1

            while [0, 0] in out:
                out.remove([0, 0])
            out = sorted(out, key=lambda l: l[1])

            if nbit == 0:
                out.append([-1, 0])

            if nbrCub == 1:
                listofblocks[nbr][vocab.PIN_1] = out[0][0]

            if nbrCub == 2:
                if nbit != 2:
                    out.append([-1, 0])
                listofblocks[nbr][vocab.PIN_1] = out[0][0]
                listofblocks[nbr][vocab.PIN_2] = out[1][0]

    return listofblocks


# Gets the magnitude of the blobs number on each block in the list
def get_braille(lst, masks):
    """# Set up the SimpleBlobdetector with default parameters
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 30
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.01
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)"""

    for n, bloc in enumerate(lst):
        if bloc[vocab.UNICITY] == vocab.COMMON:
            image_bloc = bloc[vocab.IMAGE]
            col = bloc[vocab.NAME]

            hsvFrame = cv2.cvtColor(image_bloc, cv2.COLOR_BGR2HSV)
            masked, threshed = 0, 0
            kernalMoins = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            if col == vocab.BLUE:
                masked, contours = get_masked(image_bloc, hsvFrame, masks[0], masks[1])
            if col == vocab.GREEN:
                masked, contours = get_masked(image_bloc, hsvFrame, masks[2], masks[3])
            if col == vocab.YELLOW:
                masked, contours = get_masked(image_bloc, hsvFrame, masks[4], masks[5])
            if col == vocab.ORANGE:
                masked, contours = get_masked(image_bloc, hsvFrame, masks[6], masks[7])
            if col == vocab.PURPLE:
                masked, contours = get_masked(image_bloc, hsvFrame, masks[8], masks[9])

            grayFrame = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

            if col == vocab.BLUE:
                threshed = manual_threshold(grayFrame, threshmin=200, threshmax=255)  # OK
            if col == vocab.GREEN:
                threshed = manual_threshold(grayFrame, threshmin=180, threshmax=255)  # N_ok
                threshed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernalMoins, iterations=1)
                threshed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernalMoins, iterations=1)
            if col == vocab.YELLOW:
                threshed = manual_threshold(grayFrame, threshmin=200, threshmax=255)  # N_ok 100-130 wood
            if col == vocab.ORANGE:
                threshed = manual_threshold(grayFrame, threshmin=200, threshmax=250)  # N_ok
            if col == vocab.PURPLE:
                threshed = manual_threshold(grayFrame, threshmin=190, threshmax=255)  # Q_ok 170-255
                threshed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernalMoins, iterations=1)
                threshed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernalMoins, iterations=1)

            contours, _ = cv2.findContours(threshed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            listblobs = []
            for _, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < 50:
                    if area > 3:
                        listblobs.append(cnt)

            # cv2.drawContours(image_bloc, listblobs, -1, (0, 0, 255))
            # cv2.imshow("MyBloc_" + str(col), image_bloc)
            # cv2.imshow("Threshed_" + col, threshed)
            lst[n][vocab.BRAILLE] = len(listblobs)

    return lst


# Returns the most frequent item in a list
def most_frequent(lst):
    return max(set(lst), key=lst.count)


# Reorders the list to find the most frequent element in each branch
# DON'T CHANGE THE LIST SIZE DURING THE COMPUTATION
def average(lst):

    size_tot = len(lst)
    size_algo = len(lst[0])
    size_block = vocab.SIZEBLOCK

    lst_occ = []

    for j in range(size_algo):
        lst_occ.append([])
        for k in range(size_block):
            lst_occ[j].append([])
            for i in range(size_tot):
                lst_occ[j][k].append(lst[i][j][k])
    print("")
    return lst_occ


# Not functionning on this kind of pictures - patterns too close
def identify():  # imgGray):
    """
    imgGray = rescale_frame(imgGray, scale_percent=300, back=0)

    # Import images to compare
    im1 = cv2.imread("Un.jpg", cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread("Deux.jpg", cv2.IMREAD_GRAYSCALE)
    im3 = cv2.imread("Trois.jpg", cv2.IMREAD_GRAYSCALE)
    im4 = cv2.imread("Quatre.jpg", cv2.IMREAD_GRAYSCALE)
    im5 = cv2.imread("Cinq.jpg", cv2.IMREAD_GRAYSCALE)
    im6 = cv2.imread("Six.jpg", cv2.IMREAD_GRAYSCALE)
    im7 = cv2.imread("Sept.jpg", cv2.IMREAD_GRAYSCALE)
    im8 = cv2.imread("Huit.jpg", cv2.IMREAD_GRAYSCALE)
    im9 = cv2.imread("Neuf.jpg", cv2.IMREAD_GRAYSCALE)

    # Create descriptors for each frame & images
    sift = cv2.SIFT_create()
    kpGray, descGray = sift.detectAndCompute(imgGray, None)
    kp_im1, desc_im1 = sift.detectAndCompute(im1, None)
    kp_im2, desc_im2 = sift.detectAndCompute(im2, None)
    kp_im3, desc_im3 = sift.detectAndCompute(im3, None)
    kp_im4, desc_im4 = sift.detectAndCompute(im4, None)
    kp_im5, desc_im5 = sift.detectAndCompute(im5, None)
    kp_im6, desc_im6 = sift.detectAndCompute(im6, None)
    kp_im7, desc_im7 = sift.detectAndCompute(im7, None)
    kp_im8, desc_im8 = sift.detectAndCompute(im8, None)
    kp_im9, desc_im9 = sift.detectAndCompute(im9, None)

    # Match descriptors
    bf = cv2.BFMatcher()
    match1 = bf.knnMatch(descGray, desc_im1, k=2)
    match2 = bf.knnMatch(descGray, desc_im2, k=2)
    match3 = bf.knnMatch(descGray, desc_im3, k=2)
    """
    """
    strg = "\nAppariements : "

    good = []
    for m, n in match1:
        if m.distance < 0.9 * n.distance:
            good.append([m])
    img1 = cv2.drawMatchesKnn(imgGray, kpGray, im1, kp_im1, good, None, flags=2)
    strg += str(len(good)) + " / "

    good = []
    for m, n in match2:
        if m.distance < 0.9 * n.distance:
            good.append([m])
    img2 = cv2.drawMatchesKnn(imgGray, kpGray, im2, kp_im2, good, None, flags=2)
    strg += str(len(good)) + " / "

    good = []
    for m, n in match3:
        if m.distance < 0.9 * n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(imgGray, kpGray, im3, kp_im3, good, None, flags=2)
    strg += str(len(good)) + "."
    """
    """
    cv2.imshow("Match1", img1)
    cv2.imshow("Match2", img2)
    cv2.imshow("Match3", img3)
    print(strg)
    ETC """
