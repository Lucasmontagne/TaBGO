# Libraries
import cv2
import numpy as np
from Code import lib
from Code import vocab
import sys

# Capturing video through webcam BRIO
webcam = cv2.VideoCapture(cv2.CAP_DSHOW)

# Webcam parameters
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # -1280
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # -720
webcam.set(cv2.CAP_PROP_FPS, 20)  # 5, 15, 30, 60
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
webcam.set(cv2.CAP_PROP_FOURCC, fourcc)
webcam.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # default 128
webcam.set(cv2.CAP_PROP_CONTRAST, 130)  # default 100 good 130
webcam.set(cv2.CAP_PROP_SATURATION, 90)  # default 128 good 90
webcam.set(cv2.CAP_PROP_GAIN, 0)  # default 128
webcam.set(cv2.CAP_PROP_EXPOSURE, -4)  # default -5
webcam.set(cv2.CAP_PROP_FOCUS, 15)  # default 10 15

# Opening the video settings GUI
webcam.set(cv2.CAP_PROP_SETTINGS, 0)

# Set range for blue mask color
blue_lower = np.array([94, 50, 50], np.uint8)
blue_upper = np.array([110, 255, 255], np.uint8)
# Set range for green mask color
green_lower = np.array([30, 30, 50], np.uint8)
green_upper = np.array([94, 255, 255], np.uint8)
# Set range for yellow mask color
yellow_lower = np.array([20, 60, 50], np.uint8)  # 20 60 50
yellow_upper = np.array([25, 255, 255], np.uint8)  # 25 255 255
# Set range for orange mask color
orange_lower = np.array([1, 50, 50], np.uint8)  # 1 50 50
orange_upper = np.array([10, 255, 255], np.uint8)  # 12 255 255
# Set range for purple mask color
purple_lower = np.array([114, 30, 20], np.uint8)
purple_upper = np.array([140, 255, 255], np.uint8)
# Set range for black mask color
black_lower = np.array([0, 0, 0], np.uint8)
black_upper = np.array([180, 90, 100], np.uint8)

# Regroup all masks together
all_masks = [blue_lower, blue_upper, green_lower, green_upper, yellow_lower, yellow_upper, orange_lower, orange_upper,
             purple_lower, purple_upper]

# List for reapeted measures
lst_memo = []
end = []

# Number of different measures
repeat = 1000
cpt_rep = 0


# Main loop
while cpt_rep < repeat:

    # Acquire frames from camera and isolate the background
    _, imageFrame = webcam.read()
    imageFrame = lib.crop_start_image(imageFrame)

    ###################################################################################################################
    # Color detection zone

    # Convert every frame in HSV env. from BGR to HSV (color angle, saturation, brightness)
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # See color lvls. for masks config.
    """
    print(lib.histo(hsvFrame))
    """

    # Define masks for each color
    blue_mask = lib.define_mask(hsvFrame, blue_lower, blue_upper)
    green_mask = lib.define_mask(hsvFrame, green_lower, green_upper)
    yellow_mask = lib.define_mask(hsvFrame, yellow_lower, yellow_upper)
    orange_mask = lib.define_mask(hsvFrame, orange_lower, orange_upper)
    purple_mask = lib.define_mask(hsvFrame, purple_lower, purple_upper)

    # Find contours for each color
    contoursB, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursG, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursY, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursO, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursP, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get the list of blocks
    # Each block is represented by
    # [ColorName, CX_1stPt, CY_1stPt, Width, Height, Box(ArrayOfVertices), Name, Unicity, nbCubarithms, P1, P2, Braille]
    listOfBlocks = []
    listOfBlocks = listOfBlocks + lib.get_info_from_contour(contoursB, vocab.BLUE)
    listOfBlocks = listOfBlocks + lib.get_info_from_contour(contoursG, vocab.GREEN)
    listOfBlocks = listOfBlocks + lib.get_info_from_contour(contoursY, vocab.YELLOW)
    listOfBlocks = listOfBlocks + lib.get_info_from_contour(contoursO, vocab.ORANGE)
    listOfBlocks = listOfBlocks + lib.get_info_from_contour(contoursP, vocab.PURPLE)

    ###################################################################################################################
    # Separation and sorting zone

    end_list = []
    if cpt_rep > 10:
    
        # Reorder blocks (from bottom to top because the camera is inverted above the scene)
        listOfBlocks = sorted(listOfBlocks, key=lambda x: x[2])
        listOfBlocks.reverse()

        # Isolate the colored blocks
        list_img_blocks = lib.cropRotatedBox(imageFrame, listOfBlocks)

        # Separate consecutive same-colored blocks
        new_list_img = lib.separate(list_img_blocks)  # todo

        ################################################################################################################

        # Braille detection zone
        # Acquire the values on cubarithm
        last_list_img = lib.get_cubarithm(new_list_img, black_lower, black_upper)

        # Extract the number of blobs for each block
        end_list = lib.get_braille(list_img_blocks, all_masks)
        
        ################################################################################################################

        # Show relevant info
        """print("\n" + str(lib.extract(last_list_img, vocab.NAME)) + "\n"
              # + str(lib.extract(list_img_blocks, vocab.WIDTH)) + "\n"
              # + str(lib.extract(list_img_blocks, vocab.HEIGHT)) + "\n"
              # + str(lib.extract(last_list_img, vocab.INTRACLASS)) + "\n"
              # + str(lib.extract(last_list_img, vocab.UNICITY)) + "\n"
              # + str(lib.extract(last_list_img, vocab.SLOTNUMBER)) + "\n"
              # + str(lib.extract(last_list_img, vocab.PIN_1)) + "\n"
              # + str(lib.extract(last_list_img, vocab.PIN_2)) + "\n"
              )"""

    # Average results
    if cpt_rep % vocab.COUNT_MODULO != 0:
        myStr = "[" + "#" * ((cpt_rep - 1) % vocab.COUNT_MODULO) + " " * (
                    (vocab.COUNT_MODULO-2) - ((cpt_rep - 1) % vocab.COUNT_MODULO)) + "]"
        if cpt_rep <= 10:
            sys.stdout.write("\r" + myStr + " INITIALISATION CAMERA")
        if cpt_rep > 10:
            lst_memo.append(end_list)
            sys.stdout.write("\r" + myStr + " CHARGEMENT")
    else:
        if cpt_rep <= 10:
            lst_memo.clear()
        if cpt_rep > 10:
            end = lib.average(lst_memo)
            lst_memo.clear()
    cpt_rep += 1

    # Display colored filters
    """cv2.namedWindow("p", cv2.WINDOW_NORMAL)
    cv2.imshow("p", purple_mask)
    cv2.namedWindow("y", cv2.WINDOW_NORMAL)
    cv2.imshow("y", yellow_mask)
    cv2.namedWindow("b", cv2.WINDOW_NORMAL)
    cv2.imshow("b", blue_mask)
    cv2.namedWindow("g", cv2.WINDOW_NORMAL)
    cv2.imshow("g", green_mask)
    cv2.namedWindow("o", cv2.WINDOW_NORMAL)
    cv2.imshow("o", orange_mask)"""

    # Results on original image
    """for bloc in end_list:
        strbloc = bloc[vocab.INTRACLASS]
        if bloc[vocab.SLOTNUMBER] == 1:
            strbloc += " avec pin " + str(bloc[vocab.PIN_1])
        if bloc[vocab.SLOTNUMBER] == 2:
            strbloc += " avec pin " + str(bloc[vocab.PIN_1]) + " et " + str(bloc[vocab.PIN_2])
        if bloc[vocab.UNICITY] == vocab.COMMON:
            strbloc += ", on compte " + str(bloc[vocab.BRAILLE]) + " blobs."
        cv2.putText(imageFrame, strbloc, (int(bloc[vocab.BOX][2][0]), int(bloc[vocab.BOX][2][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)"""
    """cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.imshow("original", imageFrame)"""

    # Stops the program properly by pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
