# Vocabulary


# Color class
BLUE = "Blue"
GREEN = "Green"
YELLOW = "Yellow"
ORANGE = "Orange"
PURPLE = "Purple"
BLACK = "Black"
""" TEST """
WOOD = "Wood"
""" END TEST  """

# Block Info
NAME = 0
COO_X = 1
COO_Y = 2
WIDTH = 3
HEIGHT = 4
BOX = 5
INTRACLASS = 6
UNICITY = 7
UNIQUE = "Unique"
COMMON = "Common"
SLOTNUMBER = 8
IMAGE = 9
PIN_1 = 10
PIN_2 = 11
BRAILLE = 12
SIZEBLOCK = 13


# Intra color class
RATIO = 3.262295  # 1.634 # 1.295620  # 0.65346534
# BLUE
ASK = "Bloc demander"                 # x1    # ok
ASKw, ASKh = 223*RATIO, 28*RATIO
SAY = "Bloc dire"                 # x11
SAYw, SAYh = 204*RATIO, 33*RATIO
# GREEN
SHOW = "Show"               # x1    # ok
SHOWw, SHOWh = 0, 0                                                                                 # not implemented
BUTTON = "Button"           # x2
BUTTONw, BUTTONh = 174*RATIO, 30*RATIO
PIN = "Bloc broche"                 # x3
PINw, PINh = 183*RATIO, 37*RATIO
SLOTPIN = 1
SHAKED = "Shaked"
SHAKEDw, SHAKEDh = 108*RATIO, 30*RATIO
TILTED = "Tilted"                           # x4
TILTEDw, TILTEDh = 142*RATIO, 31*RATIO
# YELLOW
IF = "If"                                                                       # DIMENTIONS EQUIVALENT TO IF BLOCK
IFw, IFh = 198*RATIO, 32*RATIO
SLOTIF = 1
ELSE = "Else"                                                                   # DIMENTIONS EQUIVALENT TO IF BLOCK
ELSEw, ELSEh = 197*RATIO, 32*RATIO
ENDIF = "EndIf"                                                                 # DIMENTIONS EQUIVALENT TO IF BLOCK
ENDIFw, ENDIFh = 197*RATIO, 32*RATIO
SENDMSG = "Sendmsg"
SENDMSGw, SENDMSGh = 154*RATIO, 32*RATIO
RECMSG = "Recmsg"
RECMSGw, RECMSGh = 190*RATIO, 33*RATIO
WHILE = "While"                                                                 # DIMENTIONS EQUIVALENT WHILE IF BLOCK
WHILEw, WHILEh = 206*RATIO, 32*RATIO
ENDWHILE = "EndWhile"                                                           # DIMENTIONS EQUIVALENT WHILE IF BLOCK
ENDWHILEw, ENDWHILEh = 207*RATIO, 32*RATIO
TOUCH = "Touch"                             # x3
TOUCHw, TOUCHh = 220*RATIO, 33*RATIO
# ORANGE                            # Class OK
VAR = "Var"                 # x1    # ok
VARw, VARh = 252*RATIO, 33*RATIO
ALEA = "Bloc Nombre aleatoire"               # x1    # ok
ALEAw, ALEAh = 303*RATIO, 32*RATIO
SLOTALEA = 2
# PURPLE
PLAYUNTIL = "Playuntil"     # x3
PLAYUNTILw, PLAYUNTILh = 274*RATIO, 32*RATIO
SLOTPLAYUNTIL = 2
PLAY = "Bloc Jouer"               # x5
PLAYw, PLAYh = 146*RATIO, 32*RATIO  # 172*RATIO, 32*RATIO  # 146x32 with old set of blocks


# Seuil d'erreur dûe à la perspective
ERRw = 15
ERRh = 8
# For blocks witout slots
SLOTDEFAULT = 0
# Maximum height for all blocks
MAXHEIGHT = 50 * RATIO


# CUBARITHM
CUBEVALUE = {
    "001110": 0,
    "000001": 1,
    "000101": 2,
    "000011": 3,
    "001011": 4,
    "001001": 5,
    "000111": 6,
    "001111": 7,
    "001101": 8,
    "000110": 9,
    "101000": 2,  # reverted
    "110000": 3  # reverted
    }

UNKNOWN = "unk"
NONE = "None"

COUNT_MODULO = 8

FRMWIDTH = 640
FRMHEIGHT = 480
COLOR = (255, 0, 255)
