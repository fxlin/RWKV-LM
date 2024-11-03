###############  touch device #####################
# cf: https://www.waveshare.com/wiki/2.13inch_Touch_e-Paper_HAT_Manual#Touch_Driver (for C) 

# GT_Development -- stores information about the current touch points
import threading
import logging

from TP_lib import gt1151, epd2in13_V4

flag_t = 1   # flag: if gt.INT is high, indicates a touch event

# touch dev polling thread, set a flag showing if a touch even has occurred
# xzl: NB: GT_Dev is a global obj. ::Touch is a flag set by this thread
# below polling???
# ::Touch will be examined by class code of GT1151::GT_scan()
def pthread_irq() :
    print("pthread running")    
    while flag_t == 1 :
    # xzl: non blocking? inefficient...     
        if(gt.digital_read(gt.INT) == 0) :    
            GT_Dev.Touch = 1
        else :
            GT_Dev.Touch = 0
    print("thread:exit")
    
###############  start of mini touch ex #####################
try: 
    epd = epd2in13_V4.EPD()
    gt = gt1151.GT1151()
    GT_Dev = gt1151.GT_Development()
    GT_Old = gt1151.GT_Development()

    epd.init(epd.FULL_UPDATE)   # must do this, otherwise, touch won't work (io error)
    gt.GT_Init()

    # touch dev polling thread
    t = threading.Thread(target = pthread_irq)
    t.setDaemon(True)
    t.start()

    while (1):
        gt.GT_Scan(GT_Dev, GT_Old)
        # dedup, avoid exposing repeated events to app
        if(GT_Old.X[0] == GT_Dev.X[0] and GT_Old.Y[0] == GT_Dev.Y[0] and GT_Old.S[0] == GT_Dev.S[0]):
            continue

        # meaning touch event ready to be read out
        if(GT_Dev.TouchpointFlag):
            GT_Dev.TouchpointFlag = 0
            print(f"touch ev GT_Dev.X[0]: {GT_Dev.X[0]}, GT_Dev.Y[0]: {GT_Dev.Y[0]}, GT_Dev.S[0]: {GT_Dev.S[0]}")

except IOError as e:
    print("io error")
    logging.info(e)

except KeyboardInterrupt:    
    logging.info("ctrl + c:")
    exit()