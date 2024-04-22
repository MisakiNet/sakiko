import adbutils
import scrcpy

from config import *

EVENT_DEV = '/dev/input/event4'

EV_SYN = 0x00
EV_KEY = 0x01
EV_ABS = 0x03

PRESS_DOWN = 1
PRESS_UP = 0

ABS_MT_TRACKING_ID = 57
TRACKING_END = 0xFFFFFFFF
ABS_MT_POSITION_X = 53
ABS_MT_POSITION_Y = 54
ABS_MT_SLOT = 47
BTN_TOUCH = 330
BTN_TOOL_FINGER = 325

SYN_REPORT = 0


class Device:
    frames = []
    state_frames = STATE_FRAMES

    def click(self, idx: int, vice: bool):
        pass

    def flick(self, idx: int, vice: bool):
        pass

    def key_down(self, idx: int, vice: bool):
        pass

    def start(self):
        pass

    def ready(self):
        pass


class AdbDevice(Device):

    def __init__(self, host='127.0.0.1', port=21503):
        self.adb = adbutils.adb
        self.conn_str = f'{host}:{port}'
        self.device = self.adb.device()
        self.press = [False, False]
        self.track = 0
        self.press_track = [0, 0]
        if not self.device:
            res = self.adb.connect(self.conn_str)
            print(f'[ADB] {res}')
            if 'already' not in res:
                raise '[ADB] connect failed'
            self.device = self.adb.device()
        print(f'[ADB] Device: {self.device}')
        self.scrcpy = scrcpy.Client(device=self.device)
        self.scrcpy.add_listener(scrcpy.EVENT_FRAME, lambda event: self.on_frame(event))
        self.cnt = 0

    def start(self, threaded=False):
        self.scrcpy.start(threaded)

    def on_frame(self, f):
        if f is not None:
            self.cnt += 1
            self.frames.append(f)
        if len(self.frames) == self.state_frames:
            self.ready()
            self.frames = []

    def send_event(self, typ, code, value):
        self.device.shell(['sendevent', EVENT_DEV, hex(typ), hex(code), hex(value)])

    def slide_down(self, idx: int, vice: bool):
        self.send_event(EV_ABS, ABS_MT_SLOT, vice)
        if not self.press[vice]:
            self.press[vice] = True
            self.track += 1
            self.press_track[vice] = self.track
            self.send_event(EV_ABS, ABS_MT_TRACKING_ID, self.press_track[vice])
            self.send_event(EV_KEY, BTN_TOUCH, PRESS_DOWN)
            self.send_event(EV_KEY, BTN_TOOL_FINGER, PRESS_DOWN)
        else:
            self.send_event(EV_ABS, ABS_MT_TRACKING_ID, self.press_track[vice])
        self.send_event(EV_ABS, ABS_MT_POSITION_X, 293 + 219 * idx)
        self.send_event(EV_ABS, ABS_MT_POSITION_Y, 854)
        self.send_event(EV_SYN, SYN_REPORT, 0)

    def slide_pos(self, idx: int, vice, flick=False):
        self.send_event(EV_ABS, ABS_MT_SLOT, vice)
        self.send_event(EV_ABS, ABS_MT_TRACKING_ID, self.press_track[vice])
        self.send_event(EV_ABS, ABS_MT_POSITION_X, 293 + 219 * idx)
        self.send_event(EV_ABS, ABS_MT_POSITION_Y, 854 - 214 * flick)
        self.send_event(EV_SYN, SYN_REPORT, 0)

    def slide_up(self, vice: bool):
        if self.press[vice]:
            self.send_event(EV_ABS, ABS_MT_SLOT, vice)
            self.send_event(EV_ABS, ABS_MT_TRACKING_ID, TRACKING_END)
            self.send_event(EV_KEY, BTN_TOUCH, PRESS_UP)
            self.send_event(EV_KEY, BTN_TOOL_FINGER, PRESS_UP)
            self.send_event(EV_SYN, SYN_REPORT, 0)
        self.press[vice] = False

    def click(self, idx: int, vice: bool):
        print(f'Click {idx} {vice}')
        if idx == 0:
            return
        x = 293 + 219 * idx
        if self.press[vice]:
            self.slide_pos(idx, vice)
            self.slide_up(vice)
        else:
            self.device.click(x, 854)

    def flick(self, idx: int, vice: bool):
        print(f'Flick {idx} {vice}')
        if idx == 0:
            return
        x = 293 + 219 * idx
        if self.press[vice]:
            self.slide_pos(idx, vice, True)
            self.slide_up(vice)
        else:
            self.device.swipe(x, 854, x, 640, 100)

    def slide_clear(self):
        if self.press[0]:
            self.slide_up(False)
        if self.press[1]:
            self.slide_up(True)


if __name__ == '__main__':
    device = AdbDevice()
    while True:
        k = int(input('>'))
        device.click(k, False)
