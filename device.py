import adbutils
import scrcpy
from minidevice.maatouch import MaaTouch

from config import *


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
        self.maa = MaaTouch(self.device.serial)
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

    def slide_down(self, idx: int, vice: bool):
        if not self.press[vice]:
            self.press[vice] = True
            self.track += 1
            self.press_track[vice] = self.track
        self.maa.send(f'd {self.press_track[vice]} {293 + 219 * idx} 854 100\nc\n')

    def slide_pos(self, idx: int, vice, flick=False):
        self.maa.send(f'm {self.press_track[vice]} {293 + 219 * idx} {854 - 213 * flick} 100\nw 100\nc\n')

    def slide_up(self, vice: bool):
        if self.press[vice]:
            self.maa.send(f'u {self.press_track[vice]}\nc\n')
        self.press[vice] = False

    def click(self, idx: int, vice: bool):
        if idx == 0:
            return
        if not self.press[vice]:
            self.slide_down(idx, vice)
        self.slide_pos(idx, vice)
        self.slide_up(vice)

    def flick(self, idx: int, vice: bool):
        if idx == 0:
            return
        if not self.press[vice]:
            self.slide_down(idx, vice)
        self.slide_pos(idx, vice, True)
        self.slide_up(vice)

    def slide_clear(self):
        if self.press[0]:
            self.slide_up(False)
        if self.press[1]:
            self.slide_up(True)


if __name__ == '__main__':
    device = AdbDevice()
    while True:
        k = int(input('>'))
        device.flick(k, False)
