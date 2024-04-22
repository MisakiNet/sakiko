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

    def send(self, l1, l2, r1, r2):
        cmd = []
        # group2: 0: click, 1: down, 2: flick
        if l1:
            cmd.append(f"{'m' if self.press[0] else 'd'} 0 {293 + 219 * l1} 854 100")
            if l2 == 2:
                cmd.append(f'm 0 {293 + 219 * l1} 641 100')
            if l2 != 1:
                cmd.append('u 0')
            self.press[0] = l2 == 1
        if l1 != r1 and r1:
            cmd.append(f"{'m' if self.press[1] else 'd'} 1 {293 + 219 * r1} 854 100")
            if r2 == 2:
                cmd.append(f'm 1 {293 + 219 * r1} 641 100')
            if r2 != 1:
                cmd.append('u 1')
            self.press[1] = r2 == 1
        cmd = '\n'.join(cmd) + '\nc\n'
        self.maa.send(cmd)

    def clear(self):
        cmd = ''
        if self.press[0]:
            cmd += 'u 0\n'
            self.press[0] = False
        if self.press[1]:
            cmd += 'u 1\n'
            self.press[1] = False
        if cmd:
            self.maa.send(cmd + 'c\n')


if __name__ == '__main__':
    device = AdbDevice()
    while True:
        k = int(input('>'))
        device.send(k, 2, k + 1, 2)
