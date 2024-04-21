import adbutils
import scrcpy

from config import *


class Device:
    frames = []
    state_frames = STATE_FRAMES

    def click(self, node_idx: int):
        """

        :param node_idx: 0-6
        :return:
        """
        pass

    def click_done(self, node1: int, node2: int):
        pass

    def start(self):
        pass

    def ready(self, dev):
        pass


class AdbDevice(Device):

    def __init__(self, host='127.0.0.1', port=21503):
        self.adb = adbutils.adb
        self.conn_str = f'{host}:{port}'
        self.device = self.adb.device()
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

    def start(self):
        self.scrcpy.start()

    def on_frame(self, f):
        if f is not None:
            self.cnt += 1
            self.frames.append(f)
        if len(self.frames) == self.state_frames:
            self.ready(self.frames)
            self.frames = []
