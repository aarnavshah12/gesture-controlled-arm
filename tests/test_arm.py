"""GestureArm over the block picker's driver, with a fake serial port. Skipped if the block picker is absent."""
import os
import struct
import tempfile
import unittest

from _common import silent_logger

from gesture import bp, config, runlog

config.LOG_DIR = os.path.join(tempfile.gettempdir(), "gesture-arm-tests")
runlog.start_run("test-arm", quiet=True)
HAVE_BP = not [p for p in bp.check() if "calibration" not in p]
if HAVE_BP:
    from gesture import arm as garm


class FakeSerial:
    """Answers FUNC_READ_XYZ with a fixed position; records everything written."""

    def __init__(self, xyz=(0, -175, 150)):
        self.written = []
        self.xyz = xyz
        self._rx = b""
        self.in_waiting = 0

    def write(self, b):
        self.written.append(bytes(b))
        if b[2] == 0x13:
            body = bytes([0x13, 6]) + struct.pack("<hhh", *self.xyz)
            frame = b"\xAA\x55" + body
            self._rx = frame + bytes([garm._bp.arm.checksum(frame)])

    def flush(self):
        pass

    def reset_input_buffer(self):
        self._rx = b""

    def read(self, n):
        out, self._rx = self._rx[:n], self._rx[n:]
        return out


@unittest.skipUnless(HAVE_BP, "block-picker project not present")
class GestureArmTest(unittest.TestCase):
    def live_arm(self, xyz=(0, -175, 150)):
        a = garm.GestureArm(dry_run=False, port="/dev/null-fake")
        a._ser = FakeSerial(xyz)
        a._cleared = True
        a.glog = silent_logger()
        a.log = silent_logger("bp-test")
        return a

    def test_mirror_box_inside_reach(self):
        box, reach = garm.mirror_box(), garm.reach_box()
        for (lo, hi), (rlo, rhi) in zip(box, reach):
            self.assertGreaterEqual(lo, rlo)
            self.assertLessEqual(hi, rhi)
        self.assertGreaterEqual(box[2][0], garm._bp.config.TABLE_Z_MM)

    def test_dry_run_sends_nothing(self):
        a = garm.GestureArm(dry_run=True)
        a.glog = silent_logger()
        a.connect()
        a.stream_to(0, -175, 150)
        self.assertEqual(a.commanded, (0.0, -175.0, 150.0))
        self.assertIsNone(a._ser)
        a.halt()
        a.grip()
        a.release()

    def test_stream_to_sends_one_set_xyz_frame(self):
        a = self.live_arm()
        a.stream_to(10, -180, 140)
        self.assertEqual(a._ser.written, [garm._bp.arm.set_xyz_frame(10, -180, 140, config.STREAM_MOVE_MS)])
        self.assertEqual(a.commanded, (10.0, -180.0, 140.0))

    def test_stream_to_refuses_outside_reach_and_below_table(self):
        a = self.live_arm()
        (xlo, xhi), _, (zlo, zhi) = garm.reach_box()
        for bad in ((xhi + 50, -175, 150), (0, -175, garm._bp.config.TABLE_Z_MM - 1), (0, -30, 150)):
            with self.assertRaises(garm.UnsafeTarget):
                a.stream_to(*bad)
        self.assertEqual(a._ser.written, [])
        self.assertIsNone(a.commanded)

    def test_stream_to_refuses_when_not_cleared(self):
        a = self.live_arm()
        a._cleared = False
        with self.assertRaises(garm.ArmError):
            a.stream_to(0, -175, 150)
        self.assertEqual(a._ser.written, [])

    def test_halt_recommands_readback_position(self):
        a = self.live_arm(xyz=(20, -190, 130))
        a.halt()
        frames = [f for f in a._ser.written if f[2] == 0x03]
        self.assertEqual(frames, [garm._bp.arm.set_xyz_frame(20, -190, 130, config.HALT_MOVE_MS)])

    def test_halt_clamps_noisy_readback_into_reach(self):
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = garm.reach_box()
        a = self.live_arm(xyz=(int(xhi) + 5, int(yhi) + 3, int(zlo) - 4))
        a.halt()
        frames = [f for f in a._ser.written if f[2] == 0x03]
        self.assertEqual(len(frames), 1)
        x, y, z, ms = struct.unpack("<hhhH", frames[0][4:12])
        self.assertEqual((x, y), (int(xhi), int(yhi)))
        self.assertGreaterEqual(z, garm._bp.config.TABLE_Z_MM)

    def test_halt_inhibits_motion_until_released(self):
        a = self.live_arm(xyz=(0, -175, 150))
        a.halt()
        self.assertTrue(a.inhibited)
        with self.assertRaises(garm.ArmError):
            a.stream_to(5, -175, 150)
        with self.assertRaises(garm.ArmError):
            a.move_to(5, -175, 150, 500)
        moves = [f for f in a._ser.written if f[2] == 0x03]
        self.assertEqual(len(moves), 1)                 # only the halt frame went out
        a.suction(True)                                 # nozzle frames are still allowed (release from FROZEN)
        self.assertEqual([f for f in a._ser.written if f[2] == 0x07][-1][4], 1)
        a.release_halt()
        a.stream_to(5, -175, 150)
        self.assertEqual(len([f for f in a._ser.written if f[2] == 0x03]), 2)

    def test_halt_with_unknown_position_sends_nothing(self):
        a = self.live_arm()
        a._ser.xyz = None
        a._ser.write = lambda b: a._ser.written.append(bytes(b))   # never answers a read
        a.commanded = (100.0, -200.0, 150.0)
        a.halt()
        self.assertTrue(a.inhibited)
        self.assertEqual([f for f in a._ser.written if f[2] == 0x03], [])   # no blind stop target

    def test_move_to_observes_abort_before_sending(self):
        a = self.live_arm()
        class Stop(Exception):
            pass
        def tick():
            raise Stop()
        a.tick = tick
        with self.assertRaises(Stop):
            a.move_to(0, -175, 150, 500)
        self.assertEqual([f for f in a._ser.written if f[2] == 0x03], [])

    def test_suction_state_tracked_from_driver_calls(self):
        a = self.live_arm()
        garm._bp.config.VENT_S = 0.0
        a.suction(True)
        self.assertTrue(a.gripping)
        a.vent()
        self.assertFalse(a.gripping)

    def test_grip_release_frames(self):
        a = self.live_arm()
        garm._bp.config.VENT_S = 0.0   # do not sleep in tests
        a.grip()
        a.release()
        nozzle = [f for f in a._ser.written if f[2] == 0x07]
        self.assertEqual([f[4] for f in nozzle], [1, 2, 3])   # pump on; pump off + vent; valve close
        self.assertFalse(a.gripping)


if __name__ == "__main__":
    unittest.main()
