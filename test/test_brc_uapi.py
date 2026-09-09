import struct
import unittest
from unittest import mock

import brc_uapi


class TestBRCUAPIEncoding(unittest.TestCase):
    def test_struct_sizes(self):
        self.assertEqual(brc_uapi.BRC_CREATE_SIZE, 24)
        self.assertEqual(brc_uapi.BRC_CONTROL_SIZE, 24)
        self.assertEqual(brc_uapi.BRC_RECLAIM_SIZE, 16)
        self.assertEqual(brc_uapi.BRC_CLASSIFY_SIZE, 48)

    def test_ioctl_numbers(self):
        self.assertEqual(brc_uapi.EXT4_IOC_BRC_TEST, 0x662D)
        self.assertEqual(brc_uapi.EXT4_IOC_BRC_CREATE, 0x4018662E)
        self.assertEqual(brc_uapi.EXT4_IOC_BRC_SEAL, 0x4018662F)
        self.assertEqual(brc_uapi.EXT4_IOC_BRC_SESSION_BEGIN, 0x6630)
        self.assertEqual(brc_uapi.EXT4_IOC_BRC_LINEAGE_BEGIN, 0x6631)
        self.assertEqual(brc_uapi.EXT4_IOC_BRC_RECLAIM_THROUGH, 0x40106632)
        self.assertEqual(brc_uapi.EXT4_IOC_BRC_CLASSIFY_PAIR, 0xC0306633)
        self.assertEqual(brc_uapi.EXT4_IOC_BRC_RECLAIM_PHYSICAL, 0x6634)
        self.assertEqual(brc_uapi.EXT4_IOC_BRC_LINEAGE_RESUME, 0x6635)


class TestBRCUAPIPacking(unittest.TestCase):
    def test_create_is_issued_on_child_fd(self):
        with mock.patch("brc_uapi.fcntl.ioctl", return_value=0) as ioctl:
            brc_uapi.create(child_fd=10, parent_fd=11, session_fd=12)

        fd, request, arg, mutate = ioctl.call_args.args

        self.assertEqual(fd, 10)
        self.assertEqual(request, brc_uapi.EXT4_IOC_BRC_CREATE)
        self.assertTrue(mutate)

        fields = struct.unpack("=iiI3I", arg)
        self.assertEqual(fields, (11, 12, 0, 0, 0, 0))

    def test_seal_is_issued_on_checkpoint_fd(self):
        with mock.patch("brc_uapi.fcntl.ioctl", return_value=0) as ioctl:
            brc_uapi.seal(checkpoint_fd=20, session_fd=21)

        fd, request, arg, mutate = ioctl.call_args.args

        self.assertEqual(fd, 20)
        self.assertEqual(request, brc_uapi.EXT4_IOC_BRC_SEAL)
        self.assertTrue(mutate)

        fields = struct.unpack("=iI4I", arg)
        self.assertEqual(fields, (21, 0, 0, 0, 0, 0))

    def test_lineage_begin_returns_session_fd(self):
        with mock.patch("brc_uapi.fcntl.ioctl", return_value=77) as ioctl:
            session_fd = brc_uapi.lineage_begin(30)

        self.assertEqual(session_fd, 77)
        ioctl.assert_called_once_with(
            30,
            brc_uapi.EXT4_IOC_BRC_LINEAGE_BEGIN,
        )

    def test_lineage_resume_returns_session_fd(self):
        with mock.patch("brc_uapi.fcntl.ioctl", return_value=88) as ioctl:
            session_fd = brc_uapi.lineage_resume(31)

        self.assertEqual(session_fd, 88)
        ioctl.assert_called_once_with(
            31,
            brc_uapi.EXT4_IOC_BRC_LINEAGE_RESUME,
        )

    def test_classify_pair_decodes_kernel_result(self):
        def fake_ioctl(fd, request, arg, mutate):
            self.assertEqual(fd, 40)
            self.assertEqual(request, brc_uapi.EXT4_IOC_BRC_CLASSIFY_PAIR)
            self.assertTrue(mutate)

            requested_generation = struct.unpack("=QQQQQII", arg)[0]

            arg[:] = struct.pack(
                "=QQQQQII",
                requested_generation,
                100,
                20,
                3,
                123,
                0,
                0,
            )
            return 0

        with mock.patch("brc_uapi.fcntl.ioctl", side_effect=fake_ioctl):
            result = brc_uapi.classify_pair(40, 5)

        self.assertEqual(result.generation, 5)
        self.assertEqual(result.shared_blocks, 100)
        self.assertEqual(result.dead_unique_blocks, 20)
        self.assertEqual(result.old_hole_blocks, 3)
        self.assertEqual(result.compared_blocks, 123)

    def test_reject_negative_generation(self):
        with self.assertRaises(ValueError):
            brc_uapi.reclaim_through(1, -1)

        with self.assertRaises(ValueError):
            brc_uapi.classify_pair(1, -1)


if __name__ == "__main__":
    unittest.main()
