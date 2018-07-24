from unittest import TestCase

from aiortc.jitterbuffer import JitterBuffer
from aiortc.rtp import RtpPacket


class JitterBufferTest(TestCase):
    def test_create(self):
        jbuffer = JitterBuffer(capacity=2)
        self.assertEqual(jbuffer._packets, [None, None])
        self.assertEqual(jbuffer._origin, None)

        jbuffer = JitterBuffer(capacity=4)
        self.assertEqual(jbuffer._packets, [None, None, None, None])
        self.assertEqual(jbuffer._origin, None)

    def test_add_ordered(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0002', sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0003', sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0004', sequence_number=4, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)

        self.assertIsNotNone(jbuffer._packets[0])
        self.assertEqual(jbuffer._packets[0].payload, b'0001')
        self.assertEqual(jbuffer._packets[0].sequence_number, 1)
        self.assertIsNotNone(jbuffer._packets[1])
        self.assertEqual(jbuffer._packets[1].payload, b'0002')
        self.assertEqual(jbuffer._packets[1].sequence_number, 2)
        self.assertIsNotNone(jbuffer._packets[2])
        self.assertEqual(jbuffer._packets[2].payload, b'0003')
        self.assertEqual(jbuffer._packets[2].sequence_number, 3)
        self.assertIsNotNone(jbuffer._packets[3])
        self.assertEqual(jbuffer._packets[3].payload, b'0004')
        self.assertEqual(jbuffer._packets[3].sequence_number, 4)

    def test_add_unordered(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0003', sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0002', sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)

        self.assertIsNotNone(jbuffer._packets[0])
        self.assertEqual(jbuffer._packets[0].payload, b'0001')
        self.assertEqual(jbuffer._packets[0].sequence_number, 1)
        self.assertIsNotNone(jbuffer._packets[1])
        self.assertEqual(jbuffer._packets[1].payload, b'0002')
        self.assertEqual(jbuffer._packets[1].sequence_number, 2)
        self.assertIsNotNone(jbuffer._packets[2])
        self.assertEqual(jbuffer._packets[2].payload, b'0003')
        self.assertEqual(jbuffer._packets[2].sequence_number, 3)
        self.assertIsNone(jbuffer._packets[3])

    def test_add_seq_too_low_drop(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'0002', sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 2)
        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 2)

        self.assertIsNotNone(jbuffer._packets[0])
        self.assertEqual(jbuffer._packets[0].payload, b'0002')
        self.assertEqual(jbuffer._packets[0].sequence_number, 2)
        self.assertIsNone(jbuffer._packets[1])
        self.assertIsNone(jbuffer._packets[2])
        self.assertIsNone(jbuffer._packets[3])

    def test_add_seq_too_low_reset(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'2000', sequence_number=2000, timestamp=1234))
        self.assertEqual(jbuffer._origin, 2000)
        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)

        self.assertIsNotNone(jbuffer._packets[0])
        self.assertEqual(jbuffer._packets[0].payload, b'0001')
        self.assertEqual(jbuffer._packets[0].sequence_number, 1)
        self.assertIsNone(jbuffer._packets[1])
        self.assertIsNone(jbuffer._packets[2])
        self.assertIsNone(jbuffer._packets[3])

    def test_add_seq_too_high_discard_one(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0002', sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0003', sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0004', sequence_number=4, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0005', sequence_number=5, timestamp=1234))
        self.assertEqual(jbuffer._origin, 2)

        self.assertIsNotNone(jbuffer._packets[0])
        self.assertEqual(jbuffer._packets[0].payload, b'0005')
        self.assertEqual(jbuffer._packets[0].sequence_number, 5)
        self.assertIsNotNone(jbuffer._packets[1])
        self.assertEqual(jbuffer._packets[1].payload, b'0002')
        self.assertEqual(jbuffer._packets[1].sequence_number, 2)
        self.assertIsNotNone(jbuffer._packets[2])
        self.assertEqual(jbuffer._packets[2].payload, b'0003')
        self.assertEqual(jbuffer._packets[2].sequence_number, 3)
        self.assertIsNotNone(jbuffer._packets[3])
        self.assertEqual(jbuffer._packets[3].payload, b'0004')
        self.assertEqual(jbuffer._packets[3].sequence_number, 4)

    def test_add_seq_too_high_discard_four(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0002', sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0003', sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0004', sequence_number=4, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0008', sequence_number=8, timestamp=1234))
        self.assertEqual(jbuffer._origin, 5)

        self.assertIsNone(jbuffer._packets[0])
        self.assertIsNone(jbuffer._packets[1])
        self.assertIsNone(jbuffer._packets[2])
        self.assertIsNotNone(jbuffer._packets[3])
        self.assertEqual(jbuffer._packets[3].payload, b'0008')
        self.assertEqual(jbuffer._packets[3].sequence_number, 8)

    def test_add_seq_too_high_discard_more(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0002', sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0003', sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0004', sequence_number=4, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'0009', sequence_number=9, timestamp=1234))
        self.assertEqual(jbuffer._origin, 9)

        self.assertIsNotNone(jbuffer._packets[0])
        self.assertEqual(jbuffer._packets[0].payload, b'0009')
        self.assertEqual(jbuffer._packets[0].sequence_number, 9)
        self.assertIsNone(jbuffer._packets[1])
        self.assertIsNone(jbuffer._packets[2])
        self.assertIsNone(jbuffer._packets[3])

    def test_add_seq_too_high_reset(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 1)
        jbuffer.add(RtpPacket(payload=b'3002', sequence_number=3002, timestamp=1234))
        self.assertEqual(jbuffer._origin, 3002)

        self.assertIsNotNone(jbuffer._packets[0])
        self.assertEqual(jbuffer._packets[0].payload, b'3002')
        self.assertEqual(jbuffer._packets[0].sequence_number, 3002)
        self.assertIsNone(jbuffer._packets[1])
        self.assertIsNone(jbuffer._packets[2])
        self.assertIsNone(jbuffer._packets[3])

    def test_peek(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        jbuffer.add(RtpPacket(payload=b'0002', sequence_number=2, timestamp=1234))
        jbuffer.add(RtpPacket(payload=b'0004', sequence_number=4, timestamp=1234))

        packet = jbuffer.peek(0)
        self.assertIsNotNone(packet)
        self.assertEqual(packet.sequence_number, 1)

        packet = jbuffer.peek(1)
        self.assertIsNotNone(packet)
        self.assertEqual(packet.sequence_number, 2)

        packet = jbuffer.peek(2)
        self.assertIsNone(packet)

        packet = jbuffer.peek(3)
        self.assertIsNotNone(packet)
        self.assertEqual(packet.sequence_number, 4)

        with self.assertRaises(IndexError) as cm:
            jbuffer.peek(4)
        self.assertEqual(str(cm.exception), 'Cannot peek at offset 4, capacity is 4')

    def test_remove(self):
        jbuffer = JitterBuffer(capacity=4)

        jbuffer.add(RtpPacket(payload=b'0001', sequence_number=1, timestamp=1234))
        jbuffer.add(RtpPacket(payload=b'0002', sequence_number=2, timestamp=1234))
        jbuffer.add(RtpPacket(payload=b'0003', sequence_number=3, timestamp=1234))
        jbuffer.add(RtpPacket(payload=b'0004', sequence_number=4, timestamp=1234))

        # remove 1 packet
        packets = jbuffer.remove(1)
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].sequence_number, 1)

        # check buffer
        self.assertEqual(jbuffer._head, 1)
        self.assertEqual(jbuffer._origin, 2)
        self.assertIsNone(jbuffer._packets[0])
        self.assertIsNotNone(jbuffer._packets[1])
        self.assertIsNotNone(jbuffer._packets[2])
        self.assertIsNotNone(jbuffer._packets[3])

        # remove 2 packets
        packets = jbuffer.remove(2)
        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0].sequence_number, 2)
        self.assertEqual(packets[1].sequence_number, 3)

        # check buffer
        self.assertEqual(jbuffer._head, 3)
        self.assertEqual(jbuffer._origin, 4)
        self.assertIsNone(jbuffer._packets[0])
        self.assertIsNone(jbuffer._packets[1])
        self.assertIsNone(jbuffer._packets[2])
        self.assertIsNotNone(jbuffer._packets[3])
