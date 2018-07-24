MAX_MISORDER = 100


class JitterFrame:
    def __init__(self, data, timestamp):
        self.data = data
        self.timestamp = timestamp


class JitterBuffer:
    def __init__(self, capacity):
        assert capacity & (capacity - 1) == 0, 'capacity must be a power of 2'
        self._capacity = capacity
        self._origin = None
        self._packets = [None for i in range(capacity)]

    @property
    def capacity(self):
        return self._capacity

    def add(self, packet):
        if self._origin is None:
            self._origin = packet.sequence_number
        elif packet.sequence_number <= self._origin - MAX_MISORDER:
            self.remove(self.capacity)
            self._origin = packet.sequence_number
        elif packet.sequence_number < self._origin:
            return

        delta = packet.sequence_number - self._origin
        if delta >= 2 * self.capacity:
            # received packet is so far beyond capacity we cannot keep any
            # previous packets, so reset the buffer
            self.remove(self.capacity)
            self._origin = packet.sequence_number
        elif delta >= self.capacity:
            # remove just enough packets to fit the received packets
            excess = delta - self.capacity + 1
            self.remove(excess)

        pos = packet.sequence_number % self._capacity
        self._packets[pos] = packet

    def remove_frame(self):
        timestamp = None
        packets = []

        for count in range(self.capacity):
            pos = (self._origin + count) % self._capacity
            packet = self._packets[pos]
            if packet is None:
                break
            if timestamp is None:
                timestamp = packet.timestamp
            elif packet.timestamp != timestamp:
                self.remove(count)
                return JitterFrame(data=b''.join([x._data for x in packets]),
                                   timestamp=timestamp)
            packets.append(packet)

    def remove(self, count):
        assert count <= self._capacity
        for i in range(count):
            pos = self._origin % self._capacity
            self._packets[pos] = None
            self._origin += 1
