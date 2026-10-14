Place the recording `EEG2501.BIN` here (it is not versioned).

Expected format: little-endian float32, 25 channels, 512 Hz, 15 s, multiplexed
(sample 1 of channels 1-25, then sample 2 of channels 1-25, ...), i.e.
`25 * 7680 * 4 = 768000` bytes.
