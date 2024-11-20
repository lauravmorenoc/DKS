import adi
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(2.5e6)
sdr.rx()