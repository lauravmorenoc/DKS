# DKS

This repository contains the files and procedurements made in order to build a phased array beamformer, according to the instructions of [this video](https://www.youtube.com/watch?v=2QXKuEYR4Bw). The device that we're using at the moment is the same as in the video, which corresponds to an ADALM PLUTO SDR Rev C.

Before using the device at all, the drivers must be installed, which can be found on [this repo](https://github.com/analogdevicesinc/plutosdr-m2k-drivers-win/releases). To ensure that the device is being read after the installation, open the device's command prompt and type

```
ipconfig
```

this should show you your current internet network ip adress, as well as the network corresponding to Pluto, which normally has the same ip adress "192.168.2.1". If this appears, the drivers were succesfully installed and the device is ready to be used.

A few excercises to prove the correct functioning of the device were made and listed below.
