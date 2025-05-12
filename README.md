# DKS

This repository contains the files used in all different researchs done by Aymen and Laura at the DKS chair of the RUB.

## Beamforming, RIS and SDR

One of this first tasks corresponds to building a phased array beamformer, according to the instructions of [this video](https://www.youtube.com/watch?v=2QXKuEYR4Bw). The device that we're using at the moment is the same as in the video, which corresponds to an ADALM PLUTO SDR Rev C.

Before using the device at all, the drivers must be installed, which can be found on [this repo](https://github.com/analogdevicesinc/plutosdr-m2k-drivers-win/releases). To ensure that the device is being read after the installation, open the device's command prompt and type

```
ipconfig
```

this should show you your current internet network ip adress, as well as the network corresponding to Pluto, which normally has the same ip adress "192.168.2.1". If this appears, the drivers were succesfully installed and the device is ready to be used.

A few excercises to prove the correct functioning of the device were made and listed below.

### GNU Radio
A first example was made with GNU Radio. [This webpage](https://wiki.gnuradio.org/index.php/InstallingGR) has all the installation links of this program. After installing, open the program and then the file "Example1.grc", which is available at [this repository](https://github.com/lauravmorenoc/DKS/blob/main/GNU_Radio). It should look like this: 

![image](https://github.com/user-attachments/assets/611f4916-e8cd-44a3-aee4-6a65cfee6501)

This small block arrange attempts to send a 10k cosine signal centered at 2.4 GHz 

### Python and Github
In order to be able to use this repository and the SDR thorugh Pyhon, git must be installed from [here](https://git-scm.com/downloads) and python from [here](https://www.python.org/downloads/). After that, the corresponding libraries should be installed. The following instructions attempt to do that: 

* Open this [analog devices link](https://wiki.analog.com/resources/tools-software/linux-software/pyadi-iio?utm_source=chatgpt.com)
* Follow the first step which corresponds to downloading the LibIIO library. If the latest version doesn't have a windows installer, check for an older version.
* After installing, follow steps 3 and 4 (commands on the command prompt).
* Last, install all the libraries that the code needs to run. This are the ones that are necesary at the moment, however, if the code changes and more things are added, more might be required:
  ```
  pip install numpy
  pip install matplotlib
  ```

### Visual Studio Code
Download [Visual Studio Code](https://code.visualstudio.com/docs/?dv=win64user) or your preferred python code editor.

### Clone this repository
That was it! Now just clone this repository using the line

```
git clone https://github.com/lauravmorenoc/DKS/
```

on the git bash and open the python code editor to start using the files.

## NI USRP-2974
Quick guide to use the NI USRP-2974 as a remote SDR:

1. Connect the Ethernet cable to your PC directly or via a router/switch