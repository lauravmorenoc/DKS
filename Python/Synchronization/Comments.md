## Comments on synchronization
The goal of this small project was to be able to detect the RIS by the phase offset it occasionates on the reflected signal. The planned steps go as follows:
1. Read and understand the synchronizations algorithms described [here](https://pysdr.org/content/sync.html)
2. Apply different types of offset to a simulated signal and try the previous mentioned sync functions
3. Adapt the code to use these functions with real live data acquired with the Pluto SDR
4. Prove sync between the two SDRs possible
5. Once the devices are synchronized, introduce the RIS into the setup and observe the added phase offset

### Task 1 and 2
All the sync functions provided [here](https://pysdr.org/content/sync.html) were tried and succesful results using simulated data and offsets were obtained. The codes are in this repository, more exactly following [this path](Python/Synchronization/Examples).

### Task 3 and 4
**Hint**: From these tasks onwards, the code needs to know the number of the usb port of each SDR. Use the command 'iio_info -s' on the console to find out.
The code was adapted and its able to synchronize two SDRs were the transmitter sends a BPSK signal (extension to QPSK or QAM is possible through modifications of the fine freq. sync. function). However, There are some comments on this task. Both the time sync. and fine sync. functions have parameters that need to be adjusted according to how fast do the freq. offset / phase offset changes. These two properties are dependent on a lot of factors, such as the devices themselves, environmental conditions, sampling frequency, etc. The ones provided in the code are the one that (after a lot of trying) gave the best results. However, there's still phase ambiguity, but that's normal for these kind of signals.
The code for this task is [here](Python/Synchronization/PlutoSDR/live_sync_twoSDRs.py).

### Task 5
This task was unfinished, and here's why: The fine freq. sync. function works with an algorithm called "Costas Loop". In this case, it corresponds to a second order algorithm used specifically for BPSK signals. This procedure focues on minimizing the error function value, which depends on the imaginary part of the samples. The bigger this part, the bigger the error. This makes the corrected samples to lie on the real axis (miminal error).

However, this also means that any phase offset provoked by the RIS would also be deleted. Many things were tried to avoid this problem, such as:

* Looking at the raw samples (without sync) and looking whether there was any noticeable offset change when changing the state of the RIS cells. This didn't give any results as the samples without any synchronizations are extremely messy, there was no pattern detected.
* Trying it anyway (even with the fine freq. sync. function). As expected, it was impossible to observe phase offset changes, the samples lie always on the real axis.
* Trying only applying coarse freq. sync. and time sync. This also didn't work. After the coarse freq. sync. and the time sync., we do get a cluster. However, it seems to rotate without any visible pattern (the phase seems to be chaning all the time even without any other estimulation).
* Trying to lock the initial synchronization and then keep working with this locked sync. instead of using both the freq. sync. functions. This was the longest step as it required to observe the behavior of the frequency offset and the algorithms that make the correction possible. For this observation, the last code used is [this one](Python/Synchronization/PlutoSDR/live_sync_SDRSandRIS.py). It has a visualization of how the variable that corrects the fine freq. offset oscillates in order to find the appropiate freq. offset. The plot also shows the mean, which is believed to be the fine freq. offset to be corrected from then onwards. The constellation plot is also there. After it seemed that the mean was quite stable (5-8 iterations), we created another code that holds the freq. offset conditions and keeps applying it to upcoming sample readings without applying the functions that delete the imaginary part. Sadly, when we tried it, we found that this procedure alone is not enough to provide stable synchronization. We do get just one cluster, but its phase changes on every reading, just like we weren't doing any fine freq. sync. This code is found [here](https://github.com/lauravmorenoc/DKS/blob/main/Python/Synchronization/PlutoSDR/test.py)
