# MLEyeTrack
Docker version: https://github.com/MagicBOTAlex/DockeredMLEyeTrack

### Currently no UI
Depending on number of users, I'll consider making an UI \
If it's only me using this, then no UI is needed.

### Instructions
This is what is included in the .zip \
![image](https://github.com/user-attachments/assets/511be61c-f02e-433e-bf90-047a95435769) \
If you have DIY'ed eyetracking, then you definitely know how to use this software. \
If not, then you just need to drag and drop your unconverted models (.h5) into the `models` folder. \
Change the settings of `Settings.json`, then run the .exe and we gucci.

# "Benchmarks"
## Pros
- Lower latency
- Less CPU and GPU usage
- Not JavaScript based

##
- Currenly licensed under Babble's restrictive license
- Uses Python
- .exe + python + dependencies = 3.64GB unzipped
- No UI

Ryan's eyetracking uses ~13% CPU and ~23% GPU
My version uses ~1.5% CPU and ~5% GPU

### My version
![image](https://github.com/user-attachments/assets/2a5a465a-223c-4a6c-b35a-6afc56bb51e3)

### Ryan's vesion
![image](https://github.com/user-attachments/assets/260255cf-2490-441d-a89e-070d3733b340)


# Building from source
You need conda, but then it's as easy as running `build.bat` on windows. Linux is slightly different. \
You can refer to the [docker version](https://github.com/MagicBOTAlex/DockeredMLEyeTrack).

# Licensing
This project unfortunately is under Project Babble's restrictive license because of their MJPG Streamer. \
If somebody could make a replacement, then please do. If not, then this project will remain under their control/license.
