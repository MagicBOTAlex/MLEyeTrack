# MLEyeTrack
Docker version: https://github.com/MagicBOTAlex/DockeredMLEyeTrack

### Currently no UI
Depending on number of users, I'll consider making an UI \
If it's only me using this, then no UI is needed.

### Recommendation: You should know a bit of python to use this. If you want a slightly easier non-python based, then go to [Ryan's](https://github.com/ryan9411vr/EyeTracking/) (It is JavaScript based 💀)

# ⚠️ The .exe has not been fully tested yet, and I need testers to finish it. Currently, only the python and docker version is confirmed to work.
Python and docker is confirmed working because I regularly use it. (If you're lucky, you'll find me at the great pug once a week) \
Else you can find me at Ryan's discord: https://discord.gg/QTyU4eNKrv

## Instructions
This is what is included in the .zip \
![image](https://github.com/user-attachments/assets/511be61c-f02e-433e-bf90-047a95435769) \
If you have DIY'ed eyetracking, then you definitely know how to use this software. \
If not, then you just need to drag and drop your unconverted models (.h5) into the `models` folder. \
These models are only V1 of Ryan's models. You still have to use [Ryan's](https://github.com/ryan9411vr/EyeTracking/) software to train the models. My software only provides a new engine to run the models.
Change the settings of `Settings.json`, then run the .exe and we gucci.

# Comparison
<table>
  <tr>
    <td>
          <h2>ETVR's Algo with LEAP fully calibrated</h2>
      <a href="https://github.com/user-attachments/assets/633e0539-c6a6-44c1-bbe9-a47a0082e21b">
        <video src="https://github.com/user-attachments/assets/633e0539-c6a6-44c1-bbe9-a47a0082e21b" alt="Video 1" width="300">
      </a>
    </td>
    <td>
          <h2>Ryan's eyetrack (Calibrated 2 months ago)</h2>
      <a href="https://github.com/user-attachments/assets/67c03609-f381-452a-952f-5274c6105fe9">
        <video src="https://github.com/user-attachments/assets/67c03609-f381-452a-952f-5274c6105fe9" alt="Video 2" width="300">
      </a>
    </td>
    <td>
          <h2>MLEyetrack (Calibrated 2 months ago)</h2>
      <a href="https://github.com/user-attachments/assets/2d0c061c-ffa8-4ea9-98f0-1c68f74040d2">
        <video src="https://github.com/user-attachments/assets/2d0c061c-ffa8-4ea9-98f0-1c68f74040d2" alt="Video 3" width="300">
      </a>
    </td>
  </tr>
</table>

# "Benchmarks"
## Pros
- Lower latency
- ONNX based (Less GPU/CPU per infrence)
- Not JavaScript based

## Cons
- Currenly licensed under Babble's restrictive license
- Uses Python
- .exe + python + CUDA + dependencies = BIG .EXE
- No UI

<table>
  <tr>
    <td>
          <h2>MLEyeTrack (old)</h2>
      <a href="https://github.com/user-attachments/assets/2a5a465a-223c-4a6c-b35a-6afc56bb51e3">
        <img src="https://github.com/user-attachments/assets/2a5a465a-223c-4a6c-b35a-6afc56bb51e3" alt="MLEyeTrack cpu/gpu usage" width="300">
      </a>
    </td>
    <td>
          <h2>Ryan's eyetrack</h2>
      <a href="https://github.com/user-attachments/assets/260255cf-2490-441d-a89e-070d3733b340">
        <img src="https://github.com/user-attachments/assets/260255cf-2490-441d-a89e-070d3733b340" alt="Ryan's cpu/gpu usage" width="300">
      </a>
    </td>
  </tr>
</table>

# Building from source
You need conda, but then it's as easy as running `build.bat` on windows. Linux is slightly different. \
You can refer to the [docker version](https://github.com/MagicBOTAlex/DockeredMLEyeTrack).

# Licensing
Two scripts are unfortunately licensed under Project Babble's restrictive license because of their MJPG Streamer. \
If somebody could make a replacement, then please do. If not, then this project will remain under their control/license. \
The rest idk, ask me on Ryan's Disocrd.
